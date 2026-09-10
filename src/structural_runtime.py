"""Teacher targets and optional mmap cache, isolated from student state_dict."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from src.dataset import TeacherFeatureDataset
from src.local_features import FinalPatches, project_patches, spatial_structure_loss
from src.structural_config import anchors_needed
from src.structural_losses import signature, structure, transport_loss


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def anchor_sentences(classes, mode):
    names = [c.replace("_", " ") for c in classes]
    phrases = [f"a {c}." for c in names]
    if mode == "all":
        phrases += [f"a photo of a {c}." for c in names]
        phrases += [f"a sketch of a {c}." for c in names]
    return phrases


class TargetStore:
    """Complete-manifest mmap storage; a partial build is never reused."""
    def __init__(self, directory, metadata, count, tokens, anchors):
        self.directory = Path(directory)
        self.metadata, self.count, self.tokens, self.anchors = metadata, count, tokens, anchors
        self.shapes = {"structure": (count, tokens, tokens)}
        if anchors:
            self.shapes["semantic"] = (count, tokens, anchors)
        self.arrays = {}

    @property
    def nbytes(self):
        return sum(int(np.prod(s))*2 for s in self.shapes.values())

    def open(self):
        manifest = self.directory / "manifest.json"
        if not manifest.is_file():
            return False
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != self.metadata:
            raise RuntimeError("Local target cache metadata mismatch; choose another directory.")
        for key, shape in self.shapes.items():
            path = self.directory / (key+".bin")
            if not path.is_file() or path.stat().st_size != int(np.prod(shape))*2:
                raise RuntimeError("Incomplete local target cache: "+str(path))
            self.arrays[key] = np.memmap(path, mode="r", dtype=np.float16, shape=shape)
        return True

    def build(self, batches):
        self.directory.mkdir(parents=True, exist_ok=True)
        reserve = 2*1024**3
        if shutil.disk_usage(self.directory).free < self.nbytes + reserve:
            raise RuntimeError(
                f"Local targets need {self.nbytes/1024**3:.2f} GiB plus 2 GiB reserve. "
                "Use --local_target_mode online or a smaller teacher_patch_grid."
            )
        writers = {k: np.memmap(self.directory/(k+".tmp"), mode="w+", dtype=np.float16, shape=s)
                   for k, s in self.shapes.items()}
        offset = 0
        try:
            for values in batches:
                size = values["structure"].shape[0]
                if offset+size > self.count:
                    raise ValueError("Too many teacher cache rows.")
                for key in writers:
                    value = values[key].detach().float().cpu().numpy()
                    if tuple(value.shape[1:]) != self.shapes[key][1:] or not np.isfinite(value).all():
                        raise ValueError("Invalid local target tensor: "+key)
                    writers[key][offset:offset+size] = value
                offset += size
            if offset != self.count:
                raise ValueError("Teacher cache row count mismatch.")
            for value in writers.values():
                value.flush()
                value._mmap.close()
            for key in self.shapes:
                os.replace(self.directory/(key+".tmp"), self.directory/(key+".bin"))
            temporary = self.directory/"manifest.tmp"
            temporary.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
            os.replace(temporary, self.directory/"manifest.json")
        finally:
            for value in writers.values():
                if not value._mmap.closed:
                    value._mmap.close()
        self.open()

    def get(self, indices, device):
        ix = indices.detach().cpu().numpy()
        if (ix < 0).any() or (ix >= self.count).any():
            raise ValueError("Local target index outside cache.")
        return {k: torch.from_numpy(np.array(v[ix], copy=True)).to(device=device, dtype=torch.float32)
                for k, v in self.arrays.items()}


class StructuralRuntime:
    def __init__(self, student, dataset, args):
        self.args = args
        self.teacher = self.prompts = self.store = None
        self.report_written = set()
        self.student_anchors = self.teacher_anchors = None
        self.sketch_count = len(dataset.all_sketches_path)
        self.local_active = args.lambda_sfgw > 0
        self.local_semantic = self.local_active and args.local_matching == "fgw" and args.fgw_alpha < 1
        self.grid = args.spatial_grid if args.local_matching == "spatial" else args.teacher_patch_grid
        self.sentences = anchor_sentences(dataset.all_categories, args.semantic_anchors)
        self._load_teacher(student, dataset)
        if anchors_needed(args):
            from clip import clip
            tokens = clip.tokenize(self.sentences).to(next(student.clip_model.parameters()).device)
            with torch.no_grad():
                self.student_anchors = student.clip_model.encode_text(tokens).float().detach()
                tokenizer = self.teacher.text_tokenizer
                self.teacher_anchors = torch.cat([
                    self.teacher.encode_text(tokenizer(self.sentences[i:i+32]).to(
                        next(self.teacher.parameters()).device)).float().detach()
                    for i in range(0, len(self.sentences), 32)
                ])
        if self.local_active:
            teacher_grid = int((self.teacher.visual.positional_embedding.shape[0]-1)**0.5)
            student_grid = int((student.clip_model.visual.positional_embedding.shape[0]-1)**0.5)
            if self.grid > teacher_grid or (args.local_matching == "spatial" and self.grid > student_grid):
                raise ValueError("Requested patch grid would upsample the teacher/student.")
            if args.local_target_mode == "cache":
                self._prepare_store(student, dataset)
                self.teacher = self.prompts = None
        else:
            self.teacher = self.prompts = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_teacher(self, student, dataset):
        from src.model import _load_teacher, _build_teacher_prompts
        args = copy.copy(self.args)
        args.teacher_cache_path = ""
        args.rebuild_teacher_cache = True
        self.teacher = _load_teacher(args)
        if self.teacher is None:
            raise RuntimeError("The enabled distillation objective requires DFN.")
        # Anchor-only and alpha<1 configurations also need text tokenization.
        import open_clip
        self.teacher.text_tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
        self.prompts = _build_teacher_prompts(args, self.teacher)
        self.teacher_identity = {"type": "raw_dfn5b"}
        if args.teacher_pretrain_epochs > 0:
            path = self.args.teacher_cache_path
            if not path or not Path(path).is_file():
                raise RuntimeError("Tuned local targets require the saved main teacher cache.")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected = student._teacher_cache_metadata(dataset)
            if payload.get("metadata") != expected:
                raise RuntimeError("Teacher metadata changed before structural target construction.")
            state = payload.get("teacher_prompt_state_dict")
            if not state:
                raise RuntimeError("Teacher cache is missing tuned visual prompts.")
            self.prompts.load_state_dict(state, strict=True)
            self.prompts.eval().requires_grad_(False)
            self.teacher_identity = {"type": "tuned_dfn5b", "global_cache_sha256": file_hash(path)}
        elif self.args.teacher_cache_path:
            raise ValueError("Raw local teacher cannot be combined with an unspecified tuned cache.")
        self.teacher.eval().requires_grad_(False)

    @torch.no_grad()
    def targets(self, images, modality):
        visual = self.teacher.visual
        dtype, device = visual.conv1.weight.dtype, visual.conv1.weight.device
        parts = []
        for start in range(0, len(images), self.args.local_teacher_batch_size):
            with FinalPatches(visual, visual.transformer.batch_first, self.grid) as capture:
                chunk = images[start:start+self.args.local_teacher_batch_size].to(device=device, dtype=dtype)
                if self.prompts is None:
                    self.teacher.encode_image(chunk)
                else:
                    self.prompts(chunk, modality)
            patches = capture.values[0]
            with torch.autocast(device_type=device.type, enabled=False):
                values = {"structure": structure(patches)}
                if self.local_semantic:
                    values["semantic"] = signature(project_patches(patches, visual), self.teacher_anchors)
            parts.append(values)
        return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}

    def _prepare_store(self, student, dataset):
        import open_clip
        metadata = dict(
            format_version=1, teacher=self.teacher_identity,
            open_clip_version=getattr(open_clip, "__version__", "unknown"),
            torch_version=str(torch.__version__),
            global_metadata=student._teacher_cache_metadata(dataset),
            sentences=self.sentences if self.local_semantic else [],
            grid=self.grid, tokens=self.grid**2, normalization="raw-final-block-cosine-v1",
            semantic_projection="pooled-final-token-ln-post-proj-cosine-v1",
            local_semantic=self.local_semantic,
            count=self.sketch_count+len(dataset.all_photo_paths),
        )
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
        self.store = TargetStore(Path(self.args.local_cache_dir)/key, metadata,
                                 metadata["count"], self.grid**2,
                                 len(self.sentences) if self.local_semantic else 0)
        print(f"[Local cache] {self.store.directory}; {self.store.nbytes/1024**3:.2f} GiB")
        if self.store.open():
            print("[Local cache] reusing complete targets")
            return
        def batches():
            from torch.utils.data import DataLoader
            for modality, paths in (("sketch", dataset.all_sketches_path),
                                    ("photo", dataset.all_photo_paths)):
                loader = DataLoader(TeacherFeatureDataset(paths, dataset.max_size),
                                    batch_size=self.args.local_teacher_batch_size,
                                    num_workers=self.args.workers, shuffle=False)
                for index, images in enumerate(loader):
                    yield self.targets(images, modality)
                    if index % 100 == 0:
                        print(f"[Local cache] {modality}: {index+1}/{len(loader)}", flush=True)
        self.store.build(batches())

    def local_loss(self, student, batch, captures):
        args = self.args
        total = batch[0].new_zeros((), dtype=torch.float32)
        logs = {}
        for modality, image_index, capture_index, cache_column in (
            ("photo", 0, 0, 1), ("sketch", 1, 1, 0)
        ):
            patches = captures[capture_index]
            if self.store is None:
                targets = self.targets(batch[image_index], modality)
            else:
                indices = batch[5][:, cache_column]
                if modality == "photo":
                    indices = indices + self.sketch_count
                targets = self.store.get(indices, patches.device)
            with torch.autocast(device_type=patches.device.type, enabled=False):
                if args.local_matching == "spatial":
                    value = spatial_structure_loss(targets["structure"], patches)
                    parts = {}
                else:
                    semantic = signature(project_patches(patches, student.clip_model.visual),
                                         self.student_anchors.to(patches.device)) if self.local_semantic else None
                    value, parts = transport_loss(
                        targets["structure"], patches, targets.get("semantic"), semantic,
                        args.fgw_alpha, args.transport_epsilon, args.gw_iterations,
                        args.sinkhorn_iterations, args.transport_tolerance,
                        return_plan=bool(getattr(args, "local_report_dir", ""))
                                    and modality not in self.report_written,
                    )
                    if "_plan" in parts:
                        from src.structural_report import save_transport_report
                        plan = parts.pop("_plan")
                        save_transport_report(args.local_report_dir, modality,
                                              batch[image_index][0], targets["structure"][0],
                                              structure(patches.detach())[0], plan[0],
                                              self.teacher_identity)
                        self.report_written.add(modality)
            total = total + value/2
            for key, v in parts.items():
                logs[modality+"_"+key] = v
        logs["sfgw"] = total.detach()
        return args.lambda_sfgw*total, logs
