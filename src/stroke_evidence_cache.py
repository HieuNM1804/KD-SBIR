"""Prepare compact teacher targets for retrieval-conditioned stroke evidence KD."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.dataset import load_image, normal_transform, sample_seed
from src.stroke_evidence import (
    TARGET_NAMES,
    FinalBlockInputCapture,
    cls_patch_attention,
    erase_by_patch_evidence,
    evidence_entropy,
    normalize_evidence,
    patch_ink_mass,
    patch_residuals,
    projected_patch_features,
    resize_evidence,
)

CACHE_FORMAT_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def add_arguments(parser):
    parser.add_argument("--retrieval_head", choices=("main", "rsed"), default="main")
    parser.add_argument("--lambda_rsed", type=float, default=0.0)
    parser.add_argument("--lambda_rsed_where", type=float, default=0.10)
    parser.add_argument("--lambda_rsed_what", type=float, default=0.25)
    parser.add_argument("--lambda_rsed_effect", type=float, default=0.10)
    parser.add_argument("--lambda_rsed_anchor", type=float, default=0.20)
    parser.add_argument("--rsed_effect_magnitude_weight", type=float, default=0.0)
    parser.add_argument("--rsed_target", choices=TARGET_NAMES + ("shuffled",), default="retrieval")
    parser.add_argument("--rsed_beta", type=float, default=0.10)
    parser.add_argument("--rsed_bottleneck", type=int, default=128)
    parser.add_argument("--rsed_head_lr", type=float, default=1e-3)
    parser.add_argument("--rsed_temperature", type=float, default=0.07)
    parser.add_argument("--rsed_graph_steps", type=int, default=1)
    parser.add_argument("--rsed_graph_mix", type=float, default=0.25)
    parser.add_argument("--rsed_mask_fraction", type=float, default=0.10)
    parser.add_argument("--rsed_ink_threshold", type=float, default=0.08)
    parser.add_argument("--rsed_ink_softness", type=float, default=0.12)
    parser.add_argument("--rsed_teacher_batch_size", type=int, default=4)
    parser.add_argument("--rsed_cache_path", type=str, default="")
    parser.add_argument("--rsed_prepare_only", action="store_true")
    parser.add_argument("--rsed_warmup_epochs", type=float, default=0.5)
    parser.add_argument("--rsed_decay_start_epoch", type=float, default=3.0)
    parser.add_argument("--rsed_diagnostics", action="store_true")
    parser.add_argument("--rsed_diagnostic_batch_size", type=int, default=32)
    parser.add_argument("--rsed_diagnostic_examples", type=int, default=8)


def validate_arguments(parser, args):
    finite_nonnegative = (
        args.lambda_rsed,
        args.lambda_rsed_where,
        args.lambda_rsed_what,
        args.lambda_rsed_effect,
        args.lambda_rsed_anchor,
        args.rsed_effect_magnitude_weight,
        args.rsed_beta,
        args.rsed_head_lr,
        args.rsed_warmup_epochs,
        args.rsed_decay_start_epoch,
    )
    if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative):
        parser.error("RSED weights, beta and schedule values must be finite and nonnegative")
    if args.retrieval_head == "main" and (args.lambda_rsed > 0 or args.rsed_prepare_only):
        parser.error("RSED preparation/training requires --retrieval_head rsed")
    if args.retrieval_head == "rsed" and args.lambda_rsed <= 0 and not args.rsed_prepare_only:
        parser.error("Use a positive --lambda_rsed for the RSED head, or select --retrieval_head main")
    if args.lambda_rsed > 0 and sum((args.lambda_rsed_where, args.lambda_rsed_what,
                                     args.lambda_rsed_effect, args.lambda_rsed_anchor)) <= 0:
        parser.error("At least one RSED component weight must be positive")
    if args.rsed_bottleneck < 1 or args.rsed_teacher_batch_size < 1:
        parser.error("RSED bottleneck and teacher batch size must be positive")
    if args.rsed_graph_steps < 0 or not 0 <= args.rsed_graph_mix <= 1:
        parser.error("RSED graph steps/mix are out of range")
    if not 0 < args.rsed_temperature or not 0 < args.rsed_mask_fraction < 1:
        parser.error("RSED temperature must be positive and mask fraction must be in (0,1)")
    if not 0 <= args.rsed_ink_threshold < 1 or not 0 < args.rsed_ink_softness <= 1:
        parser.error("RSED ink threshold/softness are out of range")
    if args.rsed_diagnostic_batch_size < 2 or args.rsed_diagnostic_examples < 0:
        parser.error("Invalid RSED diagnostic sizes")
    if (args.lambda_rsed > 0 or args.rsed_prepare_only) and args.teacher_pretrain_epochs < 1:
        parser.error("RSED requires the prompt-tuned teacher cache; keep --teacher_pretrain_epochs >= 1")


class IndexedSketchDataset(Dataset):
    def __init__(self, paths, labels, size):
        self.paths = paths
        self.labels = labels
        self.transform = normal_transform(size)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index], self.size)), index, self.labels[index]


def _path_fingerprint(paths, root):
    digest = hashlib.sha256()
    for path in paths:
        relative = os.path.relpath(path, root).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _photo_labels(dataset):
    return [dataset.category_to_label[Path(path).parent.name] for path in dataset.all_photo_paths]


def _sketch_labels(dataset):
    return [dataset.category_to_label[Path(path).parent.name] for path in dataset.all_sketches_path]


def _class_prototypes(features, labels, class_count):
    features = F.normalize(features.float(), dim=-1)
    labels = torch.as_tensor(labels)
    result = []
    for label in range(class_count):
        selected = features[labels == label]
        if len(selected) == 0:
            raise ValueError(f"No teacher photos for seen class {label}")
        result.append(F.normalize(selected.mean(0), dim=-1))
    return torch.stack(result)


def _random_map(ink, seed):
    rows = []
    for offset in range(len(ink)):
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        noise = torch.rand(ink.shape[1], generator=generator).to(ink.device)
        rows.append(normalize_evidence(noise[None], ink[offset : offset + 1])[0])
    return torch.stack(rows)


def _target_metadata(args, dataset, teacher_path, teacher_metadata):
    source_root = Path(__file__).resolve().parent.parent
    sources = (
        "clip/model.py",
        "src/dataset.py",
        "src/model.py",
        "src/losses.py",
        "src/teacher_prompts.py",
        "src/stroke_evidence.py",
        "src/stroke_evidence_cache.py",
    )
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "method": "retrieval_conditioned_stroke_evidence_distillation",
        "definition": (
            "positive_class_prototype_gradient_x_residual_times_cls_attention;"
            "ink_normalized_graph_diffused;causal_student_evidence_pooling"
        ),
        "dataset": args.dataset,
        "max_size": dataset.max_size,
        "classnames": list(dataset.all_categories),
        "sketch_count": len(dataset.all_sketches_path),
        "photo_count": len(dataset.all_photo_paths),
        "sketch_fingerprint": _path_fingerprint(dataset.all_sketches_path, args.root),
        "photo_fingerprint": _path_fingerprint(dataset.all_photo_paths, args.root),
        "teacher_cache_sha256": file_sha256(teacher_path),
        "teacher_metadata": teacher_metadata,
        "teacher_grid": args.max_size // 14,
        "student_grid": args.rsed_student_grid,
        "targets": list(TARGET_NAMES),
        "mask_fraction": args.rsed_mask_fraction,
        "ink_threshold": args.rsed_ink_threshold,
        "ink_softness": args.rsed_ink_softness,
        "graph_steps": args.rsed_graph_steps,
        "graph_mix": args.rsed_graph_mix,
        "teacher_batch_size": args.rsed_teacher_batch_size,
        "seed": args.seed,
        "source_sha256": {name: file_sha256(source_root / name) for name in sources},
    }


def _cache_path(args, metadata, teacher_path):
    if args.rsed_cache_path:
        return Path(args.rsed_cache_path)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:16]
    return teacher_path.parent / f"{args.dataset}_rsed_{key}.pt"


def validate_payload(payload, metadata):
    if payload.get("metadata") != metadata:
        raise ValueError("RSED cache metadata differs; use a new --rsed_cache_path")
    count = metadata["sketch_count"]
    variants = len(TARGET_NAMES)
    grid = metadata["student_grid"] ** 2
    width = metadata["teacher_metadata"]["teacher_output_dim"]
    shapes = {
        "maps": (count, variants, grid),
        "teacher_evidence": (count, variants, width),
        "teacher_masked": (count, variants, width),
        "confidence": (count, variants),
        "removed_ink_fraction": (count, variants),
        "teacher_positive_score": (count,),
        "positive_relevance_fraction": (count,),
        "target_entropy": (count, variants),
        "clean_cache_cosine": (count,),
    }
    for name, shape in shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid RSED tensor {name}: expected {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite RSED tensor: {name}")
    for name in ("maps", "teacher_evidence", "teacher_masked"):
        if payload[name].dtype != torch.float16:
            raise ValueError(f"RSED {name} must be float16")
    if not torch.allclose(payload["maps"].float().sum(-1), torch.ones(count, variants), atol=2e-3):
        raise ValueError("RSED maps are not normalized")
    if (payload["confidence"] <= 0).any() or (payload["confidence"] > 1).any():
        raise ValueError("RSED confidence must lie in (0,1]")
    if (payload["removed_ink_fraction"] - metadata["mask_fraction"]).abs().max() > 0.025:
        raise ValueError("RSED erasure does not respect the configured ink budget")


def prepare_cache(args, dataset, report_dir):
    if not args.teacher_cache_path:
        raise ValueError("RSED requires --teacher_cache_path")
    teacher_path = Path(args.teacher_cache_path)
    if not teacher_path.is_file():
        raise FileNotFoundError("Prepare the main prompt-tuned teacher cache before RSED")
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_metadata = teacher_payload.get("metadata", {})
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION
    if (teacher_metadata.get("format_version") != TEACHER_CACHE_FORMAT_VERSION or
            teacher_metadata.get("teacher_model") != DFN5B_MODEL or
            teacher_metadata.get("teacher_pretrained") != DFN5B_PRETRAINED or
            not teacher_payload.get("teacher_prompt_state_dict")):
        raise ValueError("RSED needs the same prompt-tuned DFN5B teacher used by main")
    if args.max_size % 14:
        raise ValueError("DFN5B target extraction requires max_size divisible by 14")
    metadata = _target_metadata(args, dataset, teacher_path, teacher_metadata)
    path = _cache_path(args, metadata, teacher_path)
    args.rsed_cache_path = str(path)
    args.rsed_target_metadata = metadata
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(payload, metadata)
        print("[RSED Cache] reused; teacher target extraction skipped:", path, flush=True)
    else:
        estimate = len(dataset) * len(TARGET_NAMES) * (2 * 1024 * 2 + metadata["student_grid"] ** 2 * 2 + 24)
        free = shutil.disk_usage(path.parent if path.parent.exists() else teacher_path.parent).free
        if free < estimate + 2 * 1024**3:
            raise OSError(f"RSED cache needs about {estimate / 1024**2:.1f} MiB plus 2 GiB reserve")
        started = time.perf_counter()
        import open_clip
        from src.teacher_prompts import TeacherPromptController
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = open_clip.create_model(
            DFN5B_MODEL,
            pretrained=DFN5B_PRETRAINED,
            precision="fp16" if device.type == "cuda" else "fp32",
            device=device,
        ).eval().requires_grad_(False)
        controller = TeacherPromptController(
            teacher.visual,
            teacher_metadata["teacher_n_ctx_visual"],
            teacher_metadata["teacher_prompt_depth"],
            teacher_metadata["teacher_prompt_std"],
            teacher_metadata["teacher_prompt_seed"],
        )
        controller.load_state_dict(teacher_payload["teacher_prompt_state_dict"], strict=True)
        controller.eval().requires_grad_(False)
        teacher_grid = metadata["teacher_grid"]
        if teacher.visual.positional_embedding.shape[0] - 1 != teacher_grid**2:
            raise ValueError("Unexpected DFN5B patch geometry")
        student_grid = metadata["student_grid"]
        sketch_labels = _sketch_labels(dataset)
        photo_labels = _photo_labels(dataset)
        prototypes = _class_prototypes(
            teacher_payload["teacher_photo_features"], photo_labels, len(dataset.all_categories)
        ).to(device)
        count = len(dataset)
        variants = len(TARGET_NAMES)
        width = teacher_payload["teacher_sketch_features"].shape[1]
        payload = {
            "metadata": metadata,
            "maps": torch.empty(count, variants, student_grid**2, dtype=torch.float16),
            "teacher_evidence": torch.empty(count, variants, width, dtype=torch.float16),
            "teacher_masked": torch.empty(count, variants, width, dtype=torch.float16),
            "confidence": torch.ones(count, variants),
            "removed_ink_fraction": torch.empty(count, variants),
            "teacher_positive_score": torch.empty(count),
            "positive_relevance_fraction": torch.empty(count),
            "target_entropy": torch.empty(count, variants),
            "clean_cache_cosine": torch.empty(count),
        }
        loader = DataLoader(
            IndexedSketchDataset(dataset.all_sketches_path, sketch_labels, dataset.max_size),
            batch_size=args.rsed_teacher_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        dtype = teacher.visual.conv1.weight.dtype
        for images, indices, labels in tqdm(loader, desc="[RSED Cache] retrieval-conditioned sketch evidence", mininterval=5):
            images = images.to(device=device, dtype=dtype, non_blocking=True).detach().requires_grad_(True)
            labels = labels.to(device)
            with torch.enable_grad(), FinalBlockInputCapture(teacher.visual) as capture:
                clean = controller(images, "sketch")
                clean_normalized = F.normalize(clean.float(), dim=-1)
                score = (clean_normalized * prototypes[labels]).sum(-1)
            sequence = capture.residual()
            gradient = torch.autograd.grad(score.sum(), sequence, retain_graph=False)[0]
            patches = patch_residuals(teacher.visual, sequence)
            patch_gradient = patch_residuals(teacher.visual, gradient)
            signed = (patches.float() * patch_gradient.float()).sum(-1)
            positive = signed.clamp_min(0)
            absolute = signed.abs()
            positive_fraction = positive.sum(-1) / absolute.sum(-1).clamp_min(1e-8)
            relevance = torch.where(positive.sum(-1, keepdim=True) > 1e-8, positive, absolute)
            attention = cls_patch_attention(teacher.visual, sequence).detach()
            teacher_ink = patch_ink_mass(images.detach(), teacher_grid,
                                         args.rsed_ink_threshold, args.rsed_ink_softness)
            retrieval_map = normalize_evidence(relevance * attention.clamp_min(1e-12).sqrt(), teacher_ink)
            attention_map = normalize_evidence(attention, teacher_ink)
            random_map = _random_map(teacher_ink, args.seed + 9100 + int(indices[0]) * 37)
            maps_teacher = torch.stack((retrieval_map, attention_map, random_map), dim=1)
            student_ink = patch_ink_mass(images.detach(), student_grid,
                                         args.rsed_ink_threshold, args.rsed_ink_softness)
            maps_student = torch.stack([
                resize_evidence(maps_teacher[:, variant], teacher_grid, student_grid, student_ink,
                                args.rsed_graph_steps, args.rsed_graph_mix)
                for variant in range(variants)
            ], dim=1)
            dense_teacher = projected_patch_features(teacher.visual, sequence)
            evidence_teacher = F.normalize(
                torch.einsum("bvp,bpd->bvd", maps_teacher, dense_teacher), dim=-1
            )
            flat_images = images.detach()[:, None].expand(-1, variants, -1, -1, -1).reshape(
                -1, *images.shape[1:]
            )
            flat_maps = maps_student.reshape(-1, student_grid**2)
            masked_images, removed = erase_by_patch_evidence(
                flat_images, flat_maps, args.rsed_mask_fraction,
                args.rsed_ink_threshold, args.rsed_ink_softness,
            )
            encoded = []
            with torch.no_grad():
                for part in masked_images.split(args.rsed_teacher_batch_size):
                    encoded.append(controller(part.to(device=device, dtype=dtype), "sketch"))
            masked_teacher = torch.cat(encoded).reshape(len(images), variants, -1)
            rows = indices.long()
            payload["maps"][rows] = maps_student.detach().half().cpu()
            payload["teacher_evidence"][rows] = evidence_teacher.detach().half().cpu()
            payload["teacher_masked"][rows] = masked_teacher.detach().half().cpu()
            retrieval_confidence = positive_fraction.detach().clamp(0.05, 1.0).cpu()
            payload["confidence"][rows, 0] = retrieval_confidence
            payload["removed_ink_fraction"][rows] = removed.reshape(len(images), variants).cpu()
            payload["teacher_positive_score"][rows] = score.detach().cpu()
            payload["positive_relevance_fraction"][rows] = positive_fraction.detach().cpu()
            payload["target_entropy"][rows] = evidence_entropy(maps_student).detach().cpu()
            cached = teacher_payload["teacher_sketch_features"][rows].float()
            payload["clean_cache_cosine"][rows] = F.cosine_similarity(
                cached, clean.detach().float().cpu(), dim=-1
            )
            del sequence, gradient, patches, patch_gradient, clean, images
        payload["preparation_seconds"] = time.perf_counter() - started
        if (payload["clean_cache_cosine"] < 0.999).any():
            raise RuntimeError("Re-encoded teacher sketch differs from the main cache")
        validate_payload(payload, metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            if path.exists():
                raise FileExistsError("Another process created the RSED cache")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        del controller, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[RSED Cache] saved {path}; {path.stat().st_size / 1024**2:.1f} MiB", flush=True)
    dataset.set_stroke_evidence_targets(payload)
    from src.stroke_evidence_diagnostics import cache_report
    cache_report(payload, dataset, Path(report_dir), args)
    return payload
