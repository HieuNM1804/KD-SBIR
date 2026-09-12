"""Persistent targets from the same tuned teacher as main's global cache."""

import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.dataset import TeacherFeatureDataset
from src.attention_output_kd import PatchOutputCapture


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(dataset, root):
    digest = hashlib.sha256()
    paths = dataset.all_sketches_path + dataset.all_photo_paths
    for path in tqdm(paths, desc="[AV Cache] image fingerprints"):
        relative = os.path.relpath(path, root).replace("\\", "/")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def validate_cache(payload, metadata, ns, np_, width=1280):
    if payload.get("metadata") != metadata:
        raise ValueError("AV cache metadata mismatch. Use a new --av_cache_path.")
    for name, count in [("sketch", ns), ("photo", np_)]:
        x = payload.get(name)
        if (
            not isinstance(x, torch.Tensor)
            or x.shape != (count, width)
            or x.dtype != torch.float16
        ):
            raise ValueError(f"Invalid {name} AV target shape/dtype")
        if not torch.isfinite(x).all() or (x.float().norm(dim=-1) <= 1e-8).any():
            raise ValueError(f"Invalid {name} AV target values")


def prepare_av_cache(args, dataset):
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION
    from src.teacher_prompts import TeacherPromptController
    import open_clip

    teacher_path = Path(args.teacher_cache_path)
    if not teacher_path.is_file():
        raise FileNotFoundError("Create/load the main teacher cache before AV targets")
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_meta = teacher_payload["metadata"]
    if (
        teacher_meta.get("format_version") != TEACHER_CACHE_FORMAT_VERSION
        or teacher_meta.get("dataset") != args.dataset
        or teacher_meta.get("max_size") != dataset.max_size
        or teacher_meta.get("teacher_model") != DFN5B_MODEL
        or teacher_meta.get("teacher_pretrained") != DFN5B_PRETRAINED
        or teacher_meta.get("pretrain_epochs", 0) < 1
        or not teacher_payload.get("teacher_prompt_state_dict")
    ):
        raise ValueError("AV targets require the matching tuned DFN5B teacher cache")
    metadata = {
        "version": 1,
        "definition": "last_CLS_patch_AVWO_full_key_softmax_fp32_no_output_bias",
        "teacher_sha256": file_sha256(teacher_path),
        "teacher_metadata": teacher_meta,
        "image_sha256": image_fingerprint(dataset, args.root),
        "torch": str(torch.__version__),
        "open_clip": getattr(open_clip, "__version__", "unknown"),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = (
        Path(args.av_cache_path)
        if args.av_cache_path
        else teacher_path.parent / f"{args.dataset}_av_{key}.pt"
    )
    args.av_cache_path = str(path)
    print("[AV Cache] path:", path)
    ns, np_ = len(dataset.all_sketches_path), len(dataset.all_photo_paths)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_cache(payload, metadata, ns, np_)
        dataset.set_attention_output_features(payload["sketch"], payload["photo"])
        print("[AV Cache] loaded; DFN5B target pass skipped")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = (
        open_clip.create_model(
            DFN5B_MODEL,
            pretrained=DFN5B_PRETRAINED,
            precision="fp16" if device.type == "cuda" else "fp32",
            device=device,
        )
        .eval()
        .requires_grad_(False)
    )
    controller = TeacherPromptController(
        teacher.visual,
        teacher_meta["teacher_n_ctx_visual"],
        teacher_meta["teacher_prompt_depth"],
        teacher_meta["teacher_prompt_std"],
        teacher_meta["teacher_prompt_seed"],
    )
    controller.load_state_dict(
        teacher_payload["teacher_prompt_state_dict"], strict=True
    )
    controller.eval().requires_grad_(False)
    del teacher_payload
    targets = {}
    for modality, paths in [
        ("sketch", dataset.all_sketches_path),
        ("photo", dataset.all_photo_paths),
    ]:
        loader = DataLoader(
            TeacherFeatureDataset(paths, dataset.max_size),
            batch_size=args.av_teacher_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + 301),
        )
        output = torch.empty(
            len(paths), teacher.visual.conv1.out_channels, dtype=torch.float16
        )
        offset = 0
        for images in tqdm(loader, desc=f"[AV Cache] {modality}"):
            images = images.to(device=device, dtype=teacher.visual.conv1.weight.dtype)
            with torch.no_grad(), PatchOutputCapture(teacher.visual) as capture:
                controller(images, modality)
            if len(capture.values) != 1:
                raise RuntimeError("Expected one AV target per image batch")
            values = capture.values[0].detach().cpu().half()
            output[offset : offset + len(values)] = values
            offset += len(values)
        targets[modality] = output
    payload = {"metadata": metadata, **targets}
    validate_cache(payload, metadata, ns, np_)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        if path.exists():
            raise FileExistsError(f"Another process created {path}; refusing overwrite")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    dataset.set_attention_output_features(targets["sketch"], targets["photo"])
    del teacher, controller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[AV Cache] saved {path} ({path.stat().st_size/1024**2:.1f} MiB); teacher released"
    )
