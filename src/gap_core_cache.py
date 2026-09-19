"""Upgrade a format-v7 CoRe teacher cache with common-prompt features."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import open_clip
import torch
from torch.utils.data import DataLoader

from src.dataset import TeacherFeatureDataset, TrainDataset
from src.model import (
    DFN5B_MODEL,
    DFN5B_OUTPUT_DIM,
    DFN5B_PRETRAINED,
    TEACHER_CACHE_FORMAT_VERSION,
)
from src.teacher_prompts import build_teacher_prompt_controller


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _valid_destination(path):
    if not path.is_file():
        return False
    payload = _load(path)
    return (
        payload.get("metadata", {}).get("format_version")
        == TEACHER_CACHE_FORMAT_VERSION
        and "common_teacher_sketch_features" in payload
        and "common_teacher_photo_features" in payload
    )


@torch.no_grad()
def _encode_common(
    controller,
    teacher,
    paths,
    modality,
    max_size,
    batch_size,
    workers,
):
    dataset = TeacherFeatureDataset(paths, max_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
    output = torch.empty(len(paths), DFN5B_OUTPUT_DIM, dtype=torch.float16)
    teacher_parameter = teacher.visual.conv1.weight
    device = teacher_parameter.device
    dtype = teacher_parameter.dtype
    offset = 0
    last_percent = -10
    for images in loader:
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        features = controller(images, modality, prompt_mode="common")
        end = offset + len(features)
        output[offset:end].copy_(features.to(dtype=torch.float16).cpu())
        offset = end
        percent = int(100 * offset / len(paths))
        if percent >= last_percent + 10 or percent == 100:
            print(f"[Gap Cache] common {modality}: {percent}%", flush=True)
            last_percent = percent
    return output


def upgrade_gap_core_cache(
    source,
    destination,
    root,
    dataset="sketchy_2",
    max_size=224,
    batch_size=64,
    workers=8,
):
    source = Path(source)
    destination = Path(destination)
    if _valid_destination(destination):
        print("[Gap Cache] format-v8 cache already exists:", destination)
        return destination
    if not source.is_file():
        raise FileNotFoundError(f"Legacy format-v7 cache is missing: {source}")

    payload = _load(source)
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != 7:
        raise RuntimeError("Gap cache upgrade requires a format-v7 source cache.")
    prompt_state = payload.get("teacher_prompt_state_dict")
    if not prompt_state:
        raise RuntimeError("Source cache does not contain trained teacher prompts.")

    train_dataset = TrainDataset(
        SimpleNamespace(
            root=str(root),
            dataset=dataset,
            max_size=max_size,
            seed=metadata.get("seed", 42),
        )
    )
    if len(payload["teacher_sketch_features"]) != len(
        train_dataset.all_sketches_path
    ) or len(payload["teacher_photo_features"]) != len(
        train_dataset.all_photo_paths
    ):
        raise RuntimeError("Source cache and current dataset have different lengths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = open_clip.create_model(
        DFN5B_MODEL,
        pretrained=DFN5B_PRETRAINED,
        precision="fp16" if device.type == "cuda" else "fp32",
        device=device,
    )
    teacher.eval().requires_grad_(False)
    controller = build_teacher_prompt_controller(
        teacher,
        n_ctx=metadata["teacher_n_ctx_visual"],
        depth=metadata["teacher_prompt_depth"],
        std=metadata["teacher_prompt_std"],
        seed=metadata["teacher_prompt_seed"],
    ).to(device)
    controller.load_state_dict(prompt_state, strict=True)
    controller.eval().requires_grad_(False)

    common_sketch = _encode_common(
        controller,
        teacher,
        train_dataset.all_sketches_path,
        "sketch",
        max_size,
        batch_size,
        workers,
    )
    common_photo = _encode_common(
        controller,
        teacher,
        train_dataset.all_photo_paths,
        "photo",
        max_size,
        batch_size,
        workers,
    )
    payload["common_teacher_sketch_features"] = common_sketch
    payload["common_teacher_photo_features"] = common_photo
    payload["metadata"] = dict(metadata)
    payload["metadata"]["format_version"] = TEACHER_CACHE_FORMAT_VERSION
    payload["upgraded_from"] = str(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(
        f"[Gap Cache] saved {destination} "
        f"({destination.stat().st_size / 1024**2:.1f} MB)"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="sketchy_2")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    upgrade_gap_core_cache(
        source=arguments.source,
        destination=arguments.destination,
        root=arguments.root,
        dataset=arguments.dataset,
        max_size=arguments.max_size,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
    )
