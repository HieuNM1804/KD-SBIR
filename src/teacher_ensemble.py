"""Four pretrained CLIP teachers and sequential target-cache preparation."""

import copy
import gc
import math
import random

import numpy as np
import torch


TEACHERS = {
    "dfn5b": {
        "model": "ViT-H-14-quickgelu", "pretrained": "dfn5b", "output_dim": 1024,
        "cache_version": 6, "repo": "apple/DFN5B-CLIP-ViT-H-14",
        "revision": "11738501a1db6d5e0a3451a71ba100be02e577e6",
        "filename": "open_clip_pytorch_model.bin",
        "sha256": "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae",
    },
    "openai_l14": {
        "model": "ViT-L-14", "pretrained": "openai", "output_dim": 768,
        "cache_version": 7, "repo": "timm/vit_large_patch14_clip_224.openai",
        "revision": "18d0535469bb561bf468d76c1d73aa35156c922b",
        "filename": "open_clip_model.safetensors",
        "sha256": "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e",
    },
    "dfn2b_l14": {
        "model": "ViT-L-14", "pretrained": "dfn2b_s39b", "output_dim": 768,
        "cache_version": 7, "repo": "apple/DFN2B-CLIP-ViT-L-14-39B",
        "revision": "a3dd4900c25439e55266b7d5f94138e0cc0a80e0",
        "filename": "open_clip_pytorch_model.bin",
        "sha256": "f2f724646a73b4cd392e5bb41dae589e17ba52dab83006457dddca739d3e2cf9",
    },
    "laion_h14": {
        "model": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "output_dim": 1024,
        "cache_version": 7, "repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        "filename": "open_clip_model.safetensors",
        "sha256": "0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5",
    },
}
DEFAULT_TEACHERS = tuple(TEACHERS)


def teacher_keys(args):
    return tuple(getattr(args, "teacher_models", None) or ("dfn5b",))


def teacher_spec(args):
    return TEACHERS[teacher_keys(args)[0]]


def is_ensemble(args):
    return len(teacher_keys(args)) > 1


def normalized_weights(values, count):
    values = [1.0] * count if values is None else list(values)
    if len(values) != count:
        raise ValueError("Provide one --teacher_ensemble_weights value per --teacher_models entry.")
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Teacher weights must be finite, non-negative and have a positive sum.")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("Sum of teacher weights must be finite.")
    return tuple(v / total for v in values)


def cache_ensemble(owner, dataset, val_sketch, val_photo, batch_size, workers, progress):
    """Reuse main's full single-teacher pipeline; only one backbone lives at a time.

    Concatenation below is storage packing, not feature-space averaging. Losses
    split these tensors by output_dim before forming each teacher distribution.
    """
    from src.model import CustomCLIP, _load_teacher, default_teacher_cache_path

    args = owner.cfg
    sketches, photos, sketch_texts, photo_texts = [], [], [], []
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        for key in teacher_keys(args):
            single = copy.copy(args)
            single.teacher_models = [key]
            single.teacher_ensemble_weights = [1.0]
            single.teacher_cache_path = default_teacher_cache_path(single, dataset)
            print(f"[Ensemble] teacher={key}; cache={single.teacher_cache_path}")
            # Teacher order/cache hits do not alter another teacher's RNG or the
            # later student training. Pair sampling is already worker invariant.
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(single.seed)
                np.random.seed(single.seed)
                torch.manual_seed(single.seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(single.seed)
                worker = CustomCLIP(single, owner.clip_model, owner.classnames,
                                    teacher=_load_teacher(single))
                worker.cache_teacher_features(dataset, val_sketch, val_photo,
                                              batch_size, workers, progress)
            dim = TEACHERS[key]["output_dim"]
            for name, value, rows in (
                ("sketch", dataset.teacher_sketch_features, len(dataset.all_sketches_path)),
                ("photo", dataset.teacher_photo_features, len(dataset.all_photo_paths)),
                ("sketch_text", worker._teacher_sketch_text, len(owner.classnames)),
                ("photo_text", worker._teacher_photo_text, len(owner.classnames)),
            ):
                if not isinstance(value, torch.Tensor) or value.shape != (rows, dim):
                    raise RuntimeError(f"{key}: invalid {name} cache shape; expected {(rows, dim)}")
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"{key}: non-finite {name} cache")
            sketches.append(dataset.teacher_sketch_features.cpu())
            photos.append(dataset.teacher_photo_features.cpu())
            sketch_texts.append(worker._teacher_sketch_text.cpu())
            photo_texts.append(worker._teacher_photo_text.cpu())
            del worker
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
    dataset.set_teacher_features(torch.cat(sketches, dim=-1), torch.cat(photos, dim=-1))
    owner._teacher_sketch_text = torch.cat(sketch_texts, dim=-1)
    owner._teacher_photo_text = torch.cat(photo_texts, dim=-1)
    owner.teacher_active = True
    print(f"[Ensemble] cached {len(sketches)} teachers; packed dimension="
          f"{dataset.teacher_sketch_features.shape[-1]}; weights={args.teacher_ensemble_weights}")
