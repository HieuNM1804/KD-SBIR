"""Prepare photo-conditioned, causally verified stroke-graph teacher targets."""

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.dataset import load_image, normal_transform
from src.stroke_graph import (
    TARGET_NAMES,
    FinalBlockInputCapture,
    erase_by_patch_evidence,
    evidence_entropy,
    local_photo_correspondence,
    normalize_evidence,
    patch_ink_mass,
    path_priority_maps,
    projected_patch_features,
    stroke_path_maps,
)

CACHE_FORMAT_VERSION = 2
STUDENT_ONLY_SOURCE_DRIFT = ("src/model.py", "src/stroke_graph_cache.py")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def add_arguments(parser):
    parser.add_argument("--retrieval_head", choices=("main", "sgcd"), default="main")
    parser.add_argument(
        "--sgcd_student_mode",
        choices=("legacy_head", "native_prompt"),
        default="legacy_head",
        help="Use the historical evidence head or supervise CLIP prompts directly.",
    )
    parser.add_argument("--lambda_sgcd", type=float, default=0.0)
    parser.add_argument("--lambda_sgcd_where", type=float, default=1.0)
    parser.add_argument("--lambda_sgcd_what", type=float, default=0.25)
    parser.add_argument("--lambda_sgcd_effect", type=float, default=0.25)
    parser.add_argument("--lambda_sgcd_anchor", type=float, default=0.20)
    parser.add_argument("--lambda_sgcd_rank", type=float, default=0.50)
    parser.add_argument("--sgcd_effect_magnitude_weight", type=float, default=0.25)
    parser.add_argument(
        "--sgcd_target", choices=TARGET_NAMES + ("shuffled",), default="verified"
    )
    parser.add_argument("--sgcd_beta", type=float, default=0.10)
    parser.add_argument("--sgcd_bottleneck", type=int, default=128)
    parser.add_argument("--sgcd_head_lr", type=float, default=3e-3)
    parser.add_argument("--sgcd_temperature", type=float, default=0.07)
    parser.add_argument("--sgcd_graph_steps", type=int, default=0)
    parser.add_argument("--sgcd_graph_mix", type=float, default=0.0)
    parser.add_argument("--sgcd_mask_fraction", type=float, default=0.10)
    parser.add_argument("--sgcd_ink_threshold", type=float, default=0.08)
    parser.add_argument("--sgcd_ink_softness", type=float, default=0.12)
    parser.add_argument("--sgcd_skeleton_grid", type=int, default=28)
    parser.add_argument("--sgcd_max_paths", type=int, default=4)
    parser.add_argument("--sgcd_photo_representatives", type=int, default=2)
    parser.add_argument("--sgcd_local_topk_patches", type=int, default=4)
    parser.add_argument("--sgcd_proposal_topk", type=int, default=3)
    parser.add_argument(
        "--sgcd_effect_mode",
        choices=("positive", "pairwise"),
        default="pairwise",
        help="Verify paths by positive-similarity drop or positive-vs-negative margin drop.",
    )
    parser.add_argument("--sgcd_negative_topk", type=int, default=3)
    parser.add_argument("--sgcd_target_temperature", type=float, default=0.05)
    parser.add_argument("--sgcd_rank_margin", type=float, default=0.20)
    parser.add_argument("--sgcd_teacher_batch_size", type=int, default=4)
    parser.add_argument("--sgcd_cache_path", type=str, default="")
    parser.add_argument("--sgcd_prepare_only", action="store_true")
    parser.add_argument("--sgcd_audit_only", action="store_true")
    parser.add_argument("--sgcd_warmup_epochs", type=float, default=0.5)
    parser.add_argument("--sgcd_decay_start_epoch", type=float, default=3.0)
    parser.add_argument("--sgcd_diagnostics", action="store_true")
    parser.add_argument("--sgcd_diagnostic_batch_size", type=int, default=32)
    parser.add_argument("--sgcd_diagnostic_examples", type=int, default=8)
    parser.add_argument("--sgcd_audit_samples", type=int, default=512)
    parser.add_argument("--sgcd_min_effect_ratio", type=float, default=1.15)
    parser.add_argument("--sgcd_min_win_rate", type=float, default=0.60)
    parser.add_argument("--sgcd_max_random_map_cosine", type=float, default=0.75)
    parser.add_argument("--sgcd_force_prepare", action="store_true")


def validate_arguments(parser, args):
    if args.retrieval_head == "sgcd" and args.sgcd_student_mode == "native_prompt":
        if args.n_ctx_visual < 1 or args.prompt_depth < 2:
            parser.error(
                "Native prompt localization requires prompts in at least two layers"
            )
        if args.sgcd_beta != 0 or args.lambda_sgcd_anchor != 0:
            parser.error(
                "Native prompt mode requires --sgcd_beta 0 --lambda_sgcd_anchor 0"
            )
    nonnegative = (
        args.lambda_sgcd,
        args.lambda_sgcd_where,
        args.lambda_sgcd_what,
        args.lambda_sgcd_effect,
        args.lambda_sgcd_anchor,
        args.lambda_sgcd_rank,
        args.sgcd_effect_magnitude_weight,
        args.sgcd_beta,
        args.sgcd_head_lr,
        args.sgcd_warmup_epochs,
        args.sgcd_decay_start_epoch,
        args.sgcd_rank_margin,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        parser.error(
            "SGCD weights, beta, learning rate and schedule must be finite and nonnegative"
        )
    if args.retrieval_head == "main" and (
        args.lambda_sgcd > 0 or args.sgcd_prepare_only or args.sgcd_audit_only
    ):
        parser.error("SGCD preparation/training requires --retrieval_head sgcd")
    if (
        args.retrieval_head == "sgcd"
        and args.lambda_sgcd <= 0
        and not args.sgcd_prepare_only
        and not args.sgcd_audit_only
    ):
        parser.error("Use positive --lambda_sgcd or select --retrieval_head main")
    if (
        args.lambda_sgcd > 0
        and sum(
            (
                args.lambda_sgcd_where,
                args.lambda_sgcd_what,
                args.lambda_sgcd_effect,
                args.lambda_sgcd_anchor,
                args.lambda_sgcd_rank,
            )
        )
        <= 0
    ):
        parser.error("At least one SGCD component must be positive")
    positive_ints = (
        args.sgcd_bottleneck,
        args.sgcd_teacher_batch_size,
        args.sgcd_max_paths,
        args.sgcd_photo_representatives,
        args.sgcd_local_topk_patches,
        args.sgcd_proposal_topk,
        args.sgcd_audit_samples,
        args.sgcd_negative_topk,
    )
    if any(value < 1 for value in positive_ints):
        parser.error("SGCD dimensions and batch/audit sizes must be positive")
    if args.sgcd_photo_representatives < 2:
        parser.error(
            "SGCD requires at least two representative photos for proposal/verification separation"
        )
    if args.sgcd_proposal_topk > args.sgcd_max_paths:
        parser.error("--sgcd_proposal_topk cannot exceed --sgcd_max_paths")
    if args.sgcd_skeleton_grid < 7:
        parser.error("--sgcd_skeleton_grid must be at least 7")
    if args.sgcd_graph_steps < 0 or not 0 <= args.sgcd_graph_mix <= 1:
        parser.error("SGCD graph steps/mix are out of range")
    if args.sgcd_temperature <= 0 or not 0 < args.sgcd_mask_fraction < 1:
        parser.error(
            "SGCD temperature must be positive and mask fraction must be in (0,1)"
        )
    if args.sgcd_target_temperature <= 0:
        parser.error("--sgcd_target_temperature must be positive")
    if not 0 <= args.sgcd_ink_threshold < 1 or not 0 < args.sgcd_ink_softness <= 1:
        parser.error("SGCD ink threshold/softness are out of range")
    if args.sgcd_diagnostic_batch_size < 2 or args.sgcd_diagnostic_examples < 0:
        parser.error("Invalid SGCD diagnostic sizes")
    if args.sgcd_min_effect_ratio < 1 or not 0 <= args.sgcd_min_win_rate <= 1:
        parser.error("Invalid SGCD teacher-only gate")
    if not 0 <= args.sgcd_max_random_map_cosine <= 1:
        parser.error("--sgcd_max_random_map_cosine must lie in [0,1]")
    if (
        args.lambda_sgcd > 0 or args.sgcd_prepare_only or args.sgcd_audit_only
    ) and args.teacher_pretrain_epochs < 1:
        parser.error("SGCD requires the prompt-tuned main teacher cache")


class IndexedSketchDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = normal_transform(dataset.max_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        path = self.dataset.all_sketches_path[index]
        category = Path(path).parent.name
        image = self.transform(load_image(path, self.dataset.max_size))
        return image, index, self.dataset.category_to_label[category]


class IndexedImageDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = list(paths)
        self.transform = normal_transform(size)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index], self.size))


def _path_fingerprint(paths, root):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.relpath(path, root).replace("\\", "/").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _photo_labels(dataset):
    return torch.tensor(
        [
            dataset.category_to_label[Path(path).parent.name]
            for path in dataset.all_photo_paths
        ]
    )


def _sketch_labels(dataset):
    return torch.tensor(
        [
            dataset.category_to_label[Path(path).parent.name]
            for path in dataset.all_sketches_path
        ]
    )


def select_photo_representatives(features, labels, class_count, count):
    """Choose deterministic photos closest to each teacher class centroid."""
    features = F.normalize(features.float(), dim=-1)
    output = []
    for label in range(class_count):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if len(indices) == 0:
            raise ValueError(f"No photos for seen class {label}")
        current = features[indices]
        centroid = F.normalize(current.mean(0), dim=-1)
        order = torch.argsort(current @ centroid, descending=True, stable=True)
        chosen = indices[order[: min(count, len(order))]].tolist()
        while len(chosen) < count:
            chosen.append(chosen[len(chosen) % len(chosen)])
        output.append(chosen)
    return torch.tensor(output, dtype=torch.long)


def _compatibility_statistics(cosine):
    values = cosine.detach().float().cpu().flatten()
    if len(values) == 0 or not torch.isfinite(values).all():
        raise ValueError("Teacher compatibility cosine is empty or nonfinite")
    return {
        "count": len(values),
        "minimum": values.min().item(),
        "p01": torch.quantile(values, 0.01).item(),
        "median": values.median().item(),
        "mean": values.mean().item(),
    }


def _compatibility_is_acceptable(values):
    return values["mean"] >= 0.995 and values["p01"] >= 0.980


@torch.no_grad()
def _teacher_compatibility_probe(
    controller, dataset, cached, device, dtype, batch_size
):
    count = min(64, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, steps=count).round().long().unique()
    similarities = []
    transform = dataset.normal_transform
    for part in indices.split(batch_size):
        images = torch.stack(
            [
                transform(
                    load_image(dataset.all_sketches_path[index], dataset.max_size)
                )
                for index in part.tolist()
            ]
        ).to(device=device, dtype=dtype)
        encoded = controller(images, "sketch").float().cpu()
        similarities.append(F.cosine_similarity(encoded, cached[part].float(), dim=-1))
    statistics = _compatibility_statistics(torch.cat(similarities))
    print("[SGCD Cache] teacher re-encoding preflight:", statistics, flush=True)
    if not _compatibility_is_acceptable(statistics):
        raise RuntimeError(
            "Reloaded teacher is incompatible with the main cache; "
            f"statistics={statistics}. Rebuild the main teacher cache."
        )
    return statistics


def _metadata(args, dataset, teacher_path, teacher_metadata, representative_indices):
    root = Path(__file__).resolve().parent.parent
    sources = (
        "clip/model.py",
        "src/dataset.py",
        "src/model.py",
        "src/losses.py",
        "src/teacher_prompts.py",
        "src/stroke_graph.py",
        "src/stroke_graph_cache.py",
    )
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "method": "pairwise_counterfactual_stroke_distillation",
        "definition": (
            "skeleton_paths;photo_patch_correspondence_proposal;"
            "held_out_positive_vs_hard_negative_margin_verification;"
            "soft_multi_path_target;fixed_ink_budget"
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
        "student_grid": args.sgcd_student_grid,
        "targets": list(TARGET_NAMES),
        "mask_fraction": args.sgcd_mask_fraction,
        "ink_threshold": args.sgcd_ink_threshold,
        "ink_softness": args.sgcd_ink_softness,
        "skeleton_grid": args.sgcd_skeleton_grid,
        "max_paths": args.sgcd_max_paths,
        "photo_representatives": args.sgcd_photo_representatives,
        "photo_split": "all_but_last_for_local_proposal;last_for_global_verification",
        "representative_indices_sha256": hashlib.sha256(
            representative_indices.numpy().tobytes()
        ).hexdigest(),
        "local_topk_patches": args.sgcd_local_topk_patches,
        "proposal_topk": args.sgcd_proposal_topk,
        "effect_mode": args.sgcd_effect_mode,
        "negative_topk": args.sgcd_negative_topk,
        "target_temperature": args.sgcd_target_temperature,
        "seed": args.seed,
        "source_sha256": {name: file_sha256(root / name) for name in sources},
    }


def _cache_path(args, metadata, teacher_path):
    if args.sgcd_cache_path:
        return Path(args.sgcd_cache_path)
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return teacher_path.parent / f"{args.dataset}_sgcd_{key}.pt"


def _validation_metadata(metadata, stored_metadata, allow_student_source_drift):
    if not allow_student_source_drift:
        return metadata
    expected = dict(metadata)
    expected["source_sha256"] = dict(metadata["source_sha256"])
    stored_sources = stored_metadata.get("source_sha256", {})
    for name in STUDENT_ONLY_SOURCE_DRIFT:
        if name in stored_sources:
            expected["source_sha256"][name] = stored_sources[name]
    return expected


def validate_payload(payload, metadata, allow_student_source_drift=False):
    stored_metadata = payload.get("metadata")
    if allow_student_source_drift and not isinstance(stored_metadata, dict):
        raise ValueError("SGCD cache metadata is missing")
    comparison_metadata = _validation_metadata(
        metadata,
        stored_metadata if isinstance(stored_metadata, dict) else {},
        allow_student_source_drift,
    )
    if stored_metadata != comparison_metadata:
        raise ValueError("SGCD cache metadata differs; use a new --sgcd_cache_path")
    n, variants = metadata["sketch_count"], len(TARGET_NAMES)
    grid, width, paths = (
        metadata["student_grid"] ** 2,
        metadata["teacher_metadata"]["teacher_output_dim"],
        metadata["max_paths"],
    )
    shapes = {
        "maps": (n, variants, grid),
        "mask_priorities": (n, variants, grid),
        "teacher_evidence": (n, variants, width),
        "teacher_masked": (n, variants, width),
        "confidence": (n, variants),
        "selected_effect": (n, variants),
        "clean_margin": (n,),
        "masked_margin": (n, variants),
        "removed_ink_fraction": (n, variants),
        "candidate_effects": (n, paths),
        "candidate_local_scores": (n, paths),
        "candidate_weights": (n, paths),
        "hard_negative_label": (n,),
        "path_valid": (n, paths),
        "selected_path_index": (n, variants),
        "path_count": (n,),
        "clean_cache_cosine": (n,),
    }
    for name, shape in shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid SGCD tensor {name}: expected {shape}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite SGCD tensor: {name}")
    for name in (
        "maps",
        "mask_priorities",
        "teacher_evidence",
        "teacher_masked",
        "candidate_weights",
    ):
        if payload[name].dtype != torch.float16:
            raise ValueError(f"SGCD {name} must be float16")
    if not torch.allclose(
        payload["maps"].float().sum(-1), torch.ones(n, variants), atol=2e-3
    ):
        raise ValueError("SGCD maps are not normalized")
    if not torch.allclose(
        payload["mask_priorities"].float().sum(-1), torch.ones(n, variants), atol=2e-3
    ):
        raise ValueError("SGCD mask priorities are not normalized")
    if not torch.allclose(
        payload["candidate_weights"].float().sum(-1), torch.ones(n), atol=2e-3
    ):
        raise ValueError("SGCD candidate weights are not normalized")
    if metadata.get("effect_mode") == "pairwise":
        labels = payload["hard_negative_label"].long()
        if (labels < 0).any() or (labels >= len(metadata["classnames"])).any():
            raise ValueError("SGCD hard-negative labels are out of range")
    if (payload["confidence"] < 0).any() or (payload["confidence"] > 1).any():
        raise ValueError("SGCD confidence must lie in [0,1]")
    if (
        payload["removed_ink_fraction"] - metadata["mask_fraction"]
    ).abs().max() > 0.025:
        raise ValueError("SGCD erasure does not respect the ink budget")


@torch.no_grad()
def _encode_photo_representatives(
    controller,
    paths,
    class_count,
    representatives,
    size,
    batch_size,
    workers,
    device,
    dtype,
):
    selected_paths = [paths[index] for index in representatives.flatten().tolist()]
    loader = DataLoader(
        IndexedImageDataset(selected_paths, size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
    dense = []
    for images in tqdm(
        loader, desc="[SGCD Cache] representative photo patches", mininterval=3
    ):
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        with FinalBlockInputCapture(controller._visual) as capture:
            controller(images, "photo")
        dense.append(
            F.normalize(
                projected_patch_features(controller._visual, capture.residual()), dim=-1
            ).half()
        )
    result = torch.cat(dense)
    return result.reshape(
        class_count, representatives.shape[1], result.shape[1], result.shape[2]
    )


def _choose_random(valid, indices, seed, exclude=None):
    selected = []
    for row, index in enumerate(indices.tolist()):
        choices = torch.nonzero(valid[row], as_tuple=False).flatten().cpu()
        if exclude is not None and len(choices) > 1:
            choices = choices[choices != int(exclude[row])]
        generator = torch.Generator().manual_seed(seed + int(index) * 104729)
        selected.append(
            choices[torch.randint(len(choices), (1,), generator=generator)].item()
        )
    return torch.tensor(selected, device=valid.device, dtype=torch.long)


@torch.no_grad()
def _target_batch(images, indices, labels, controller, photo_dense, photo_global, args):
    visual = controller._visual
    with FinalBlockInputCapture(visual) as capture:
        clean = F.normalize(controller(images, "sketch").float(), dim=-1)
    dense = F.normalize(projected_patch_features(visual, capture.residual()), dim=-1)
    path_maps, valid, skeleton = stroke_path_maps(
        images,
        args.sgcd_student_grid,
        args.sgcd_skeleton_grid,
        args.sgcd_max_paths,
        args.sgcd_ink_threshold,
        args.sgcd_ink_softness,
    )
    batch, paths, grid = path_maps.shape
    teacher_grid = args.max_size // 14
    teacher_ink = patch_ink_mass(
        images, teacher_grid, args.sgcd_ink_threshold, args.sgcd_ink_softness
    )
    upsampled = F.interpolate(
        path_maps.reshape(
            batch * paths, 1, args.sgcd_student_grid, args.sgcd_student_grid
        ),
        size=(teacher_grid, teacher_grid),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    upsampled = normalize_evidence(
        upsampled,
        teacher_ink[:, None].expand(-1, paths, -1).reshape(batch * paths, -1),
    ).reshape(batch, paths, -1)
    path_features = F.normalize(torch.einsum("bkp,bpd->bkd", upsampled, dense), dim=-1)
    local = local_photo_correspondence(
        path_features, photo_dense[labels, :-1], args.sgcd_local_topk_patches
    ).masked_fill(~valid, -torch.inf)

    student_ink = patch_ink_mass(
        images, args.sgcd_student_grid, args.sgcd_ink_threshold, args.sgcd_ink_softness
    )
    priority = path_priority_maps(path_maps, student_ink, args.sgcd_student_grid, valid)
    priority = normalize_evidence(
        priority.reshape(batch * paths, -1),
        student_ink[:, None].expand(-1, paths, -1).reshape(batch * paths, -1),
    ).reshape(batch, paths, grid)
    expanded_images = (
        images[:, None]
        .expand(-1, paths, -1, -1, -1)
        .reshape(batch * paths, *images.shape[1:])
    )
    masked_images, removed = erase_by_patch_evidence(
        expanded_images,
        priority.reshape(batch * paths, grid),
        args.sgcd_mask_fraction,
        args.sgcd_ink_threshold,
        args.sgcd_ink_softness,
    )
    masked = []
    for part in masked_images.split(args.sgcd_teacher_batch_size):
        masked.append(F.normalize(controller(part, "sketch").float(), dim=-1))
    masked = torch.cat(masked).reshape(batch, paths, -1)
    positives = F.normalize(photo_global[:, -1].float(), dim=-1)[labels]
    positive_clean = torch.einsum("bd,bd->b", clean, positives)
    positive_masked = torch.einsum("bkd,bd->bk", masked, positives)
    hard_negative_label = torch.full_like(labels, -1)
    if args.sgcd_effect_mode == "pairwise":
        if photo_global.shape[0] < 2:
            raise ValueError("Pairwise SGCD needs at least two seen classes")
        class_centroids = F.normalize(photo_global.float().mean(1), dim=-1)
        negative_scores = clean @ class_centroids.T
        negative_scores.scatter_(1, labels[:, None], -torch.inf)
        negative_count = min(args.sgcd_negative_topk, photo_global.shape[0] - 1)
        negative_labels = negative_scores.topk(negative_count, dim=-1).indices
        hard_negative_label = negative_labels[:, 0]
        negative_gallery = F.normalize(photo_global[:, -1].float(), dim=-1)[
            negative_labels
        ]
        negative_clean = torch.einsum("bd,bnd->bn", clean, negative_gallery).mean(-1)
        negative_masked = torch.einsum("bkd,bnd->bkn", masked, negative_gallery).mean(
            -1
        )
        clean_margin = positive_clean - negative_clean
        candidate_masked_margin = positive_masked - negative_masked
    else:
        negative_gallery = None
        clean_margin = positive_clean
        candidate_masked_margin = positive_masked
    effects = (clean_margin[:, None] - candidate_masked_margin).masked_fill(
        ~valid, -torch.inf
    )

    proposal_count = min(args.sgcd_proposal_topk, paths)
    proposed = local.topk(proposal_count, dim=-1).indices
    proposed_mask = torch.zeros_like(valid)
    proposed_mask.scatter_(1, proposed, True)
    proposed_mask &= valid
    proposed_effects = effects.gather(1, proposed)
    verified = proposed.gather(1, proposed_effects.argmax(-1, keepdim=True))[:, 0]
    if args.sgcd_effect_mode == "pairwise":
        target_logits = (effects / args.sgcd_target_temperature).masked_fill(
            ~proposed_mask, -torch.inf
        )
        candidate_weights = target_logits.softmax(-1)
    else:
        candidate_weights = F.one_hot(verified, paths).float()

    local_choice = local.argmax(-1)
    random_choice = _choose_random(valid, indices, args.seed + 17171, verified)
    rows_1d = torch.arange(batch, device=images.device)
    verified_map = torch.einsum("bk,bkp->bp", candidate_weights, path_maps)
    verified_map = verified_map / verified_map.sum(-1, keepdim=True).clamp_min(1e-8)
    verified_priority = torch.einsum("bk,bkp->bp", candidate_weights, priority)
    verified_priority = normalize_evidence(verified_priority, student_ink)
    verified_evidence = F.normalize(
        torch.einsum("bk,bkd->bd", candidate_weights, path_features), dim=-1
    )
    verified_images, verified_removed = erase_by_patch_evidence(
        images,
        verified_priority,
        args.sgcd_mask_fraction,
        args.sgcd_ink_threshold,
        args.sgcd_ink_softness,
    )
    verified_masked_parts = []
    for part in verified_images.split(args.sgcd_teacher_batch_size):
        verified_masked_parts.append(
            F.normalize(controller(part, "sketch").float(), dim=-1)
        )
    verified_masked = torch.cat(verified_masked_parts)
    verified_positive = torch.einsum("bd,bd->b", verified_masked, positives)
    if negative_gallery is None:
        verified_masked_margin = verified_positive
    else:
        verified_negative = torch.einsum(
            "bd,bnd->bn", verified_masked, negative_gallery
        ).mean(-1)
        verified_masked_margin = verified_positive - verified_negative

    selected = torch.stack((verified, local_choice, random_choice), dim=1)
    local_random = selected[:, 1:]
    local_random_masked = masked[rows_1d[:, None], local_random]
    local_random_margin = candidate_masked_margin[rows_1d[:, None], local_random]
    selected_masked_margin = torch.cat(
        (verified_masked_margin[:, None], local_random_margin), dim=1
    )
    selected_effect = clean_margin[:, None] - selected_masked_margin
    random_effect = selected_effect[:, 2]
    advantage = selected_effect[:, 0] - random_effect
    verified_confidence = (
        advantage.clamp_min(0)
        / (selected_effect[:, 0].abs() + random_effect.abs() + 1e-3)
    ).clamp(0, 1)
    confidence = torch.stack(
        (verified_confidence, torch.ones_like(advantage), torch.ones_like(advantage)),
        dim=1,
    )
    maps = torch.stack(
        (
            verified_map,
            path_maps[rows_1d, local_choice],
            path_maps[rows_1d, random_choice],
        ),
        dim=1,
    )
    mask_priorities = torch.stack(
        (
            verified_priority,
            priority[rows_1d, local_choice],
            priority[rows_1d, random_choice],
        ),
        dim=1,
    )
    teacher_evidence = torch.stack(
        (
            verified_evidence,
            path_features[rows_1d, local_choice],
            path_features[rows_1d, random_choice],
        ),
        dim=1,
    )
    teacher_masked = torch.cat((verified_masked[:, None], local_random_masked), dim=1)
    removed_selected = torch.stack(
        (
            verified_removed,
            removed.reshape(batch, paths)[rows_1d, local_choice],
            removed.reshape(batch, paths)[rows_1d, random_choice],
        ),
        dim=1,
    )
    return {
        "indices": indices,
        "maps": maps,
        "mask_priorities": mask_priorities,
        "teacher_evidence": teacher_evidence,
        "teacher_masked": teacher_masked,
        "confidence": confidence,
        "selected_effect": selected_effect,
        "clean_margin": clean_margin,
        "masked_margin": selected_masked_margin,
        "removed_ink_fraction": removed_selected,
        "candidate_effects": effects.masked_fill(~valid, 0),
        "candidate_local_scores": local.masked_fill(~valid, 0),
        "candidate_weights": candidate_weights,
        "hard_negative_label": hard_negative_label,
        "candidate_maps": path_maps,
        "path_valid": valid,
        "selected_path_index": selected,
        "path_count": valid.sum(-1),
        "clean": clean,
        "skeleton": skeleton,
    }


def _audit_summary(parts, args):
    effects = torch.cat([part["selected_effect"].cpu() for part in parts])
    verified, local, random = effects.unbind(1)
    verified_positive, random_positive = verified.clamp_min(0), random.clamp_min(0)
    ratio = (verified_positive.mean() / random_positive.mean().clamp_min(1e-8)).item()
    win = ((verified > random) & (verified > 0)).float().mean().item()
    positive = (verified > 0).float().mean().item()
    selections = torch.cat([part["selected_path_index"].cpu() for part in parts])
    path_counts = torch.cat([part["path_count"].cpu() for part in parts]).float()
    maps = torch.cat([part["maps"].float().cpu() for part in parts])
    weights = torch.cat([part["candidate_weights"].float().cpu() for part in parts])
    clean_margin = torch.cat([part["clean_margin"].float().cpu() for part in parts])
    random_map_cosine = (
        F.cosine_similarity(maps[:, 0], maps[:, 2], dim=-1).mean().item()
    )
    summary = {
        "samples": len(effects),
        "verified_positive_effect_mean": verified_positive.mean().item(),
        "local_positive_effect_mean": local.clamp_min(0).mean().item(),
        "random_positive_effect_mean": random_positive.mean().item(),
        "verified_random_effect_ratio": ratio,
        "verified_beats_random_rate": win,
        "verified_positive_rate": positive,
        "verified_differs_from_local_rate": (selections[:, 0] != selections[:, 1])
        .float()
        .mean()
        .item(),
        "mean_path_count": path_counts.mean().item(),
        "verified_random_map_cosine_mean": random_map_cosine,
        "effect_mode": args.sgcd_effect_mode,
        "clean_retrieval_margin_mean": clean_margin.mean().item(),
        "soft_target_effective_paths_mean": torch.exp(evidence_entropy(weights))
        .mean()
        .item(),
        "minimum_effect_ratio": args.sgcd_min_effect_ratio,
        "minimum_win_rate": args.sgcd_min_win_rate,
        "maximum_random_map_cosine": args.sgcd_max_random_map_cosine,
        "passed": (
            ratio >= args.sgcd_min_effect_ratio
            and win >= args.sgcd_min_win_rate
            and random_map_cosine <= args.sgcd_max_random_map_cosine
        ),
        "forced": bool(args.sgcd_force_prepare),
    }
    return summary


def prepare_cache(args, dataset, report_dir):
    if not args.teacher_cache_path:
        raise ValueError("SGCD requires --teacher_cache_path")
    teacher_path = Path(args.teacher_cache_path)
    if not teacher_path.is_file():
        raise FileNotFoundError(
            "Prepare the main prompt-tuned teacher cache before SGCD"
        )
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_metadata = teacher_payload.get("metadata", {})
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION

    if (
        teacher_metadata.get("format_version") != TEACHER_CACHE_FORMAT_VERSION
        or teacher_metadata.get("teacher_model") != DFN5B_MODEL
        or teacher_metadata.get("teacher_pretrained") != DFN5B_PRETRAINED
        or not teacher_payload.get("teacher_prompt_state_dict")
    ):
        raise ValueError("SGCD needs the same prompt-tuned DFN5B teacher used by main")
    photo_labels = _photo_labels(dataset)
    _sketch_labels(dataset)
    representatives = select_photo_representatives(
        teacher_payload["teacher_photo_features"],
        photo_labels,
        len(dataset.all_categories),
        args.sgcd_photo_representatives,
    )
    metadata = _metadata(args, dataset, teacher_path, teacher_metadata, representatives)
    path = _cache_path(args, metadata, teacher_path)
    args.sgcd_cache_path = str(path)
    args.sgcd_target_metadata = metadata
    report_dir = Path(report_dir)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        allow_student_source_drift = (
            getattr(args, "sgcd_student_mode", "legacy_head") == "native_prompt"
        )
        validate_payload(
            payload,
            metadata,
            allow_student_source_drift=allow_student_source_drift,
        )
        if allow_student_source_drift:
            print(
                "[SGCD Cache] native_prompt reusing target cache; "
                "student-only source drift accepted for src/model.py and "
                "src/stroke_graph_cache.py.",
                flush=True,
            )
        print("[SGCD Cache] reused; teacher extraction skipped:", path, flush=True)
    else:
        estimate = len(dataset) * (
            len(TARGET_NAMES) * (2 * 1024 * 2 + 2 * 49 * 2 + 32)
            + args.sgcd_max_paths * 8
        )
        free = shutil.disk_usage(
            path.parent if path.parent.exists() else teacher_path.parent
        ).free
        if free < estimate + 2 * 1024**3:
            raise OSError(
                f"SGCD cache needs about {estimate / 1024**2:.1f} MiB plus 2 GiB reserve"
            )
        import open_clip

        from src.teacher_prompts import TeacherPromptController

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
        if (
            teacher.visual.positional_embedding.shape[0] - 1
            != (args.max_size // 14) ** 2
        ):
            raise ValueError("Unexpected DFN5B patch geometry for SGCD")
        controller = TeacherPromptController(
            teacher.visual,
            teacher_metadata["teacher_n_ctx_visual"],
            teacher_metadata["teacher_prompt_depth"],
            teacher_metadata["teacher_prompt_std"],
            teacher_metadata["teacher_prompt_seed"],
        )
        controller.load_state_dict(
            teacher_payload["teacher_prompt_state_dict"], strict=True
        )
        controller.eval().requires_grad_(False)
        dtype = teacher.visual.conv1.weight.dtype
        preflight = _teacher_compatibility_probe(
            controller,
            dataset,
            teacher_payload["teacher_sketch_features"],
            device,
            dtype,
            args.sgcd_teacher_batch_size,
        )
        photo_dense = _encode_photo_representatives(
            controller,
            dataset.all_photo_paths,
            len(dataset.all_categories),
            representatives,
            dataset.max_size,
            args.sgcd_teacher_batch_size,
            args.workers,
            device,
            dtype,
        )
        photo_global = teacher_payload["teacher_photo_features"][representatives].to(
            device=device, dtype=torch.float32
        )

        audit_indices = (
            torch.linspace(
                0, len(dataset) - 1, steps=min(args.sgcd_audit_samples, len(dataset))
            )
            .round()
            .long()
            .unique()
        )
        audit_loader = DataLoader(
            IndexedSketchDataset(dataset, audit_indices.tolist()),
            batch_size=args.sgcd_teacher_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        audit_parts = []
        for images, indices, labels in tqdm(
            audit_loader, desc="[SGCD Audit] teacher-only stroke gate", mininterval=3
        ):
            audit_parts.append(
                _target_batch(
                    images.to(device=device, dtype=dtype, non_blocking=True),
                    indices.to(device),
                    labels.to(device),
                    controller,
                    photo_dense,
                    photo_global,
                    args,
                )
            )
        audit = _audit_summary(audit_parts, args)
        from src.stroke_graph_reports import audit_report

        audit_report(audit, audit_parts, dataset, report_dir, args)
        print("[SGCD Audit]", json.dumps(audit), flush=True)
        if not audit["passed"] and not args.sgcd_force_prepare:
            raise RuntimeError(
                "SGCD teacher-only gate failed before full cache construction. "
                "Inspect sgcd_diagnostics/teacher_audit.*; use --sgcd_force_prepare only for diagnosis."
            )
        if args.sgcd_audit_only:
            del photo_dense, controller, teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "[SGCD Audit] audit-only run complete; full cache construction skipped",
                flush=True,
            )
            return None
        del audit_parts

        n, variants, width = (
            len(dataset),
            len(TARGET_NAMES),
            teacher_metadata["teacher_output_dim"],
        )
        grid, paths = args.sgcd_student_grid**2, args.sgcd_max_paths
        payload = {
            "metadata": metadata,
            "teacher_audit": audit,
            "teacher_compatibility_preflight": preflight,
            "maps": torch.empty(n, variants, grid, dtype=torch.float16),
            "mask_priorities": torch.empty(n, variants, grid, dtype=torch.float16),
            "teacher_evidence": torch.empty(n, variants, width, dtype=torch.float16),
            "teacher_masked": torch.empty(n, variants, width, dtype=torch.float16),
            "confidence": torch.empty(n, variants),
            "selected_effect": torch.empty(n, variants),
            "clean_margin": torch.empty(n),
            "masked_margin": torch.empty(n, variants),
            "removed_ink_fraction": torch.empty(n, variants),
            "candidate_effects": torch.empty(n, paths, dtype=torch.float16),
            "candidate_local_scores": torch.empty(n, paths, dtype=torch.float16),
            "candidate_weights": torch.empty(n, paths, dtype=torch.float16),
            "hard_negative_label": torch.empty(n, dtype=torch.int16),
            "path_valid": torch.empty(n, paths, dtype=torch.bool),
            "selected_path_index": torch.empty(n, variants, dtype=torch.int16),
            "path_count": torch.empty(n, dtype=torch.int16),
            "clean_cache_cosine": torch.empty(n),
        }
        loader = DataLoader(
            IndexedSketchDataset(dataset, range(n)),
            batch_size=args.sgcd_teacher_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        started = time.perf_counter()
        for images, indices, labels in tqdm(
            loader, desc="[SGCD Cache] causal stroke targets", mininterval=5
        ):
            result = _target_batch(
                images.to(device=device, dtype=dtype, non_blocking=True),
                indices.to(device),
                labels.to(device),
                controller,
                photo_dense,
                photo_global,
                args,
            )
            rows = indices.long()
            for name in (
                "maps",
                "mask_priorities",
                "teacher_evidence",
                "teacher_masked",
                "candidate_effects",
                "candidate_local_scores",
                "candidate_weights",
            ):
                payload[name][rows] = result[name].detach().half().cpu()
            for name in (
                "confidence",
                "selected_effect",
                "clean_margin",
                "masked_margin",
                "removed_ink_fraction",
            ):
                payload[name][rows] = result[name].detach().float().cpu()
            payload["hard_negative_label"][rows] = (
                result["hard_negative_label"].short().cpu()
            )
            payload["path_valid"][rows] = result["path_valid"].cpu()
            payload["selected_path_index"][rows] = (
                result["selected_path_index"].short().cpu()
            )
            payload["path_count"][rows] = result["path_count"].short().cpu()
            cached = teacher_payload["teacher_sketch_features"][rows].float()
            payload["clean_cache_cosine"][rows] = F.cosine_similarity(
                cached, result["clean"].float().cpu(), dim=-1
            )
        payload["preparation_seconds"] = time.perf_counter() - started
        compatibility = _compatibility_statistics(payload["clean_cache_cosine"])
        payload["teacher_compatibility_full"] = compatibility
        payload["teacher_compatibility_full_acceptable"] = _compatibility_is_acceptable(
            compatibility
        )
        print("[SGCD Cache] full teacher agreement:", compatibility, flush=True)
        validate_payload(payload, metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            if path.exists():
                raise FileExistsError("Another process created the SGCD cache")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        del photo_dense, controller, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[SGCD Cache] saved {path}; {path.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )
    dataset.set_stroke_graph_targets(payload)
    from src.stroke_graph_reports import cache_report

    cache_report(payload, dataset, report_dir, args)
    return payload
