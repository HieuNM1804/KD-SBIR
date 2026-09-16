"""Teacher-only audit and full-cache visual reports for SGCD."""

import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.dataset import load_image
from src.stroke_graph import (
    CLIP_MEAN,
    CLIP_STD,
    TARGET_NAMES,
    erase_by_patch_evidence,
    evidence_entropy,
)


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rgb(image):
    image = image.detach().float().cpu()
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _heatmap(axis, image, values, grid, title):
    heat = F.interpolate(
        values.reshape(1, 1, grid, grid), size=image.shape[-2:],
        mode="bilinear", align_corners=False,
    )[0, 0]
    axis.imshow(_rgb(image))
    axis.imshow(heat.detach().float().cpu().numpy(), cmap="inferno", alpha=0.62)
    axis.set_title(title)
    axis.axis("off")


def _example_figure(path, indices, maps, priorities, dataset, args):
    import matplotlib.pyplot as plt

    count = min(args.sgcd_diagnostic_examples, len(indices))
    if not count:
        return
    fig, axes = plt.subplots(count, 7, figsize=(20, 3 * count), squeeze=False)
    for row in range(count):
        index = int(indices[row])
        image = dataset.normal_transform(load_image(
            dataset.all_sketches_path[index], dataset.max_size
        ))
        axes[row, 0].imshow(_rgb(image))
        axes[row, 0].set_title(f"sketch {index}")
        axes[row, 0].axis("off")
        for variant, name in enumerate(TARGET_NAMES):
            _heatmap(
                axes[row, 1 + variant], image, maps[row, variant],
                args.sgcd_student_grid, name + " path",
            )
            erased, removed = erase_by_patch_evidence(
                image[None], priorities[row:row + 1, variant],
                args.sgcd_mask_fraction, args.sgcd_ink_threshold,
                args.sgcd_ink_softness,
            )
            axes[row, 4 + variant].imshow(_rgb(erased[0]))
            axes[row, 4 + variant].set_title(
                f"{name} erased {removed[0].item():.1%}"
            )
            axes[row, 4 + variant].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def audit_report(summary, parts, dataset, out, args):
    """Write the gate result before full target construction starts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    effects = torch.cat([
        part["selected_effect"].detach().float().cpu() for part in parts
    ])
    selections = torch.cat([
        part["selected_path_index"].detach().cpu() for part in parts
    ])
    counts = torch.cat([part["path_count"].detach().cpu() for part in parts])
    indices = torch.cat([part["indices"].detach().cpu() for part in parts])
    candidate_effects = torch.cat([
        part["candidate_effects"].detach().float().cpu() for part in parts
    ])
    candidate_local = torch.cat([
        part["candidate_local_scores"].detach().float().cpu() for part in parts
    ])
    candidate_valid = torch.cat([
        part["path_valid"].detach().cpu() for part in parts
    ])
    candidate_weights = torch.cat([
        part["candidate_weights"].detach().float().cpu() for part in parts
    ])
    hard_negative = torch.cat([
        part["hard_negative_label"].detach().cpu() for part in parts
    ])
    clean_margin = torch.cat([
        part["clean_margin"].detach().float().cpu() for part in parts
    ])
    rows = []
    for row in range(len(indices)):
        rows.append({
            "sketch_index": indices[row].item(),
            "path_count": counts[row].item(),
            "hard_negative_label": hard_negative[row].item(),
            "clean_margin": clean_margin[row].item(),
            **{
                name + "_effect": effects[row, variant].item()
                for variant, name in enumerate(TARGET_NAMES)
            },
            **{
                name + "_path": selections[row, variant].item()
                for variant, name in enumerate(TARGET_NAMES)
            },
            **{
                f"path_{index}_local": candidate_local[row, index].item()
                for index in range(candidate_local.shape[1])
            },
            **{
                f"path_{index}_effect": candidate_effects[row, index].item()
                for index in range(candidate_effects.shape[1])
            },
            **{
                f"path_{index}_weight": candidate_weights[row, index].item()
                for index in range(candidate_weights.shape[1])
            },
            **{
                f"path_{index}_valid": bool(candidate_valid[row, index])
                for index in range(candidate_valid.shape[1])
            },
        })
    _write_csv(out / "teacher_audit.csv", rows)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for variant, name in enumerate(TARGET_NAMES):
        axes[0].hist(effects[:, variant].numpy(), bins=40, alpha=0.45, label=name)
    axes[0].set(title="Teacher retrieval-margin drop", xlabel="clean - erased margin")
    axes[0].legend()
    axes[1].hist((effects[:, 0] - effects[:, 2]).numpy(), bins=40)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(title="Verified advantage", xlabel="verified - random effect")
    axes[2].hist(
        counts.numpy(), bins=range(1, args.sgcd_max_paths + 2), align="left"
    )
    axes[2].set(title="Stroke graph candidates", xlabel="valid paths")
    entropy = -(candidate_weights.clamp_min(1e-8) *
                candidate_weights.clamp_min(1e-8).log()).sum(-1)
    axes[3].hist(entropy.numpy(), bins=40)
    axes[3].set(title="Soft target path entropy", xlabel="entropy")
    fig.tight_layout()
    fig.savefig(out / "teacher_audit.png", dpi=160)
    plt.close(fig)

    maps = torch.cat([part["maps"].detach().float().cpu() for part in parts])
    priorities = torch.cat([
        part["mask_priorities"].detach().float().cpu() for part in parts
    ])
    _example_figure(
        out / "teacher_audit_examples.png", indices.tolist(), maps,
        priorities, dataset, args,
    )

    skeletons = torch.cat([part["skeleton"].detach().cpu() for part in parts])
    candidates = torch.cat([
        part["candidate_maps"].detach().float().cpu() for part in parts
    ])
    count = min(args.sgcd_diagnostic_examples, len(indices))
    if count:
        fig, axes = plt.subplots(
            count, 2 + args.sgcd_max_paths,
            figsize=(3 * (2 + args.sgcd_max_paths), 3 * count), squeeze=False,
        )
        for row in range(count):
            index = int(indices[row])
            image = dataset.normal_transform(load_image(
                dataset.all_sketches_path[index], dataset.max_size
            ))
            axes[row, 0].imshow(_rgb(image)); axes[row, 0].set_title(f"sketch {index}")
            axes[row, 1].imshow(skeletons[row], cmap="gray_r"); axes[row, 1].set_title("28x28 skeleton")
            for candidate in range(args.sgcd_max_paths):
                _heatmap(
                    axes[row, 2 + candidate], image, candidates[row, candidate],
                    args.sgcd_student_grid,
                    f"path {candidate}: local={candidate_local[row, candidate]:.3f}\n"
                    f"effect={candidate_effects[row, candidate]:.4f}; "
                    f"weight={candidate_weights[row, candidate]:.3f}",
                )
            for axis in axes[row]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(out / "teacher_stroke_graph_examples.png", dpi=130)
        plt.close(fig)


def cache_report(payload, dataset, out, args):
    """Summarize the complete target cache without duplicating large tensors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    maps = payload["maps"].float()
    effects = payload["selected_effect"].float()
    overlap_local = F.cosine_similarity(maps[:, 0], maps[:, 1], dim=-1)
    overlap_random = F.cosine_similarity(maps[:, 0], maps[:, 2], dim=-1)
    positive = effects.clamp_min(0)
    ratio = (
        positive[:, 0].mean() / positive[:, 2].mean().clamp_min(1e-8)
    ).item()
    win = (
        (effects[:, 0] > effects[:, 2]) & (effects[:, 0] > 0)
    ).float().mean().item()
    summary = {
        "samples": len(maps),
        "targets": list(TARGET_NAMES),
        "teacher_audit": payload.get("teacher_audit"),
        "verified_local_map_cosine_mean": overlap_local.mean().item(),
        "verified_random_map_cosine_mean": overlap_random.mean().item(),
        "verified_random_positive_effect_ratio": ratio,
        "verified_beats_random_rate": win,
        "verified_confidence_nonzero_rate": (
            payload["confidence"][:, 0] > 0
        ).float().mean().item(),
        "verified_confidence_mean": payload["confidence"][:, 0].mean().item(),
        "mean_path_count": payload["path_count"].float().mean().item(),
        "effect_mode": payload["metadata"].get("effect_mode", "positive"),
        "clean_retrieval_margin_mean": payload["clean_margin"].float().mean().item(),
        "verified_masked_margin_mean": payload["masked_margin"][:, 0].float().mean().item(),
        "soft_target_effective_paths_mean": torch.exp(
            evidence_entropy(payload["candidate_weights"].float())
        ).mean().item(),
        "selected_effect_mean": {
            name: effects[:, index].mean().item()
            for index, name in enumerate(TARGET_NAMES)
        },
        "removed_ink_fraction": {
            name: payload["removed_ink_fraction"][:, index].mean().item()
            for index, name in enumerate(TARGET_NAMES)
        },
        "target_entropy": {
            name: evidence_entropy(maps[:, index]).mean().item()
            for index, name in enumerate(TARGET_NAMES)
        },
        "teacher_compatibility_preflight": payload.get(
            "teacher_compatibility_preflight"
        ),
        "teacher_compatibility_full": payload.get("teacher_compatibility_full"),
        "teacher_compatibility_full_acceptable": payload.get(
            "teacher_compatibility_full_acceptable"
        ),
        "preparation_seconds": payload.get("preparation_seconds"),
        "metadata": payload["metadata"],
        "notes": [
            "Local proposal uses sketch-path to representative-photo patch correspondence.",
            "Global verification measures positive-vs-hard-negative retrieval-margin drop.",
            "Verified targets softly mix locally proposed paths by pairwise causal effect.",
            "Random selects another structural path; shuffled is applied during training.",
            "Unseen validation images and labels never enter target construction.",
        ],
    }
    (out / "teacher_probe.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    generator = torch.Generator().manual_seed(args.seed + 31337)
    sample_count = min(2048, len(maps))
    sample = torch.randperm(len(maps), generator=generator)[:sample_count]
    rows = []
    for index in sample.tolist():
        rows.append({
            "sketch_index": index,
            "path_count": payload["path_count"][index].item(),
            "hard_negative_label": payload["hard_negative_label"][index].item(),
            "clean_margin": payload["clean_margin"][index].item(),
            "confidence": payload["confidence"][index, 0].item(),
            "verified_local_cosine": overlap_local[index].item(),
            "verified_random_cosine": overlap_random[index].item(),
            **{
                name + "_effect": effects[index, variant].item()
                for variant, name in enumerate(TARGET_NAMES)
            },
        })
    _write_csv(out / "teacher_target_sample.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(overlap_local.numpy(), bins=40, alpha=0.6, label="verified-local")
    axes[0].hist(overlap_random.numpy(), bins=40, alpha=0.6, label="verified-random")
    axes[0].set(title="Structural target separation", xlabel="map cosine")
    axes[0].legend()
    for variant, name in enumerate(TARGET_NAMES):
        axes[1].hist(effects[:, variant].numpy(), bins=40, alpha=0.45, label=name)
    axes[1].set(title="Teacher causal effects", xlabel="clean - erased retrieval margin")
    axes[1].legend()
    axes[2].hist(
        payload["path_count"].numpy(), bins=range(1, args.sgcd_max_paths + 2),
        align="left",
    )
    axes[2].set(title="Path count", xlabel="paths")
    fig.tight_layout()
    fig.savefig(out / "teacher_target_summary.png", dpi=160)
    plt.close(fig)

    chosen = torch.randperm(len(dataset), generator=generator)[
        :min(args.sgcd_diagnostic_examples, len(dataset))
    ].tolist()
    _example_figure(
        out / "teacher_target_examples.png", chosen, maps[chosen],
        payload["mask_priorities"][chosen].float(), dataset, args,
    )
    printable = {key: value for key, value in summary.items()
                 if key not in ("metadata", "notes")}
    print("[SGCD Teacher Probe]", json.dumps(printable), flush=True)
    return summary
