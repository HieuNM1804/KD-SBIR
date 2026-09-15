"""Teacher-target and training diagnostics for RSED."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback
from torch.nn import functional as F
from torch.utils.data import default_collate

from src.stroke_evidence import CLIP_MEAN, CLIP_STD, TARGET_NAMES, erase_by_patch_evidence
from src.losses import loss_fn


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rgb(image):
    image = image.detach().float().cpu()
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def cache_report(payload, dataset, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    maps = payload["maps"].float()
    confidence = payload["confidence"].float()
    overlap = F.cosine_similarity(maps[:, 0], maps[:, 1], dim=-1)
    random_overlap = F.cosine_similarity(maps[:, 0], maps[:, 2], dim=-1)
    summary = {
        "samples": len(maps),
        "targets": list(TARGET_NAMES),
        "retrieval_attention_map_cosine_mean": overlap.mean().item(),
        "retrieval_random_map_cosine_mean": random_overlap.mean().item(),
        "retrieval_confidence_mean": confidence[:, 0].mean().item(),
        "retrieval_confidence_median": confidence[:, 0].median().item(),
        "positive_relevance_fraction_mean": payload["positive_relevance_fraction"].mean().item(),
        "teacher_positive_score_mean": payload["teacher_positive_score"].mean().item(),
        "removed_ink_fraction": {
            name: payload["removed_ink_fraction"][:, index].mean().item()
            for index, name in enumerate(TARGET_NAMES)
        },
        "target_entropy": {
            name: payload["target_entropy"][:, index].mean().item()
            for index, name in enumerate(TARGET_NAMES)
        },
        "preparation_seconds": payload.get("preparation_seconds"),
        "teacher_compatibility_preflight": payload.get("teacher_compatibility_preflight"),
        "teacher_compatibility_full": payload.get("teacher_compatibility_full"),
        "teacher_compatibility_full_acceptable": payload.get(
            "teacher_compatibility_full_acceptable"
        ),
        "notes": [
            "Retrieval evidence is gradient-times-residual for similarity to the seen-class photo prototype, gated by final CLS attention and ink mass.",
            "Attention is a raw CLS-attention control; random uses the same ink support and erasure budget.",
            "All target maps are resized to the student patch lattice and diffused only through adjacent ink patches.",
            "The cache uses seen training classes only; unseen validation labels never enter target construction.",
        ],
        "metadata": payload["metadata"],
    }
    (out / "teacher_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for index in range(len(maps)):
        row = {
            "sketch_index": index,
            "positive_score": payload["teacher_positive_score"][index].item(),
            "positive_relevance_fraction": payload["positive_relevance_fraction"][index].item(),
            "confidence": confidence[index, 0].item(),
            "retrieval_attention_cosine": overlap[index].item(),
            "retrieval_random_cosine": random_overlap[index].item(),
        }
        for variant, name in enumerate(TARGET_NAMES):
            row[name + "_entropy"] = payload["target_entropy"][index, variant].item()
            row[name + "_removed_ink_fraction"] = payload["removed_ink_fraction"][index, variant].item()
        rows.append(row)
    write_csv(out / "teacher_targets.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(confidence[:, 0].numpy(), bins=40)
    axes[0].set(title="Retrieval-evidence confidence", xlabel="positive relevance fraction")
    axes[1].hist(overlap.numpy(), bins=40, alpha=0.65, label="retrieval vs attention")
    axes[1].hist(random_overlap.numpy(), bins=40, alpha=0.65, label="retrieval vs random")
    axes[1].set(title="Target-map similarity", xlabel="cosine")
    axes[1].legend()
    for variant, name in enumerate(TARGET_NAMES):
        axes[2].hist(payload["target_entropy"][:, variant].numpy(), bins=40, alpha=0.45, label=name)
    axes[2].set(title="Evidence concentration", xlabel="entropy")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "teacher_target_summary.png", dpi=160)
    plt.close(fig)

    count = min(args.rsed_diagnostic_examples, len(dataset))
    if count:
        generator = torch.Generator().manual_seed(args.seed + 9300)
        chosen = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        fig, axes = plt.subplots(count, 7, figsize=(20, 3 * count), squeeze=False)
        transform = dataset.normal_transform
        from src.dataset import load_image
        for row, index in enumerate(chosen):
            image = transform(load_image(dataset.all_sketches_path[index], dataset.max_size))
            axes[row, 0].imshow(_rgb(image))
            axes[row, 0].set_title(f"sketch {index}")
            for variant, name in enumerate(TARGET_NAMES):
                heat = F.interpolate(maps[index, variant].reshape(1, 1, args.rsed_student_grid, args.rsed_student_grid),
                                     size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                axes[row, 1 + variant].imshow(_rgb(image))
                axes[row, 1 + variant].imshow(heat.numpy(), cmap="inferno", alpha=0.62)
                axes[row, 1 + variant].set_title(name + " evidence")
                masked, removed = erase_by_patch_evidence(
                    image[None], maps[index:index + 1, variant],
                    args.rsed_mask_fraction, args.rsed_ink_threshold, args.rsed_ink_softness,
                )
                axes[row, 4 + variant].imshow(_rgb(masked[0]))
                axes[row, 4 + variant].set_title(f"{name} erased {removed[0]:.1%}")
            for axis in axes[row]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(out / "teacher_evidence_examples.png", dpi=130)
        plt.close(fig)
    print("[RSED Teacher Probe]", json.dumps({k: v for k, v in summary.items() if k not in ("metadata", "notes")}), flush=True)
    return summary


def _gradient_stats(a, b):
    norm_a = math.sqrt(sum(value.float().square().sum().item() for value in a))
    norm_b = math.sqrt(sum(value.float().square().sum().item() for value in b))
    dot = sum((x.float() * y.float()).sum().item() for x, y in zip(a, b))
    return {
        "main_norm": norm_a,
        "weighted_rsed_norm": norm_b,
        "rsed_over_main": norm_b / norm_a if norm_a > 1e-12 else None,
        "cosine": dot / (norm_a * norm_b) if norm_a * norm_b > 1e-20 else None,
    }


class StrokeEvidenceDiagnostics(Callback):
    def _setup(self, trainer, module):
        if hasattr(self, "out"):
            return
        self.out = Path(trainer.log_dir or trainer.default_root_dir) / "rsed_diagnostics"
        self.out.mkdir(parents=True, exist_ok=True)
        self.epochs = []
        self.fixed = []
        self.gradients = []
        (self.out / "configuration.json").write_text(
            json.dumps(vars(module.args), indent=2), encoding="utf-8"
        )

    def on_train_start(self, trainer, module):
        self._setup(trainer, module)
        dataset = trainer.train_dataloader.dataset
        generator = torch.Generator().manual_seed(module.args.seed + 9400)
        count = min(module.args.rsed_diagnostic_batch_size, len(dataset))
        self.indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        self.batch = default_collate([dataset[(0, index)] for index in self.indices])
        (self.out / "fixed_batch.json").write_text(
            json.dumps({"indices": self.indices, "sample_epoch": 0,
                        "notes": "Fixed seen training batch; measurements do not update parameters."}, indent=2),
            encoding="utf-8",
        )
        self.measure(trainer, module, "initial")

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking or not getattr(module, "_rsed_last_validation", None):
            return
        self._setup(trainer, module)
        row = {
            "epoch": int(trainer.current_epoch) + (1 if trainer.global_step else 0),
            "global_step": int(trainer.global_step),
            **module._rsed_last_validation,
        }
        self.epochs.append(row)
        write_csv(self.out / "epochs.csv", self.epochs)
        if trainer.global_step and hasattr(self, "batch"):
            self.measure(trainer, module, f"epoch_{row['epoch']}")
        self.figure()
        print(
            f"[RSED Diagnostics] epoch={row['epoch']} deployed/native mAP="
            f"{row['mAP']:.4f}/{row['native_mAP']:.4f}; "
            f"descriptor-native cosine={row['descriptor_native_cosine_sketch']:.4f}",
            flush=True,
        )

    def measure(self, trainer, module, stage):
        self._setup(trainer, module)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            batch = module.transfer_batch_to_device(self.batch, module.device, 0)
            named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
            params = [parameter for _, parameter in named]
            with torch.enable_grad():
                features, output = module.model.forward_with_stroke_evidence(batch[:5])
                main, _ = loss_fn(module.args, features)
                rsed, statistics, masked = module.stroke_evidence_loss(batch, features, output)
                # Use nominal weighting here so the initial gradient audit is meaningful even during warm-up.
                weighted = rsed * module.lambda_rsed

                def gradients(loss):
                    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    return [torch.zeros_like(parameter) if value is None else value.detach()
                            for parameter, value in zip(params, values)]

                grad_main = gradients(main)
                grad_rsed = gradients(weighted)
                groups = {
                    "all": list(range(len(named))),
                    "sketch_prompts": [i for i, (name, _) in enumerate(named) if "sketch_visual_prompt." in name],
                    "photo_prompts": [i for i, (name, _) in enumerate(named) if "photo_visual_prompt." in name],
                    "evidence_head": [i for i, (name, _) in enumerate(named) if "stroke_evidence_head." in name],
                }
                for group, indices in groups.items():
                    if indices:
                        self.gradients.append({
                            "stage": stage,
                            "global_step": int(trainer.global_step),
                            "group": group,
                            **_gradient_stats([grad_main[i] for i in indices], [grad_rsed[i] for i in indices]),
                        })
            record = {
                "stage": stage,
                "global_step": int(trainer.global_step),
                "main_loss": float(main.detach()),
                "rsed_loss": float(rsed.detach()),
                "weighted_rsed_loss": float(weighted.detach()),
                **{key: float(value) for key, value in statistics.items()},
            }
            self.fixed.append(record)
            write_csv(self.out / "fixed_batch.csv", self.fixed)
            write_csv(self.out / "gradient_interaction.csv", self.gradients)
            self.evidence_figure(module, batch, output, masked, stage)
            print("[RSED Fixed Batch]", json.dumps(record, allow_nan=False), flush=True)

    def evidence_figure(self, module, batch, output, masked, stage):
        import matplotlib.pyplot as plt
        target = module._select_rsed_target(batch[5])
        teacher = target["map"].detach().float().cpu()
        student = output["weights"].detach().float().cpu()
        count = min(4, len(student))
        fig, axes = plt.subplots(count, 5, figsize=(15, 3 * count), squeeze=False)
        grid = module.args.rsed_student_grid
        for row in range(count):
            image = batch[1][row].detach().cpu()
            axes[row, 0].imshow(_rgb(image))
            axes[row, 0].set_title("clean sketch")
            for col, values, title in ((1, teacher, "teacher evidence"), (2, student, "student evidence")):
                heat = F.interpolate(values[row].reshape(1, 1, grid, grid), size=image.shape[-2:],
                                     mode="bilinear", align_corners=False)[0, 0]
                axes[row, col].imshow(_rgb(image))
                axes[row, col].imshow(heat.numpy(), cmap="inferno", alpha=0.62)
                axes[row, col].set_title(title)
            difference = (student[row] - teacher[row]).reshape(1, 1, grid, grid)
            difference = F.interpolate(difference, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            limit = max(float(difference.abs().max()), 1e-6)
            axes[row, 3].imshow(difference.numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[row, 3].set_title("student - teacher")
            if masked is not None:
                erased, _ = erase_by_patch_evidence(
                    batch[1][row:row + 1], target["map"][row:row + 1],
                    module.args.rsed_mask_fraction, module.args.rsed_ink_threshold,
                    module.args.rsed_ink_softness,
                )
                axes[row, 4].imshow(_rgb(erased[0]))
                axes[row, 4].set_title("teacher-selected erasure")
            for axis in axes[row]:
                axis.axis("off")
        fig.suptitle(stage + "; fixed seen examples")
        fig.tight_layout()
        fig.savefig(self.out / f"evidence_{stage}.png", dpi=130)
        plt.close(fig)

    def figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        epochs = [row["epoch"] for row in self.epochs]
        for key in ("mAP", "precision", "native_mAP", "native_precision"):
            axes[0, 0].plot(epochs, [100 * row[key] for row in self.epochs], "o-", label=key)
        steps = [row["global_step"] for row in self.fixed]
        for key in ("where", "what", "effect", "anchor"):
            axes[0, 1].plot(steps, [row[key] for row in self.fixed], "o-", label=key)
        for key in ("map_cosine", "what_cosine", "effect_cosine"):
            axes[0, 2].plot(steps, [row[key] for row in self.fixed], "o-", label=key)
        for key in ("student_entropy", "teacher_entropy"):
            axes[1, 0].plot(steps, [row[key] for row in self.fixed], "o-", label=key)
        for key in ("descriptor_native_cosine", "correction_norm"):
            axes[1, 1].plot(steps, [row[key] for row in self.fixed], "o-", label=key)
        for group in ("all", "sketch_prompts", "photo_prompts", "evidence_head"):
            rows = [row for row in self.gradients if row["group"] == group and row["cosine"] is not None]
            if rows:
                axes[1, 2].plot([row["global_step"] for row in rows],
                                [row["cosine"] for row in rows], "o-", label=group)
        titles = (
            "Full unseen retrieval (%)",
            "Fixed-batch component losses",
            "Teacher/student agreement",
            "Evidence entropy",
            "Descriptor intervention",
            "Main vs RSED gradient cosine",
        )
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / "training_diagnostics.png", dpi=160)
        plt.close(fig)
