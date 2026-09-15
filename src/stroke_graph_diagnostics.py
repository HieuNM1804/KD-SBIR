"""Non-mutating training diagnostics for SGCD."""

import csv
import json
import math
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch.nn import functional as F
from torch.utils.data import default_collate

from src.stroke_graph import CLIP_MEAN, CLIP_STD, erase_by_patch_evidence
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


def _gradient_stats(a, b):
    norm_a = math.sqrt(sum(value.float().square().sum().item() for value in a))
    norm_b = math.sqrt(sum(value.float().square().sum().item() for value in b))
    dot = sum((x.float() * y.float()).sum().item() for x, y in zip(a, b))
    return {
        "main_norm": norm_a,
        "weighted_sgcd_norm": norm_b,
        "sgcd_over_main": norm_b / norm_a if norm_a > 1e-12 else None,
        "cosine": dot / (norm_a * norm_b) if norm_a * norm_b > 1e-20 else None,
    }


class StrokeGraphDiagnostics(Callback):
    def _setup(self, trainer, module):
        if hasattr(self, "out"):
            return
        self.out = Path(trainer.log_dir or trainer.default_root_dir) / "sgcd_diagnostics"
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
        count = min(module.args.sgcd_diagnostic_batch_size, len(dataset))
        self.indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        self.batch = default_collate([dataset[(0, index)] for index in self.indices])
        (self.out / "fixed_batch.json").write_text(
            json.dumps({"indices": self.indices, "sample_epoch": 0,
                        "notes": "Fixed seen training batch; measurements do not update parameters."}, indent=2),
            encoding="utf-8",
        )
        self.measure(trainer, module, "initial")

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking or not getattr(module, "_sgcd_last_validation", None):
            return
        self._setup(trainer, module)
        row = {
            "epoch": int(trainer.current_epoch) + (1 if trainer.global_step else 0),
            "global_step": int(trainer.global_step),
            **module._sgcd_last_validation,
        }
        self.epochs.append(row)
        write_csv(self.out / "epochs.csv", self.epochs)
        if trainer.global_step and hasattr(self, "batch"):
            self.measure(trainer, module, f"epoch_{row['epoch']}")
        self.figure()
        print(
            f"[SGCD Diagnostics] epoch={row['epoch']} deployed/native mAP="
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
                features, output = module.model.forward_with_stroke_graph(batch[:5])
                main, _ = loss_fn(module.args, features)
                sgcd, statistics, masked = module.stroke_graph_loss(batch, features, output)
                # Use nominal weighting here so the initial gradient audit is meaningful even during warm-up.
                weighted = sgcd * module.lambda_sgcd

                def gradients(loss):
                    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    return [torch.zeros_like(parameter) if value is None else value.detach()
                            for parameter, value in zip(params, values)]

                grad_main = gradients(main)
                grad_sgcd = gradients(weighted)
                groups = {
                    "all": list(range(len(named))),
                    "sketch_prompts": [i for i, (name, _) in enumerate(named) if "sketch_visual_prompt." in name],
                    "photo_prompts": [i for i, (name, _) in enumerate(named) if "photo_visual_prompt." in name],
                    "evidence_head": [i for i, (name, _) in enumerate(named) if "stroke_graph_head." in name],
                }
                for group, indices in groups.items():
                    if indices:
                        self.gradients.append({
                            "stage": stage,
                            "global_step": int(trainer.global_step),
                            "group": group,
                            **_gradient_stats([grad_main[i] for i in indices], [grad_sgcd[i] for i in indices]),
                        })
            record = {
                "stage": stage,
                "global_step": int(trainer.global_step),
                "main_loss": float(main.detach()),
                "sgcd_loss": float(sgcd.detach()),
                "weighted_sgcd_loss": float(weighted.detach()),
                **{key: float(value) for key, value in statistics.items()},
            }
            self.fixed.append(record)
            write_csv(self.out / "fixed_batch.csv", self.fixed)
            write_csv(self.out / "gradient_interaction.csv", self.gradients)
            self.evidence_figure(module, batch, output, masked, stage)
            print("[SGCD Fixed Batch]", json.dumps(record, allow_nan=False), flush=True)

    def evidence_figure(self, module, batch, output, masked, stage):
        import matplotlib.pyplot as plt
        target = module._select_sgcd_target(batch[5])
        teacher = target["map"].detach().float().cpu()
        student = output["weights"].detach().float().cpu()
        count = min(4, len(student))
        fig, axes = plt.subplots(count, 5, figsize=(15, 3 * count), squeeze=False)
        grid = module.args.sgcd_student_grid
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
                    batch[1][row:row + 1], target["mask_priority"][row:row + 1],
                    module.args.sgcd_mask_fraction, module.args.sgcd_ink_threshold,
                    module.args.sgcd_ink_softness,
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
            "Main vs SGCD gradient cosine",
        )
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / "training_diagnostics.png", dpi=160)
        plt.close(fig)
