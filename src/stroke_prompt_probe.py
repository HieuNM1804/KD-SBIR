"""Short localization capacity check; restore all prompt state afterwards."""

import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.stroke_graph import evidence_entropy, hellinger_loss


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def localization_probe(
    model, images, targets, confidence, steps=60, lr=0.01, optimizer_name="sgd"
):
    """Overfit a fixed seen batch using only the sketch prompts and where loss.

    SGD uses the main optimizer settings; Adam is an optional capacity check.
    This does not select training hyperparameters or establish retrieval gains.
    No checkpoint is written.
    """
    if model.retrieval_head != "sgcd" or model.sgcd_student_mode != "native_prompt":
        raise ValueError("Localization probe requires native_prompt mode")
    if steps < 1 or lr <= 0 or not torch.isfinite(torch.tensor(lr)):
        raise ValueError("Probe steps and learning rate must be positive and finite")
    if optimizer_name not in ("sgd", "adam"):
        raise ValueError("Probe optimizer must be sgd or adam")
    if model.stroke_graph_head is not None:
        raise ValueError("The probe must not contain an evidence head")
    if any(p.requires_grad for p in model.clip_model.parameters()):
        raise ValueError("CLIP backbone must remain frozen")
    named = list(model.sketch_visual_prompt.named_parameters())
    if not named:
        raise ValueError("No trainable sketch prompts")
    allowed = {id(p) for p in model.sketch_visual_prompt.parameters()}
    allowed.update(id(p) for p in model.photo_visual_prompt.parameters())
    if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
        raise ValueError("Unexpected trainable parameters outside the visual prompts")
    if confidence.sum() <= 0 or not torch.isfinite(confidence).all():
        raise ValueError("Probe batch has no finite positive confidence")

    saved = [(p, p.detach().clone(), p.grad) for _, p in named]
    modes = [(child, child.training) for child in model.modules()]
    frozen_before = tensor_hash(model.clip_model.state_dict().items())
    photo_before = tensor_hash(model.photo_visual_prompt.state_dict().items())
    rows, gradients = [], []
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode(False):
            model.eval()
            images = images.detach().clone().to(device)
            targets = targets.detach().clone().to(device).float()
            confidence = confidence.detach().clone().to(device).float()
            parameters = [p for _, p in named]
            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(
                    parameters, lr=lr, momentum=0.9, weight_decay=5e-4
                )
            else:
                optimizer = torch.optim.Adam(parameters, lr=lr)
            with torch.no_grad():
                initial_native = model.encode_student_image(images, "sketch").float()
            for step in range(steps + 1):
                with torch.enable_grad():
                    output = model.encode_student_image_details(images, "sketch")
                    loss = hellinger_loss(output["weights"], targets, confidence)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite localization loss")
                    row = {
                        "step": step,
                        "where": loss.item(),
                        "map_cosine": F.cosine_similarity(
                            output["weights"], targets, dim=-1
                        )
                        .mean()
                        .item(),
                        "student_entropy": evidence_entropy(output["weights"])
                        .mean()
                        .item(),
                        "teacher_entropy": evidence_entropy(targets).mean().item(),
                        "native_initial_cosine": F.cosine_similarity(
                            output["native"].float(), initial_native, dim=-1
                        )
                        .mean()
                        .item(),
                    }
                    rows.append(row)
                    if step % 10 == 0 or step == steps:
                        print(
                            "[Prompt Probe]",
                            json.dumps(row, allow_nan=False),
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for name, parameter in named:
                        gradient = parameter.grad
                        if gradient is not None and not torch.isfinite(gradient).all():
                            raise RuntimeError(f"Nonfinite prompt gradient: {name}")
                        if step in (0, steps):
                            gradients.append(
                                {
                                    "step": step,
                                    "parameter": name,
                                    "gradient_norm": 0.0
                                    if gradient is None
                                    else gradient.norm().item(),
                                }
                            )
                    if step == 0:
                        initial_weights = output["weights"].detach().cpu()
                    if step < steps:
                        optimizer.step()
            final_weights = output["weights"].detach().cpu()
            trained_prompts = tensor_hash((name, p) for name, p in named)
    finally:
        with torch.no_grad():
            for parameter, value, grad in saved:
                parameter.copy_(value)
                parameter.grad = grad
        for child, training in modes:
            child.training = training

    frozen_after = tensor_hash(model.clip_model.state_dict().items())
    photo_after = tensor_hash(model.photo_visual_prompt.state_dict().items())
    restored = all(torch.equal(p, value) for p, value, _ in saved)
    if frozen_before != frozen_after or photo_before != photo_after or not restored:
        raise RuntimeError("Probe modified frozen state or failed to restore prompts")
    summary = {
        "scope": "fixed-batch localization capacity; no retrieval evaluation",
        "optimizer": optimizer_name,
        "lr": lr,
        "steps": steps,
        "samples": len(images),
        "initial": rows[0],
        "final": rows[-1],
        "relative_where_reduction": (rows[0]["where"] - rows[-1]["where"])
        / max(rows[0]["where"], 1e-8),
        "learning_observed": rows[-1]["where"] < rows[0]["where"] - 1e-6,
        "trainable_head_parameters": 0,
        "frozen_backbone_sha256": frozen_after,
        "backbone_unchanged": True,
        "photo_prompts_unchanged": True,
        "sketch_prompts_restored": restored,
        "trained_sketch_prompts_sha256": trained_prompts,
    }
    return summary, rows, gradients, initial_weights, final_weights


def save_probe(out, result, images, targets, provenance):
    """Write separate JSON/CSV/PNG files, including before/after evidence maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.stroke_graph_diagnostics import _rgb

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary, rows, gradients, before, after = result
    summary = {**summary, "provenance": provenance}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name, values in (
        ("localization.csv", rows),
        ("prompt_gradients.csv", gradients),
    ):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key in zip(axes, ("where", "map_cosine", "native_initial_cosine")):
        axis.plot([r["step"] for r in rows], [r[key] for r in rows])
        axis.set(xlabel="Probe step", ylabel=key)
        axis.grid(alpha=0.2)
    fig.suptitle("Fixed-batch localization probe; no retrieval claim")
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=160)
    plt.close(fig)
    count = min(6, len(images))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    grid = int(before.shape[-1] ** 0.5)
    for row in range(count):
        rgb = _rgb(images[row])
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Sketch")
        values = (targets[row].detach().cpu(), before[row], after[row])
        vmax = max(v.max().item() for v in values)
        for col, value, title in zip(
            range(1, 4), values, ("Target", "Before", "After")
        ):
            heat = F.interpolate(
                value.float().view(1, 1, grid, grid),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            axes[row, col].imshow(rgb)
            axes[row, col].imshow(heat, cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
            axes[row, col].set_title(title)
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(out / "evidence.png", dpi=160)
    plt.close(fig)
    return summary
