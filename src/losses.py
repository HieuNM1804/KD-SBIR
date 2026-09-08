"""Losses used by the CLIP-KD gradient-distillation experiment."""

import torch
from torch.nn import functional as F


def _validate_temperature(temperature, reference):
    if not torch.is_tensor(temperature):
        temperature = reference.new_tensor(temperature)
    temperature = temperature.to(device=reference.device, dtype=torch.float32)
    if (
        temperature.numel() != 1
        or not torch.isfinite(temperature).all()
        or temperature.item() <= 0
    ):
        raise ValueError("temperature must be one finite positive scalar.")
    return temperature


def multi_positive_targets(anchor_labels, candidate_labels, dtype):
    """Return a row-normalized target over every same-class candidate."""
    anchor_labels = anchor_labels.reshape(-1)
    candidate_labels = candidate_labels.reshape(-1).to(anchor_labels.device)
    positive = anchor_labels[:, None].eq(candidate_labels[None, :])
    counts = positive.sum(dim=-1, keepdim=True)
    if (counts == 0).any():
        missing = (counts.squeeze(-1) == 0).nonzero(as_tuple=False).flatten()
        raise ValueError(
            "Every anchor needs a positive candidate; missing rows "
            f"{missing.tolist()}."
        )
    return positive.to(dtype=dtype) / counts.to(dtype=dtype)


def soft_target_contrastive_loss(
    anchors,
    candidates,
    anchor_labels,
    candidate_labels,
    temperature,
):
    """Mean cross entropy with uniform same-class positive targets."""
    anchors = F.normalize(anchors.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    if anchors.ndim != 2 or candidates.ndim != 2:
        raise ValueError("anchors and candidates must be 2D tensors.")
    if anchors.shape[-1] != candidates.shape[-1]:
        raise ValueError("anchors and candidates must have the same width.")
    if len(anchor_labels) != len(anchors):
        raise ValueError("anchor_labels has the wrong length.")
    if len(candidate_labels) != len(candidates):
        raise ValueError("candidate_labels has the wrong length.")

    temperature = _validate_temperature(temperature, anchors)
    targets = multi_positive_targets(
        anchor_labels.to(anchors.device),
        candidate_labels.to(anchors.device),
        anchors.dtype,
    )
    logits = anchors @ candidates.t() / temperature
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def contrastive_embedding_gradients(
    anchors,
    candidates,
    anchor_labels,
    candidate_labels,
    temperature,
):
    """Return analytic derivatives of directional contrastive cross entropy.

    The outputs are dL/d(anchor embedding) and dL/d(candidate embedding), where
    L is mean soft-target contrastive CE and the variables are the normalized
    embedding vectors. The expression remains differentiable so the student can
    learn from it without autograd.grad(create_graph=True).
    """
    anchors = F.normalize(anchors.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    if anchors.ndim != 2 or candidates.ndim != 2:
        raise ValueError("anchors and candidates must be 2D tensors.")
    if anchors.shape[-1] != candidates.shape[-1]:
        raise ValueError("anchors and candidates must have the same width.")
    if len(anchor_labels) != len(anchors):
        raise ValueError("anchor_labels has the wrong length.")
    if len(candidate_labels) != len(candidates):
        raise ValueError("candidate_labels has the wrong length.")

    temperature = _validate_temperature(temperature, anchors)
    targets = multi_positive_targets(
        anchor_labels.to(anchors.device),
        candidate_labels.to(anchors.device),
        anchors.dtype,
    )
    probabilities = F.softmax(anchors @ candidates.t() / temperature, dim=-1)
    residual = probabilities - targets
    batch_size = anchors.shape[0]
    grad_anchors = residual @ candidates / (batch_size * temperature)
    grad_candidates = residual.t() @ anchors / (batch_size * temperature)
    return grad_anchors, grad_candidates


def _batch_mean_squared_l2(student, teacher):
    """Implement 1/B sum ||student_i - teacher_i||_2^2 from the paper."""
    return (student - teacher).square().sum(dim=-1).mean()


def gradient_distillation_loss(
    projected_photo,
    projected_sketch,
    teacher_photo,
    teacher_sketch,
    labels,
    temperature,
):
    """Match all four sketch/photo contrastive embedding-gradient roles."""
    teacher_photo = teacher_photo.detach().to(
        device=projected_photo.device,
        dtype=torch.float32,
    )
    teacher_sketch = teacher_sketch.detach().to(
        device=projected_sketch.device,
        dtype=torch.float32,
    )
    labels = labels.detach().to(projected_photo.device)

    if projected_photo.shape != teacher_photo.shape:
        raise ValueError("Projected photo and teacher photo shapes must match.")
    if projected_sketch.shape != teacher_sketch.shape:
        raise ValueError("Projected sketch and teacher sketch shapes must match.")

    # Direction 1: sketch anchors retrieve photo candidates.
    student_sketch_anchor, student_photo_key = (
        contrastive_embedding_gradients(
            projected_sketch,
            projected_photo,
            labels,
            labels,
            temperature,
        )
    )
    with torch.no_grad():
        teacher_sketch_anchor, teacher_photo_key = (
            contrastive_embedding_gradients(
                teacher_sketch,
                teacher_photo,
                labels,
                labels,
                temperature,
            )
        )

    # Direction 2: photo anchors retrieve sketch candidates.
    student_photo_anchor, student_sketch_key = (
        contrastive_embedding_gradients(
            projected_photo,
            projected_sketch,
            labels,
            labels,
            temperature,
        )
    )
    with torch.no_grad():
        teacher_photo_anchor, teacher_sketch_key = (
            contrastive_embedding_gradients(
                teacher_photo,
                teacher_sketch,
                labels,
                labels,
                temperature,
            )
        )

    terms = {
        "gd_sketch_anchor": _batch_mean_squared_l2(
            student_sketch_anchor, teacher_sketch_anchor
        ),
        "gd_photo_key": _batch_mean_squared_l2(
            student_photo_key, teacher_photo_key
        ),
        "gd_photo_anchor": _batch_mean_squared_l2(
            student_photo_anchor, teacher_photo_anchor
        ),
        "gd_sketch_key": _batch_mean_squared_l2(
            student_sketch_key, teacher_sketch_key
        ),
    }
    return sum(terms.values()), terms


def loss_fn(args, features):
    """Combine the student's SBIR task loss with gradient distillation."""
    (
        photo,
        sketch,
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
        labels,
    ) = features

    task_sketch_to_photo = soft_target_contrastive_loss(
        sketch,
        photo,
        labels,
        labels,
        args.task_temperature,
    )
    task_photo_to_sketch = soft_target_contrastive_loss(
        photo,
        sketch,
        labels,
        labels,
        args.task_temperature,
    )
    task_loss = 0.5 * (task_sketch_to_photo + task_photo_to_sketch)

    gd_loss, values = gradient_distillation_loss(
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
        labels,
        args.gd_temperature,
    )
    values.update(
        {
            "task_sketch_to_photo": task_sketch_to_photo,
            "task_photo_to_sketch": task_photo_to_sketch,
            "task": task_loss,
            "gd": gd_loss,
        }
    )
    total = args.lambda_task * task_loss + args.lambda_gd * gd_loss
    return total, values


def batch_hard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric batch-hard triplet loss for teacher prompt pretraining."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    labels = labels.to(sketch_features.device)

    distance = 1.0 - sketch_features @ photo_features.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask

    def one_direction(dist):
        valid_negative = negative_mask.any(dim=-1)
        hardest_positive = dist.masked_fill(
            ~positive_mask, -torch.inf
        ).max(dim=-1).values
        hardest_negative = dist.masked_fill(
            ~negative_mask, torch.inf
        ).min(dim=-1).values
        losses = F.relu(hardest_positive - hardest_negative + margin)
        if valid_negative.any():
            return losses[valid_negative].mean()
        return dist.new_zeros(())

    return 0.5 * (one_direction(distance) + one_direction(distance.t()))
