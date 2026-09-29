"""Losses used by the visual-and-text interactive-contrastive experiment."""

import torch
from torch.nn import functional as F


def multi_positive_contrastive_loss(
    anchors,
    candidates,
    anchor_labels,
    candidate_labels,
    logit_scale,
):
    """Contrast anchors against detached candidates with class-level positives.

    The numerator sums the probability of every candidate with the same class
    as the anchor. This avoids treating repeated examples of a Sketchy class as
    false negatives, unlike the diagonal-only CLIP objective.
    """
    anchors = F.normalize(anchors.float(), dim=-1)
    candidates = F.normalize(
        candidates.detach().to(device=anchors.device, dtype=torch.float32),
        dim=-1,
    )
    if anchors.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Anchors and candidates must both be 2D tensors.")
    if anchors.shape[-1] != candidates.shape[-1]:
        raise ValueError(
            "Anchors and candidates must have the same feature dimension; "
            f"got {anchors.shape[-1]} and {candidates.shape[-1]}."
        )

    anchor_labels = anchor_labels.detach().to(device=anchors.device).reshape(-1)
    candidate_labels = candidate_labels.detach().to(
        device=anchors.device
    ).reshape(-1)
    if len(anchor_labels) != len(anchors):
        raise ValueError("anchor_labels has the wrong length.")
    if len(candidate_labels) != len(candidates):
        raise ValueError("candidate_labels has the wrong length.")

    positive_mask = anchor_labels[:, None].eq(candidate_labels[None, :])
    valid = positive_mask.any(dim=-1)
    if not valid.all():
        missing = (~valid).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "Every anchor must have at least one positive candidate; "
            f"missing positives for rows {missing}."
        )

    if not torch.is_tensor(logit_scale):
        logit_scale = anchors.new_tensor(logit_scale)
    scale = logit_scale.to(device=anchors.device, dtype=torch.float32)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError("logit_scale must be one finite positive scalar.")

    logits = scale * anchors @ candidates.t()
    log_denominator = torch.logsumexp(logits, dim=-1)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    log_numerator = torch.logsumexp(positive_logits, dim=-1)
    return (log_denominator - log_numerator).mean()


def interactive_contrastive_loss(
    projected_photo,
    projected_sketch,
    teacher_photo,
    teacher_sketch,
    teacher_photo_text,
    teacher_sketch_text,
    labels,
    class_labels,
    logit_scale,
    sketch_to_photo_weight=0.25,
    photo_to_sketch_weight=0.25,
    sketch_to_text_weight=0.25,
    photo_to_text_weight=0.25,
):
    """Match student images to cross-domain images and teacher text targets."""
    weights = (
        sketch_to_photo_weight,
        photo_to_sketch_weight,
        sketch_to_text_weight,
        photo_to_text_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("ICL component weights must be non-negative.")
    if sum(weights) <= 0:
        raise ValueError("At least one ICL component weight must be positive.")

    zero = projected_photo.new_zeros(())
    if sketch_to_photo_weight > 0:
        sketch_to_photo = multi_positive_contrastive_loss(
            projected_sketch,
            teacher_photo,
            labels,
            labels,
            logit_scale,
        )
    else:
        sketch_to_photo = zero

    if photo_to_sketch_weight > 0:
        photo_to_sketch = multi_positive_contrastive_loss(
            projected_photo,
            teacher_sketch,
            labels,
            labels,
            logit_scale,
        )
    else:
        photo_to_sketch = zero

    if sketch_to_text_weight > 0:
        sketch_to_text = multi_positive_contrastive_loss(
            projected_sketch,
            teacher_sketch_text,
            labels,
            class_labels,
            logit_scale,
        )
    else:
        sketch_to_text = zero

    if photo_to_text_weight > 0:
        photo_to_text = multi_positive_contrastive_loss(
            projected_photo,
            teacher_photo_text,
            labels,
            class_labels,
            logit_scale,
        )
    else:
        photo_to_text = zero

    total = (
        sketch_to_photo_weight * sketch_to_photo
        + photo_to_sketch_weight * photo_to_sketch
        + sketch_to_text_weight * sketch_to_text
        + photo_to_text_weight * photo_to_text
    )
    return total, {
        "icl_sketch_to_photo": sketch_to_photo,
        "icl_photo_to_sketch": photo_to_sketch,
        "icl_sketch_to_text": sketch_to_text,
        "icl_photo_to_text": photo_to_text,
    }


def loss_fn(args, features):
    """Compute the isolated visual-and-text student-to-teacher ICL objective."""
    (
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
        teacher_photo_text,
        teacher_sketch_text,
        labels,
        class_labels,
        logit_scale,
    ) = features

    icl_loss, values = interactive_contrastive_loss(
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
        teacher_photo_text,
        teacher_sketch_text,
        labels,
        class_labels,
        logit_scale,
        sketch_to_photo_weight=args.lambda_icl_sketch_to_photo,
        photo_to_sketch_weight=args.lambda_icl_photo_to_sketch,
        sketch_to_text_weight=args.lambda_icl_sketch_to_text,
        photo_to_text_weight=args.lambda_icl_photo_to_text,
    )
    values["icl"] = icl_loss
    return icl_loss, values


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
