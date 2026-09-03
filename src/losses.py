"""Losses used by the CLIP feature-distillation SBIR experiment."""

import torch
from torch.nn import functional as F


FEATURE_LOSS_CHOICES = ("mse", "cosine")


def feature_distillation_loss(
    projected_student,
    teacher,
    loss_type="mse",
):
    """Match L2-normalized student and detached teacher image features."""
    if loss_type not in FEATURE_LOSS_CHOICES:
        raise ValueError(
            f"Unsupported feature loss {loss_type!r}; "
            f"expected one of {FEATURE_LOSS_CHOICES}."
        )

    student = F.normalize(projected_student.float(), dim=-1)
    teacher = F.normalize(
        teacher.detach().to(
            device=student.device,
            dtype=torch.float32,
        ),
        dim=-1,
    )
    if student.shape != teacher.shape:
        raise ValueError(
            "Projected student and teacher features must have the same shape; "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}."
        )

    if loss_type == "mse":
        return F.mse_loss(student, teacher)
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def loss_fn(args, features):
    """Compute exactly one feature-distillation objective for both modalities."""
    (
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
    ) = features

    photo_loss = feature_distillation_loss(
        projected_photo,
        teacher_photo,
        args.feature_loss,
    )
    sketch_loss = feature_distillation_loss(
        projected_sketch,
        teacher_sketch,
        args.feature_loss,
    )
    feature_loss = 0.5 * (photo_loss + sketch_loss)
    total_loss = args.lambda_fd * feature_loss
    return total_loss, {
        "fd_photo": photo_loss,
        "fd_sketch": sketch_loss,
        "fd": feature_loss,
    }


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
