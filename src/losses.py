"""Losses used by the CLIP masked-feature-distillation experiment."""

import torch
from torch.nn import functional as F


def masked_feature_distillation_loss(
    projected_student,
    teacher,
):
    """MSE-match a masked student's normalized feature to its full teacher."""
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

    return F.mse_loss(student, teacher)


def loss_fn(args, features):
    """Compute MFD for masked photo and sketch student features."""
    (
        projected_photo,
        projected_sketch,
        teacher_photo,
        teacher_sketch,
    ) = features

    photo_loss = masked_feature_distillation_loss(
        projected_photo,
        teacher_photo,
    )
    sketch_loss = masked_feature_distillation_loss(
        projected_sketch,
        teacher_sketch,
    )
    masked_feature_loss = 0.5 * (photo_loss + sketch_loss)
    total_loss = args.lambda_mfd * masked_feature_loss
    return total_loss, {
        "mfd_photo": photo_loss,
        "mfd_sketch": sketch_loss,
        "mfd": masked_feature_loss,
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
