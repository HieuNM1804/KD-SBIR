"""Losses used by the DFN5B-to-CLIP SBIR benchmark."""

import torch
from torch.nn import functional as F


def relational_kd_loss(
    student_sketch,
    student_photo,
    teacher_sketch,
    teacher_photo,
    temperature=0.07,
):
    """Match the student and teacher sketch-to-photo similarity distributions."""
    student_device = student_sketch.device
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(student_photo.float(), dim=-1)

    student_logits = student_sketch @ student_photo.t() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    with torch.no_grad():
        teacher_sketch = F.normalize(
            teacher_sketch.to(device=student_device, dtype=torch.float32),
            dim=-1,
        )
        teacher_photo = F.normalize(
            teacher_photo.to(device=student_device, dtype=torch.float32),
            dim=-1,
        )
        teacher_logits = teacher_sketch @ teacher_photo.t() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def nt_xent_loss(
    features_view1,
    features_view2,
    temperature=0.07,
):
    """Symmetric SimCLR NT-Xent between paired sketch/photo features."""
    features_view1 = F.normalize(features_view1.float(), dim=-1)
    features_view2 = F.normalize(features_view2.float(), dim=-1)

    batch_size = features_view1.shape[0]
    device = features_view1.device
    features = torch.cat([features_view1, features_view2], dim=0)

    logits = features @ features.t()
    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=device,
    )
    logits = logits.masked_fill(self_mask, float("-inf"))
    logits = logits / temperature

    labels = torch.cat(
        [
            torch.arange(
                batch_size,
                2 * batch_size,
                device=device,
            ),
            torch.arange(batch_size, device=device),
        ],
        dim=0,
    )
    return F.cross_entropy(logits, labels)


def batch_hard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric batch-hard triplet loss for jointly trained teacher adapters."""
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


def teacher_semantic_loss(
    sketch_features,
    photo_features,
    labels,
    sketch_text,
    photo_text,
    temperature,
):
    return 0.5 * (
        F.cross_entropy(
            sketch_features @ sketch_text.t() / temperature,
            labels,
        )
        + F.cross_entropy(
            photo_features @ photo_text.t() / temperature,
            labels,
        )
    )


def loss_fn(args, features):
    (
        photo_features,
        sketch_features,
        teacher_photo_features,
        teacher_sketch_features,
        labels,
        photo_logits,
        sketch_logits,
        teacher_active,
        joint_teacher_adapter,
        teacher_sketch_text,
        teacher_photo_text,
    ) = features

    labels = labels.to(photo_logits.device)
    classification_loss = (
        F.cross_entropy(photo_logits, labels)
        + F.cross_entropy(sketch_logits, labels)
    )

    nt_xent = torch.zeros((), device=photo_logits.device)
    if args.lambda_nt_xent > 0:
        nt_xent = nt_xent_loss(
            sketch_features,
            photo_features,
            args.nt_xent_temperature,
        )

    kd_loss = torch.zeros((), device=photo_logits.device)
    if teacher_active and args.lambda_kd > 0:
        kd_loss = relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
        )

    teacher_triplet_loss = torch.zeros((), device=photo_logits.device)
    teacher_semantic = torch.zeros((), device=photo_logits.device)
    if joint_teacher_adapter:
        teacher_triplet_loss = batch_hard_teacher_triplet_loss(
            teacher_sketch_features,
            teacher_photo_features,
            labels,
            args.teacher_triplet_margin,
        )
        teacher_semantic = teacher_semantic_loss(
            teacher_sketch_features,
            teacher_photo_features,
            labels,
            teacher_sketch_text,
            teacher_photo_text,
            args.teacher_temperature,
        )

    total_loss = (
        args.lambda_cls * classification_loss
        + args.lambda_nt_xent * nt_xent
        + args.lambda_kd * kd_loss
        + args.lambda_teacher_retrieval * teacher_triplet_loss
        + args.lambda_teacher_semantic * teacher_semantic
    )
    return total_loss, {
        "cls": classification_loss,
        "nt_xent": nt_xent,
        "kd_sketch_photo": kd_loss,
        "teacher_triplet": teacher_triplet_loss,
        "teacher_semantic": teacher_semantic,
    }
