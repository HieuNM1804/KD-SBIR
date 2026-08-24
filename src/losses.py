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


def image_text_kd_loss(
    student_image,
    student_text,
    teacher_image,
    teacher_text,
    temperature=0.1,
):
    """Symmetrically match teacher and student class distributions."""
    student_image = F.normalize(student_image.float(), dim=-1)
    student_text = F.normalize(student_text.float(), dim=-1)
    student_log_probs = F.log_softmax(
        student_image @ student_text.t() / temperature,
        dim=-1,
    )
    student_probs = student_log_probs.exp()

    with torch.no_grad():
        teacher_image = F.normalize(
            teacher_image.to(student_image.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_text = F.normalize(
            teacher_text.to(student_image.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_log_probs = F.log_softmax(
            teacher_image @ teacher_text.t() / temperature,
            dim=-1,
        )
        teacher_probs = teacher_log_probs.exp()

    teacher_to_student = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    student_to_teacher = (
        student_probs * (student_log_probs - teacher_log_probs)
    ).sum(dim=-1).mean()
    return 0.5 * (teacher_to_student + student_to_teacher)


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


def loss_fn(args, features):
    (
        photo_features,
        sketch_features,
        teacher_photo_features,
        teacher_sketch_features,
        teacher_active,
        student_sketch_text,
        student_photo_text,
        teacher_sketch_text,
        teacher_photo_text,
    ) = features

    zero = torch.zeros((), device=photo_features.device)

    domain_loss = zero
    if teacher_active and args.lambda_domain > 0:
        domain_loss = relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
        )

    photo_text_kd = zero
    sketch_text_kd = zero
    if teacher_active and args.lambda_modality > 0:
        photo_text_kd = image_text_kd_loss(
            photo_features,
            student_photo_text,
            teacher_photo_features,
            teacher_photo_text,
            args.photo_text_kd_temperature,
        )
        sketch_text_kd = image_text_kd_loss(
            sketch_features,
            student_sketch_text,
            teacher_sketch_features,
            teacher_sketch_text,
            args.sketch_text_kd_temperature,
        )
    modality_loss = photo_text_kd + sketch_text_kd

    total_loss = (
        args.lambda_domain * domain_loss
        + args.lambda_modality * modality_loss
    )
    return total_loss, {
        "domain_kd": domain_loss,
        "modality_kd": modality_loss,
    }
