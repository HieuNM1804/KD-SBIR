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


def standard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric paired triplet loss without feature-based mining."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    labels = labels.to(sketch_features.device)

    batch_size = labels.numel()
    if batch_size < 2:
        return sketch_features.new_zeros(())

    indices = torch.arange(batch_size, device=labels.device)
    offsets = torch.arange(1, batch_size, device=labels.device)
    candidates = (indices[:, None] + offsets[None, :]) % batch_size
    is_negative = labels[candidates].ne(labels[:, None])
    valid = is_negative.any(dim=-1)
    if not valid.any():
        return sketch_features.new_zeros(())

    first_negative = is_negative.to(torch.int64).argmax(dim=-1)
    negative_indices = candidates.gather(
        1,
        first_negative[:, None],
    ).squeeze(1)
    anchors = indices[valid]
    negatives = negative_indices[valid]

    sketch_to_photo = F.triplet_margin_loss(
        sketch_features[anchors],
        photo_features[anchors],
        photo_features[negatives],
        margin=margin,
        p=2,
    )
    photo_to_sketch = F.triplet_margin_loss(
        photo_features[anchors],
        sketch_features[anchors],
        sketch_features[negatives],
        margin=margin,
        p=2,
    )
    return 0.5 * (sketch_to_photo + photo_to_sketch)


def teacher_semantic_loss(
    sketch_features,
    photo_features,
    labels,
    sketch_text,
    photo_text,
    temperature,
):
    return 0.5 * (
        F.cross_entropy(sketch_features @ sketch_text.t() / temperature, labels)
        + F.cross_entropy(photo_features @ photo_text.t() / temperature, labels)
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
        teacher_triplet_loss = standard_teacher_triplet_loss(
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
        + args.lambda_kd * kd_loss
        + args.lambda_teacher_retrieval * teacher_triplet_loss
        + args.lambda_teacher_semantic * teacher_semantic
    )
    return total_loss, {
        "cls": classification_loss,
        "kd_sketch_photo": kd_loss,
        "teacher_triplet": teacher_triplet_loss,
        "teacher_semantic": teacher_semantic,
    }
