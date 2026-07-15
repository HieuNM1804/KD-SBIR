"""Losses used by the DFN5B-to-CoPrompt SBIR benchmark."""

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
    teacher_sketch = F.normalize(
        teacher_sketch.to(device=student_device, dtype=torch.float32), dim=-1
    )
    teacher_photo = F.normalize(
        teacher_photo.to(device=student_device, dtype=torch.float32), dim=-1
    )

    student_logits = student_sketch @ student_photo.t() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    with torch.no_grad():
        teacher_logits = teacher_sketch @ teacher_photo.t() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def generalized_text_kd_loss(
    student_sketch_text,
    student_photo_text,
    teacher_sketch_text,
    teacher_photo_text,
    manual_sketch_text,
    manual_photo_text,
    temperature=0.07,
    anchor_weight=0.5,
):
    """Preserve teacher text relations while anchoring prompts to frozen CLIP text."""

    def relation_loss(student_a, student_b, teacher_a, teacher_b):
        device = student_a.device
        student_a = F.normalize(student_a.float(), dim=-1)
        student_b = F.normalize(student_b.float(), dim=-1)
        teacher_a = F.normalize(
            teacher_a.to(device=device, dtype=torch.float32), dim=-1
        )
        teacher_b = F.normalize(
            teacher_b.to(device=device, dtype=torch.float32), dim=-1
        )
        student_log_probs = F.log_softmax(
            student_a @ student_b.t() / temperature, dim=-1
        )
        with torch.no_grad():
            teacher_probs = F.softmax(
                teacher_a @ teacher_b.t() / temperature, dim=-1
            )
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    def anchor_loss(student_text, manual_text):
        manual_text = manual_text.to(
            device=student_text.device, dtype=torch.float32
        )
        return 1.0 - F.cosine_similarity(
            F.normalize(student_text.float(), dim=-1),
            F.normalize(manual_text, dim=-1),
            dim=-1,
        ).mean()

    text_relation = (
        relation_loss(
            student_sketch_text,
            student_sketch_text,
            teacher_sketch_text,
            teacher_sketch_text,
        )
        + relation_loss(
            student_photo_text,
            student_photo_text,
            teacher_photo_text,
            teacher_photo_text,
        )
        + relation_loss(
            student_sketch_text,
            student_photo_text,
            teacher_sketch_text,
            teacher_photo_text,
        )
    ) / 3.0
    text_anchor = 0.5 * (
        anchor_loss(student_sketch_text, manual_sketch_text)
        + anchor_loss(student_photo_text, manual_photo_text)
    )
    return text_relation + anchor_weight * text_anchor, text_relation, text_anchor


def batch_hard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric batch-hard triplet loss for the jointly trained teacher adapters."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    labels = labels.to(sketch_features.device)

    distance = 1.0 - sketch_features @ photo_features.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask

    def one_direction(dist):
        valid_negative = negative_mask.any(dim=-1)
        hardest_positive = dist.masked_fill(~positive_mask, -torch.inf).max(dim=-1).values
        hardest_negative = dist.masked_fill(~negative_mask, torch.inf).min(dim=-1).values
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
        teacher_active,
        joint_teacher_adapter,
        teacher_sketch_text,
        teacher_photo_text,
        student_sketch_text,
        student_photo_text,
        manual_sketch_text,
        manual_photo_text,
    ) = features

    labels = labels.to(photo_features.device)

    kd_loss = torch.zeros((), device=photo_features.device)
    if teacher_active and args.lambda_kd > 0:
        kd_loss = relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
        )

    text_kd_loss = torch.zeros((), device=photo_features.device)
    text_relation = torch.zeros((), device=photo_features.device)
    text_anchor = torch.zeros((), device=photo_features.device)
    if teacher_active and args.lambda_text_kd > 0:
        text_kd_loss, text_relation, text_anchor = generalized_text_kd_loss(
            student_sketch_text,
            student_photo_text,
            teacher_sketch_text,
            teacher_photo_text,
            manual_sketch_text,
            manual_photo_text,
            args.text_kd_temperature,
            args.text_anchor_weight,
        )

    teacher_triplet_loss = torch.zeros((), device=photo_features.device)
    teacher_semantic = torch.zeros((), device=photo_features.device)
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
        args.lambda_kd * kd_loss
        + args.lambda_text_kd * text_kd_loss
        + args.lambda_teacher_retrieval * teacher_triplet_loss
        + args.lambda_teacher_semantic * teacher_semantic
    )
    return total_loss, {
        "kd_sketch_photo": kd_loss,
        "text_kd": text_kd_loss,
        "text_relation": text_relation,
        "text_anchor": text_anchor,
        "teacher_triplet": teacher_triplet_loss,
        "teacher_semantic": teacher_semantic,
    }
