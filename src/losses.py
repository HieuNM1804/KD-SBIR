"""Student classification, NT-Xent, and EVA01-g-14 relational distillation loss."""

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


def nt_xent_loss(features_view1, features_view2, temperature=0.07):
    """NT-Xent contrastive loss between paired sketch/photo features."""
    features_view1 = F.normalize(features_view1.float(), dim=-1)
    features_view2 = F.normalize(features_view2.float(), dim=-1)

    batch_size = features_view1.shape[0]
    device = features_view1.device
    features = torch.cat([features_view1, features_view2], dim=0)

    logits = features @ features.t()
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float("-inf"))
    logits = logits / temperature

    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ],
        dim=0,
    ).long()

    return F.cross_entropy(logits, labels)


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
    ) = features

    labels = labels.to(photo_logits.device)
    classification_loss = (
        F.cross_entropy(photo_logits, labels)
        + F.cross_entropy(sketch_logits, labels)
    )

    nt_xent = nt_xent_loss(
        sketch_features,
        photo_features,
        temperature=args.nt_xent_temperature,
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

    total_loss = (
        args.lambda_cls * classification_loss
        + args.lambda_nt_xent * nt_xent
        + args.lambda_kd * kd_loss
    )
    return total_loss, {
        "cls": classification_loss,
        "nt_xent": nt_xent,
        "kd_sketch_photo": kd_loss,
    }
