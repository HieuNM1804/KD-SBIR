import torch
from torch.nn import functional as F

from src.losses import image_text_kd_loss, relational_kd_loss


def fine_grained_teacher_infonce_loss(
    sketch_features,
    photo_features,
    target_photo_indices,
    temperature=0.07,
):
    """Exact-instance InfoNCE over a category's complete photo gallery."""
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be greater than zero.")
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    targets = target_photo_indices.to(sketch_features.device).long()
    if photo_features.shape[0] != 100:
        raise RuntimeError("Fine-grained training expects a 100-photo gallery.")
    if targets.shape[0] != sketch_features.shape[0]:
        raise RuntimeError("Every sketch query needs one exact photo target.")
    logits = sketch_features @ photo_features.t() / temperature
    return F.cross_entropy(logits, targets)


def hardest_wrong_instance_features(
    sketch_features,
    photo_features,
    target_photo_indices,
):
    """Select the most similar non-target photo from the category gallery."""
    targets = target_photo_indices.to(sketch_features.device).long()
    if photo_features.shape[0] < 2:
        raise ValueError("Hard-negative selection needs at least two photos.")
    if targets.shape != (sketch_features.shape[0],):
        raise ValueError("Every sketch needs one target photo index.")
    if (
        targets.min().item() < 0
        or targets.max().item() >= photo_features.shape[0]
    ):
        raise ValueError("Target photo index is outside the gallery.")

    with torch.no_grad():
        similarities = F.normalize(sketch_features.detach().float(), dim=-1) @ (
            F.normalize(photo_features.detach().float(), dim=-1).t()
        )
        similarities.scatter_(1, targets[:, None], -torch.inf)
        negative_indices = similarities.argmax(dim=1)
    return photo_features[negative_indices], negative_indices


def conditional_cross_modal_jigsaw_loss(
    solver,
    sketch_features,
    shuffled_sketch_features,
    positive_photo_features,
    negative_photo_features,
    permutation_labels,
    hinge_margin=0.0,
):
    """SpLIP-style conditional jigsaw CE plus positive/negative ranking hinge."""
    if hinge_margin < 0:
        raise ValueError("Jigsaw hinge margin must be non-negative.")
    labels = permutation_labels.to(sketch_features.device).long()
    anchor_logits = solver(sketch_features, shuffled_sketch_features)
    positive_logits = solver(positive_photo_features, shuffled_sketch_features)
    negative_logits = solver(negative_photo_features, shuffled_sketch_features)

    anchor_ce = F.cross_entropy(anchor_logits, labels)
    positive_ce = F.cross_entropy(positive_logits, labels, reduction="none")
    negative_ce = F.cross_entropy(negative_logits, labels, reduction="none")
    hinge_values = F.relu(positive_ce - negative_ce + hinge_margin)
    loss = anchor_ce + hinge_values.mean()
    metrics = {
        "anchor_ce": anchor_ce.detach(),
        "hinge": hinge_values.mean().detach(),
        "accuracy": (
            anchor_logits.argmax(dim=1).eq(labels).float().mean().detach()
        ),
        "active_hinge": hinge_values.gt(0).float().mean().detach(),
    }
    return loss, metrics


def full_gallery_relational_kd_loss(
    student_sketch,
    student_photo,
    teacher_sketch,
    teacher_photo,
    temperature=0.07,
):
    """Match sketch-to-photo distributions over all 100 category photos."""
    if student_photo.shape[0] != 100 or teacher_photo.shape[0] != 100:
        raise RuntimeError("Domain KD requires the full 100-photo gallery.")
    return relational_kd_loss(
        student_sketch,
        student_photo,
        teacher_sketch,
        teacher_photo,
        temperature,
    )


def fine_grained_distillation_loss(args, features):
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

    zero = photo_features.new_zeros((), dtype=torch.float32)
    domain_loss = zero
    if teacher_active and args.lambda_domain > 0:
        domain_loss = full_gallery_relational_kd_loss(
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
    total = args.lambda_domain * domain_loss + args.lambda_modality * modality_loss
    return total, {
        "domain_kd": domain_loss,
        "modality_kd": modality_loss,
    }
