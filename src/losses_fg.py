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


def image_conditioned_text_classification_loss(
    image_features,
    conditioned_target_text,
    fixed_class_text,
    class_labels,
    temperature=0.07,
):
    """Classify images using a patch-conditioned true-class text prompt.

    The fixed text bank supplies negatives for every seen class. Each image's
    target column is replaced by its image-conditioned text score, which keeps
    category-pure fine-grained batches compatible with all-class CE.
    """
    if temperature <= 0:
        raise ValueError("Classification temperature must be greater than zero.")
    images = F.normalize(image_features.float(), dim=-1)
    targets = F.normalize(conditioned_target_text.float(), dim=-1)
    class_text = F.normalize(
        fixed_class_text.detach().to(images.device, dtype=torch.float32),
        dim=-1,
    )
    labels = class_labels.long().to(images.device).reshape(-1)

    if images.ndim != 2 or targets.shape != images.shape:
        raise ValueError("Image and conditioned text features must match in 2D.")
    if class_text.ndim != 2 or class_text.shape[1] != images.shape[1]:
        raise ValueError("The fixed class text bank has an incompatible width.")
    if len(labels) != len(images):
        raise ValueError("Every image needs one class label.")
    if labels.numel() and (
        labels.min().item() < 0 or labels.max().item() >= len(class_text)
    ):
        raise ValueError("A class label is outside the fixed text bank.")

    logits = images @ class_text.t() / temperature
    conditioned_scores = (images * targets).sum(dim=-1) / temperature
    logits = logits.scatter(1, labels[:, None], conditioned_scores[:, None])
    return F.cross_entropy(logits, labels)


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
