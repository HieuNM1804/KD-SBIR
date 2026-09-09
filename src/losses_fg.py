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
    logits = (sketch_features @ photo_features.t()).float() / temperature
    return F.cross_entropy(logits, targets)


def fine_grained_prompt_infonce_loss(
    sketch_image_features,
    photo_image_features,
    sketch_prompt_text_features,
    photo_prompt_text_features,
    target_photo_indices,
    temperature=0.07,
):
    """Exact-instance image/text InfoNCE over the 100-photo gallery.

    The two cross-domain directions are sketch image -> photo-conditioned text
    and sketch-conditioned text -> photo image. Both use the paired photo index
    as target, so the objective preserves instance discrimination within a
    category rather than collapsing all photos into one class prototype.
    """
    if temperature <= 0:
        raise ValueError("Prompt InfoNCE temperature must be greater than zero.")

    sketch_images = F.normalize(sketch_image_features.float(), dim=-1)
    photo_images = F.normalize(photo_image_features.float(), dim=-1)
    sketch_text = F.normalize(sketch_prompt_text_features.float(), dim=-1)
    photo_text = F.normalize(photo_prompt_text_features.float(), dim=-1)
    targets = target_photo_indices.to(sketch_images.device).long().reshape(-1)

    if photo_images.ndim != 2 or photo_images.shape[0] != 100:
        raise RuntimeError("Prompt InfoNCE requires a 100-photo image gallery.")
    if photo_text.shape != photo_images.shape:
        raise ValueError("Photo image/text gallery features must match.")
    if sketch_images.ndim != 2 or sketch_text.shape != sketch_images.shape:
        raise ValueError("Sketch image/text query features must match.")
    if sketch_images.shape[1] != photo_images.shape[1]:
        raise ValueError("Sketch and photo features have incompatible widths.")
    if len(targets) != len(sketch_images):
        raise ValueError("Every sketch query needs one exact photo target.")
    if targets.numel() and (
        targets.min().item() < 0 or targets.max().item() >= len(photo_images)
    ):
        raise ValueError("A target lies outside the 100-photo gallery.")

    # Keep logits in FP32 under CUDA autocast for stable cross-entropy.
    sketch_to_photo_text_logits = (
        sketch_images @ photo_text.t()
    ).float() / temperature
    sketch_text_to_photo_logits = (
        sketch_text @ photo_images.t()
    ).float() / temperature
    sketch_to_photo_text = F.cross_entropy(
        sketch_to_photo_text_logits,
        targets,
    )
    sketch_text_to_photo = F.cross_entropy(
        sketch_text_to_photo_logits,
        targets,
    )
    loss = 0.5 * (sketch_to_photo_text + sketch_text_to_photo)
    return loss, {
        "sketch_to_photo_text": sketch_to_photo_text,
        "sketch_text_to_photo": sketch_text_to_photo,
    }


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
