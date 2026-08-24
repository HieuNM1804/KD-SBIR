import torch
from torch.nn import functional as F

from src.losses import image_text_kd_loss, relational_kd_loss


def fine_grained_teacher_triplet_loss(
    sketch_features,
    photo_features,
    category_ids,
    instance_ids,
    margin=0.2,
):
    """Exact-pair triplet loss with hard same-category instance negatives."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    category_ids = category_ids.to(sketch_features.device)
    instance_ids = instance_ids.to(sketch_features.device)

    distance = 1.0 - sketch_features @ photo_features.t()
    positive = distance.diagonal()
    same_category = category_ids[:, None].eq(category_ids[None, :])
    different_instance = instance_ids[:, None].ne(instance_ids[None, :])
    negative_mask = same_category & different_instance
    valid = negative_mask.any(dim=-1)
    if not valid.all():
        raise RuntimeError(
            "Every fine-grained sample needs a different-instance negative "
            "from the same category."
        )

    def one_direction(current_distance):
        hardest_negative = current_distance.masked_fill(
            ~negative_mask, torch.inf
        ).min(dim=-1).values
        return F.relu(positive - hardest_negative + margin).mean()

    return 0.5 * (
        one_direction(distance) + one_direction(distance.t())
    )


def category_relational_kd_loss(
    student_sketch,
    student_photo,
    teacher_sketch,
    teacher_photo,
    category_ids,
    temperature=0.07,
):
    """Apply relational KD independently inside every category block."""
    weighted_loss = student_sketch.new_zeros((), dtype=torch.float32)
    sample_count = 0
    for category in torch.unique(category_ids, sorted=True):
        mask = category_ids.eq(category)
        count = int(mask.sum().item())
        if count < 2:
            raise RuntimeError(
                "Fine-grained relational KD needs at least two instances per category."
            )
        current = relational_kd_loss(
            student_sketch[mask],
            student_photo[mask],
            teacher_sketch[mask],
            teacher_photo[mask],
            temperature,
        )
        weighted_loss = weighted_loss + current * count
        sample_count += count
    return weighted_loss / sample_count


def fine_grained_distillation_loss(args, features, category_ids):
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
        domain_loss = category_relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            category_ids.to(photo_features.device),
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
