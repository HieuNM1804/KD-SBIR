"""Losses used by the DFN5B-to-CLIP SBIR benchmark."""

import torch
from torch.nn import functional as F
from src.teacher_ensemble import TEACHERS, teacher_keys, is_ensemble, normalized_weights


def ensemble_teacher_log_probs(images, candidates, args, temperature):
    """Mix teacher probabilities, never cosine features or raw logits.

    Packed cache columns are split into each teacher's own embedding space.
    logsumexp retains very small probabilities for the symmetric KL objective.
    """
    keys = teacher_keys(args)
    dims = [TEACHERS[key]["output_dim"] for key in keys]
    weights = normalized_weights(getattr(args, "teacher_ensemble_weights", None), len(keys))
    if images.ndim != 2 or candidates.ndim != 2 or images.shape[-1] != sum(dims) or candidates.shape[-1] != sum(dims):
        raise ValueError("Packed teacher features have incorrect ensemble dimensions.")
    with torch.no_grad():
        image_parts = images.detach().float().split(dims, dim=-1)
        candidate_parts = candidates.detach().to(images.device, dtype=torch.float32).split(dims, dim=-1)
        terms = []
        for weight, image, candidate in zip(weights, image_parts, candidate_parts):
            if weight > 0:
                logits = F.normalize(image, dim=-1) @ F.normalize(candidate, dim=-1).t()
                log_probs = F.log_softmax(logits / temperature, dim=-1)
                terms.append(log_probs + log_probs.new_tensor(weight).log())
        return torch.logsumexp(torch.stack(terms), dim=0)


def ensemble_relational_kd_loss(student_sketch, student_photo, teacher_sketch,
                                teacher_photo, temperature, args):
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(student_photo.float(), dim=-1)
    student_log_probs = F.log_softmax(student_sketch @ student_photo.t() / temperature, dim=-1)
    teacher_log_probs = ensemble_teacher_log_probs(
        teacher_sketch.to(student_sketch.device), teacher_photo, args, temperature,
    )
    return F.kl_div(student_log_probs, teacher_log_probs.exp(), reduction="batchmean")


def ensemble_image_text_kd_loss(student_image, student_text, teacher_image,
                               teacher_text, temperature, args):
    student_image = F.normalize(student_image.float(), dim=-1)
    student_text = F.normalize(student_text.float(), dim=-1)
    student_log_probs = F.log_softmax(student_image @ student_text.t() / temperature, dim=-1)
    teacher_log_probs = ensemble_teacher_log_probs(
        teacher_image.to(student_image.device), teacher_text, args, temperature,
    )
    forward = F.kl_div(student_log_probs, teacher_log_probs.exp(), reduction="batchmean")
    reverse = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(-1).mean()
    return 0.5 * (forward + reverse)


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
        domain_loss = (ensemble_relational_kd_loss if is_ensemble(args) else relational_kd_loss)(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
            **({"args": args} if is_ensemble(args) else {}),
        )

    photo_text_kd = zero
    sketch_text_kd = zero
    if teacher_active and args.lambda_modality > 0:
        photo_text_kd = (ensemble_image_text_kd_loss if is_ensemble(args) else image_text_kd_loss)(
            photo_features,
            student_photo_text,
            teacher_photo_features,
            teacher_photo_text,
            args.photo_text_kd_temperature,
            **({"args": args} if is_ensemble(args) else {}),
        )
        sketch_text_kd = (ensemble_image_text_kd_loss if is_ensemble(args) else image_text_kd_loss)(
            sketch_features,
            student_sketch_text,
            teacher_sketch_features,
            teacher_sketch_text,
            args.sketch_text_kd_temperature,
            **({"args": args} if is_ensemble(args) else {}),
        )
    modality_loss = photo_text_kd + sketch_text_kd

    total_loss = (
        args.lambda_domain * domain_loss
        + args.lambda_modality * modality_loss
    )
    return total_loss, {
        "domain_kd": domain_loss,
        "modality_kd": modality_loss,
        "photo_text_kd": photo_text_kd,
        "sketch_text_kd": sketch_text_kd,
    }
