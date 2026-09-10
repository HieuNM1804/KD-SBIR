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


@torch.no_grad()
def fine_grained_prompt_retrieval_accuracy(
    sketch_image_features,
    photo_image_features,
    sketch_prompt_text_features,
    photo_prompt_text_features,
    target_photo_indices,
):
    """Acc@1/Acc@5 for both exact-instance prompt retrieval directions."""
    sketch_images = F.normalize(sketch_image_features.float(), dim=-1)
    photo_images = F.normalize(photo_image_features.float(), dim=-1)
    sketch_text = F.normalize(sketch_prompt_text_features.float(), dim=-1)
    photo_text = F.normalize(photo_prompt_text_features.float(), dim=-1)
    targets = target_photo_indices.to(sketch_images.device).long().reshape(-1)
    if photo_images.shape[0] != 100 or photo_text.shape[0] != 100:
        raise RuntimeError("Prompt retrieval metrics require 100 photos.")
    if len(targets) != len(sketch_images) or len(targets) != len(sketch_text):
        raise ValueError("Every sketch query needs one exact photo target.")

    def accuracy(logits):
        ranking = logits.topk(k=5, dim=-1, largest=True, sorted=True).indices
        matches = ranking.eq(targets[:, None])
        return matches[:, 0].float().mean(), matches.any(dim=-1).float().mean()

    image_to_text = accuracy(sketch_images @ photo_text.t())
    text_to_image = accuracy(sketch_text @ photo_images.t())
    return {
        "sketch_to_photo_text_acc1": image_to_text[0],
        "sketch_to_photo_text_acc5": image_to_text[1],
        "sketch_text_to_photo_acc1": text_to_image[0],
        "sketch_text_to_photo_acc5": text_to_image[1],
    }


def image_conditioned_text_anchor_loss(
    sketch_text_features,
    photo_text_features,
    sketch_class_anchors,
    photo_class_anchors,
):
    """Keep dynamic prompts close to frozen CLIP class semantics."""
    pairs = (
        (sketch_text_features, sketch_class_anchors),
        (photo_text_features, photo_class_anchors),
    )
    losses = []
    for conditioned, anchor in pairs:
        if conditioned.shape != anchor.shape:
            raise ValueError(
                "Conditioned text features and class anchors must match."
            )
        conditioned = F.normalize(conditioned.float(), dim=-1)
        anchor = F.normalize(anchor.detach().float(), dim=-1)
        losses.append(1.0 - (conditioned * anchor).sum(dim=-1).mean())
    return 0.5 * (losses[0] + losses[1])


def teacher_visual_preservation_loss(
    sketch_features,
    photo_features,
    source_sketch_features,
    source_photo_features,
):
    """Cosine preservation against the frozen Phase-A visual teacher."""
    current_sketch = F.normalize(sketch_features.float(), dim=-1)
    current_photo = F.normalize(photo_features.float(), dim=-1)
    source_sketch = F.normalize(
        source_sketch_features.detach().float(), dim=-1
    )
    source_photo = F.normalize(
        source_photo_features.detach().float(), dim=-1
    )
    return 0.5 * (
        (1.0 - (current_sketch * source_sketch).sum(dim=-1)).mean()
        + (1.0 - (current_photo * source_photo).sum(dim=-1)).mean()
    )


def teacher_visual_refinement_control_loss(
    sketch_features,
    photo_features,
    source_sketch_features,
    source_photo_features,
    target_photo_indices,
    visual_temperature,
    lambda_retrieval,
    lambda_keep,
):
    """Matched Phase-C control with no information from the text branch."""
    retrieval = fine_grained_teacher_infonce_loss(
        sketch_features,
        photo_features,
        target_photo_indices,
        visual_temperature,
    )
    keep = teacher_visual_preservation_loss(
        sketch_features,
        photo_features,
        source_sketch_features,
        source_photo_features,
    )
    total = lambda_retrieval * retrieval + lambda_keep * keep
    return total, {"retrieval": retrieval, "keep": keep}


def teacher_semantic_refinement_loss(
    sketch_features,
    photo_features,
    source_sketch_features,
    source_photo_features,
    fixed_sketch_text_features,
    fixed_photo_text_features,
    target_photo_indices,
    visual_temperature,
    semantic_temperature,
    lambda_retrieval,
    lambda_semantic,
    lambda_keep,
):
    """Refine teacher visual prompts against frozen semantic targets.

    Text and source-visual tensors are detached here by construction. This
    makes the optimization direction explicit: semantic targets supervise
    current visual prompts, while the text prompt learner cannot move.
    """
    retrieval = fine_grained_teacher_infonce_loss(
        sketch_features,
        photo_features,
        target_photo_indices,
        visual_temperature,
    )
    semantic, semantic_parts = fine_grained_prompt_infonce_loss(
        sketch_features,
        photo_features,
        fixed_sketch_text_features.detach(),
        fixed_photo_text_features.detach(),
        target_photo_indices,
        semantic_temperature,
    )
    keep = teacher_visual_preservation_loss(
        sketch_features,
        photo_features,
        source_sketch_features,
        source_photo_features,
    )
    total = (
        lambda_retrieval * retrieval
        + lambda_semantic * semantic
        + lambda_keep * keep
    )
    return total, {
        "retrieval": retrieval,
        "semantic": semantic,
        "keep": keep,
        **semantic_parts,
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
