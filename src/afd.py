"""Dual-axis augmented feature distillation for sketch-photo retrieval."""

import torch
from torch import nn
from torch.nn import functional as F


AFD_CONTROLS = (
    "verified",
    "shuffled_image",
    "shuffled_text",
    "shuffled_both",
    "student_only",
    "teacher_only",
)


class AugmentedFeatureFusion(nn.Module):
    """Fuse normalized student and detached teacher features into student space."""

    def __init__(self, student_dim, teacher_dim, initialization="student_identity"):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.projection = nn.Linear(student_dim + teacher_dim, student_dim)
        self.reset_parameters(initialization)

    def reset_parameters(self, initialization):
        if initialization == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            return
        if initialization != "student_identity":
            raise ValueError(f"Unsupported AFD initialization: {initialization}")
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:, : self.student_dim].copy_(
                torch.eye(self.student_dim)
            )
            teacher_block = self.projection.weight[:, self.student_dim :]
            nn.init.xavier_uniform_(teacher_block)
            teacher_block.mul_(0.01)
            self.projection.bias.zero_()

    def forward(self, student, teacher):
        student = F.normalize(student.float(), dim=-1)
        teacher = F.normalize(
            teacher.detach().to(student.device, dtype=torch.float32), dim=-1
        )
        fused = self.projection(torch.cat((student, teacher), dim=-1))
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def branch_norms(self):
        weight = self.projection.weight
        return (
            weight[:, : self.student_dim].norm(),
            weight[:, self.student_dim :].norm(),
        )


def controlled_afd_inputs(student, teacher, control, kind):
    """Apply a deterministic control without changing marginal feature values."""
    if control not in AFD_CONTROLS:
        raise ValueError(f"Unsupported AFD control: {control}")
    if kind not in {"image", "text"}:
        raise ValueError(f"Unsupported AFD input kind: {kind}")
    teacher = teacher.detach()
    if control in {"shuffled_both", f"shuffled_{kind}"}:
        teacher = teacher.roll(1, dims=0)
    if control == "student_only":
        teacher = torch.zeros_like(teacher)
    if control == "teacher_only":
        student = torch.zeros_like(student)
    return student, teacher


def multi_positive_contrastive_loss(
    anchors,
    candidates,
    anchor_labels,
    candidate_labels,
    temperature,
):
    """InfoNCE with every same-class candidate treated as a positive."""
    if temperature <= 0:
        raise ValueError("AFD temperature must be positive.")
    anchor_labels = anchor_labels.to(anchors.device)
    candidate_labels = candidate_labels.to(anchors.device)
    logits = F.normalize(anchors.float(), dim=-1) @ F.normalize(
        candidates.float(), dim=-1
    ).T
    logits = logits / temperature
    positives = anchor_labels[:, None].eq(candidate_labels[None, :])
    if not positives.any(dim=1).all():
        raise ValueError("Every AFD anchor must have at least one positive.")
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def image_class_text_contrastive_loss(images, class_text, labels, temperature):
    """Symmetric image/class-text contrast with multi-positive reverse direction."""
    if temperature <= 0:
        raise ValueError("AFD temperature must be positive.")
    labels = labels.to(images.device).long()
    images = F.normalize(images.float(), dim=-1)
    class_text = F.normalize(class_text.float(), dim=-1)
    image_to_text = F.cross_entropy(images @ class_text.T / temperature, labels)
    present = torch.unique(labels, sorted=True)
    text_to_image = multi_positive_contrastive_loss(
        class_text[present], images, present, labels, temperature
    )
    return 0.5 * (image_to_text + text_to_image), {
        "image_to_text": image_to_text.detach(),
        "text_to_image": text_to_image.detach(),
    }


def dual_axis_afd_loss(
    augmented_sketch,
    augmented_photo,
    augmented_sketch_text,
    augmented_photo_text,
    labels,
    *,
    sketch_photo_weight,
    image_text_weight,
    sketch_photo_temperature,
    image_text_temperature,
):
    """AFD-only objective; no main domain or modality KD is included."""
    zero = augmented_sketch.new_zeros(())
    sketch_photo = zero
    sketch_to_photo = zero
    photo_to_sketch = zero
    if sketch_photo_weight > 0:
        sketch_to_photo = multi_positive_contrastive_loss(
            augmented_sketch,
            augmented_photo,
            labels,
            labels,
            sketch_photo_temperature,
        )
        photo_to_sketch = multi_positive_contrastive_loss(
            augmented_photo,
            augmented_sketch,
            labels,
            labels,
            sketch_photo_temperature,
        )
        sketch_photo = 0.5 * (sketch_to_photo + photo_to_sketch)

    sketch_text = zero
    photo_text = zero
    sketch_text_parts = {"image_to_text": zero, "text_to_image": zero}
    photo_text_parts = {"image_to_text": zero, "text_to_image": zero}
    if image_text_weight > 0:
        if augmented_sketch_text is None or augmented_photo_text is None:
            raise ValueError("Image-text AFD requires both augmented text banks.")
        sketch_text, sketch_text_parts = image_class_text_contrastive_loss(
            augmented_sketch,
            augmented_sketch_text,
            labels,
            image_text_temperature,
        )
        photo_text, photo_text_parts = image_class_text_contrastive_loss(
            augmented_photo,
            augmented_photo_text,
            labels,
            image_text_temperature,
        )
    image_text = 0.5 * (sketch_text + photo_text)
    weighted_sp = sketch_photo_weight * sketch_photo
    weighted_it = image_text_weight * image_text
    total = weighted_sp + weighted_it
    return total, {
        "afd_sp": sketch_photo,
        "afd_sp_weighted": weighted_sp,
        "afd_sketch_to_photo": sketch_to_photo,
        "afd_photo_to_sketch": photo_to_sketch,
        "afd_it": image_text,
        "afd_it_weighted": weighted_it,
        "afd_sketch_text": sketch_text,
        "afd_photo_text": photo_text,
        "afd_sketch_image_to_text": sketch_text_parts["image_to_text"],
        "afd_sketch_text_to_image": sketch_text_parts["text_to_image"],
        "afd_photo_image_to_text": photo_text_parts["image_to_text"],
        "afd_photo_text_to_image": photo_text_parts["text_to_image"],
    }
