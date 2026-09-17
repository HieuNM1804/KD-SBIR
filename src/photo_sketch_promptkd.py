"""PromptKD-style distillation through photo-sketch retrieval prototypes."""

import torch
from torch.nn import functional as F


def select_central_anchor_indices(
    features,
    labels,
    class_count,
    anchors_per_class,
):
    """Select deterministic real-image anchors nearest each class centroid."""
    if anchors_per_class < 1:
        raise ValueError("anchors_per_class must be at least 1.")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length.")

    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    selected = []
    for class_index in range(class_count):
        class_indices = torch.where(labels == class_index)[0]
        if len(class_indices) < anchors_per_class:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} samples; "
                f"{anchors_per_class} anchors were requested."
            )
        class_features = F.normalize(
            features[class_indices].float(),
            dim=-1,
        )
        centroid = F.normalize(class_features.mean(dim=0), dim=0)
        similarity = class_features @ centroid
        ranking = sorted(
            range(len(class_indices)),
            key=lambda index: (-similarity[index].item(), index),
        )
        selected.append(class_indices[ranking[:anchors_per_class]].clone())
    return torch.stack(selected)


def build_cross_modal_prototypes(
    sketch_features,
    photo_features,
    sketch_anchor_indices,
    photo_anchor_indices,
):
    """Average matched photo/sketch anchor groups into one vector per class."""
    if sketch_anchor_indices.shape != photo_anchor_indices.shape:
        raise ValueError("Sketch and photo anchor index grids must match.")
    if sketch_anchor_indices.ndim != 2:
        raise ValueError("Anchor indices must have shape [classes, anchors].")

    sketch = F.normalize(
        sketch_features[sketch_anchor_indices].float(),
        dim=-1,
    ).mean(dim=1)
    photo = F.normalize(
        photo_features[photo_anchor_indices].float(),
        dim=-1,
    ).mean(dim=1)
    sketch = F.normalize(sketch, dim=-1)
    photo = F.normalize(photo, dim=-1)
    return F.normalize(sketch + photo, dim=-1)


def prototype_kd_loss(
    student_features,
    teacher_features,
    student_prototypes,
    teacher_prototypes,
    student_temperature,
    teacher_temperature,
):
    """Match teacher and student rankings over shared prototype identities."""
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("Prototype temperatures must be greater than 0.")
    if len(student_prototypes) != len(teacher_prototypes):
        raise ValueError("Teacher and student prototype counts must match.")

    student_features = F.normalize(student_features.float(), dim=-1)
    student_prototypes = F.normalize(student_prototypes.float(), dim=-1)
    student_log_probs = F.log_softmax(
        student_features @ student_prototypes.t() / student_temperature,
        dim=-1,
    )

    with torch.no_grad():
        teacher_features = F.normalize(
            teacher_features.to(student_features.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_prototypes = F.normalize(
            teacher_prototypes.to(student_features.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_probs = F.softmax(
            teacher_features @ teacher_prototypes.t() / teacher_temperature,
            dim=-1,
        )

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
