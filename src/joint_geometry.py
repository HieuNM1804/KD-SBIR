"""Match final-embedding geometry across and within the two SBIR modalities.

No feature projection, patch alignment, classifier, or learned graph module.
The fixed teacher supplies all cosine targets, including negative similarities.
"""

import math

import torch
from torch.nn import functional as F


def _normalized_features(student_sketch, student_photo, teacher_sketch, teacher_photo):
    tensors = (student_sketch, student_photo, teacher_sketch, teacher_photo)
    if any(t.ndim != 2 or t.shape[1] == 0 for t in tensors):
        raise ValueError("Geometry features must be nonempty [batch, width] tensors.")
    batch = student_sketch.shape[0]
    if batch < 2 or any(t.shape[0] != batch for t in tensors):
        raise ValueError("Joint geometry requires equal sketch/photo batches of at least two.")
    if student_sketch.shape[1] != student_photo.shape[1]:
        raise ValueError("Student modalities must share their embedding width.")
    if teacher_sketch.shape[1] != teacher_photo.shape[1]:
        raise ValueError("Teacher modalities must share their embedding width.")
    target_device = student_sketch.device
    if student_photo.device != target_device:
        raise ValueError("Student sketch and photo must be on the same device.")
    ss = F.normalize(student_sketch.float(), dim=-1)
    sp = F.normalize(student_photo.float(), dim=-1)
    with torch.no_grad():
        ts = F.normalize(teacher_sketch.to(device=target_device, dtype=torch.float32), dim=-1)
        tp = F.normalize(teacher_photo.to(device=target_device, dtype=torch.float32), dim=-1)
    return ss, sp, ts, tp


def joint_geometry_loss(student_sketch, student_photo, teacher_sketch,
                        teacher_photo, cross_weight=0.5):
    """Block-balanced MSE of the signed cosine Gram matrix of [sketch; photo].

    L = alpha * mean(SP error^2)
        + (1-alpha)/2 * (mean(SS off-diagonal error^2)
                         + mean(PP off-diagonal error^2)).

    PS is SP.T, so counting it again adds no information. SS/PP diagonal
    self-similarities are excluded; the SP diagonal IS retained because main
    samples a same-class photo for each sketch, not an identical input.
    Blocks are averaged separately to keep weighting independent of batch size.
    """
    if not math.isfinite(cross_weight) or not 0 <= cross_weight <= 1:
        raise ValueError("joint_cross_weight must be finite and in [0, 1].")
    with torch.autocast(device_type=student_sketch.device.type, enabled=False):
        ss, sp, ts, tp = _normalized_features(
            student_sketch, student_photo, teacher_sketch, teacher_photo
        )
        mask = ~torch.eye(len(ss), dtype=torch.bool, device=ss.device)
        with torch.no_grad():
            target_sp = ts @ tp.t()
            target_ss = ts @ ts.t()
            target_pp = tp @ tp.t()
        cross = (ss @ sp.t() - target_sp).square().mean()
        sketch = (ss @ ss.t() - target_ss)[mask].square().mean()
        photo = (sp @ sp.t() - target_pp)[mask].square().mean()
        total = cross_weight * cross + (1 - cross_weight) * (sketch + photo) / 2
    return total, {"joint_sp": cross, "joint_ss": sketch, "joint_pp": photo}


@torch.no_grad()
def geometry_diagnostics(student_sketch, student_photo, teacher_sketch,
                         teacher_photo, labels):
    """Descriptive batch diagnostics, not losses or estimates over the full set.

    Label masks are used ONLY to report SP same/different-class similarities.
    No presumed instance pairing, pseudo-labels, or class targets in the loss.
    Zero pair counts distinguish unavailable statistics from cosine zero.
    """
    with torch.autocast(device_type=student_sketch.device.type, enabled=False):
        ss, sp, ts, tp = _normalized_features(
            student_sketch, student_photo, teacher_sketch, teacher_photo
        )
        labels = labels.to(ss.device)
        if labels.ndim != 1 or len(labels) != len(ss):
            raise ValueError("Expected one class label per paired batch row.")
        same = labels[:, None].eq(labels[None, :])
        masks = {"same_class": same, "different_class": ~same}
        stats = {f"{name}_count": mask.sum().float() for name, mask in masks.items()}
        for prefix, sketch, photo in (("student", ss, sp), ("teacher", ts, tp)):
            relations = sketch @ photo.t()
            for name, mask in masks.items():
                stats[f"{prefix}_sp_{name}"] = relations[mask].sum() / mask.sum().clamp_min(1)
            stats[f"{prefix}_sketch_variance"] = sketch.var(dim=0, unbiased=False).sum()
            stats[f"{prefix}_photo_variance"] = photo.var(dim=0, unbiased=False).sum()
            stats[f"{prefix}_centroid_gap"] = (sketch.mean(0) - photo.mean(0)).norm()
            stats[f"{prefix}_sp_std"] = relations.std(unbiased=False)
        return stats
