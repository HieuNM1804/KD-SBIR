"""Attention-verified counterfactual retrieval-field distillation.

The teacher attention output proposes ink regions. A teacher global-feature
counterfactual verifies which proposal actually changes sketch-to-photo geometry.
Training matches the clean geometry and the intervention-induced geometry change.
No class labels, ranks, margins, or inference-time masks are used by this loss.
"""

import math
import torch
from torch.nn import functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CACHE_FORMAT_VERSION = 1


def _image_constants(image):
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    return mean, std


def ink_alpha(image, threshold=0.08, softness=0.12):
    """Return a soft raster-ink mask from a CLIP-normalized RGB sketch."""
    if image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError("Expected [3,H,W] or [B,3,H,W] normalized images")
    mean, std = _image_constants(image)
    if image.ndim == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    rgb = (image * std + mean).clamp(0, 1)
    gray = 0.2989 * rgb[..., 0, :, :] + 0.5870 * rgb[..., 1, :, :] + 0.1140 * rgb[..., 2, :, :]
    return ((1.0 - gray - threshold) / max(float(softness), 1e-6)).clamp(0, 1)


def erase_ink_box(image, box, threshold=0.08, softness=0.12):
    """Whiten only raster ink inside ``box=(top,left,bottom,right)``."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one normalized RGB image [3,H,W]")
    top, left, bottom, right = [int(x) for x in box]
    height, width = image.shape[-2:]
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError(f"Invalid counterfactual box {tuple(box)} for {(height, width)}")
    alpha = torch.zeros((height, width), dtype=image.dtype, device=image.device)
    local_ink = ink_alpha(image, threshold, softness)
    alpha[top:bottom, left:right] = local_ink[top:bottom, left:right]
    white = (image.new_ones(3, 1, 1) - image.new_tensor(CLIP_MEAN).view(3, 1, 1)) / image.new_tensor(CLIP_STD).view(3, 1, 1)
    return image * (1.0 - alpha.unsqueeze(0)) + white * alpha.unsqueeze(0)


def patch_ink_mass(image, grid, threshold=0.08, softness=0.12):
    """Pool raster-ink mass onto a square teacher patch grid."""
    alpha = ink_alpha(image, threshold, softness)
    if alpha.ndim == 2:
        alpha = alpha[None, None]
    else:
        alpha = alpha[:, None]
    pooled = F.adaptive_avg_pool2d(alpha.float(), (grid, grid))
    return pooled[:, 0] if image.ndim == 4 else pooled[0, 0]


def _window_sums(values, window):
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Expected a square patch map")
    if not 1 <= window <= values.shape[0]:
        raise ValueError("Counterfactual window must fit the patch grid")
    kernel = values.new_ones(1, 1, window, window)
    return F.conv2d(values[None, None], kernel)[0, 0]


def _grid_box(row, col, window, grid, image_size):
    top = round(row * image_size / grid)
    left = round(col * image_size / grid)
    bottom = round((row + window) * image_size / grid)
    right = round((col + window) * image_size / grid)
    return (top, left, bottom, right)


def _overlap_fraction(a, b):
    at, al, ab, ar = a
    bt, bl, bb, br = b
    intersection = max(0, min(ab, bb) - max(at, bt)) * max(0, min(ar, br) - max(al, bl))
    area = max(1, (ab - at) * (ar - al))
    return intersection / area


def attention_proposals(attention_map, ink_map, window, count, image_size, max_overlap=0.25):
    """Select deterministic, non-overlapping ink-aware attention proposals.

    Returns pixel boxes plus proposal score and pooled ink mass. Attention is a
    proposal prior only; the cache builder later verifies every returned box by
    re-encoding its counterfactual with the teacher.
    """
    if attention_map.shape != ink_map.shape or attention_map.ndim != 2:
        raise ValueError("Attention and ink maps must have the same 2D shape")
    grid = attention_map.shape[0]
    if count < 1:
        raise ValueError("At least one proposal is required")
    attention = attention_map.float().clamp_min(0)
    ink = ink_map.float().clamp_min(0)
    attention = attention / attention.sum().clamp_min(1e-12)
    # Multiplying by sqrt(ink) suppresses blank high-attention patches without
    # letting a thick stroke dominate the semantic attention signal entirely.
    joint = attention * ink.clamp_min(0).sqrt()
    score = _window_sums(joint, window)
    mass = _window_sums(ink, window)
    flat_order = torch.argsort(score.flatten(), descending=True, stable=True)
    proposals = []
    for flat in flat_order.tolist():
        row, col = divmod(flat, score.shape[1])
        grid_box = (row, col, row + window, col + window)
        if any(_overlap_fraction(grid_box, item[3]) > max_overlap for item in proposals):
            continue
        proposals.append((
            _grid_box(row, col, window, grid, image_size),
            float(score[row, col]),
            float(mass[row, col]),
            grid_box,
        ))
        if len(proposals) == count:
            break
    if not proposals:
        raise RuntimeError("No attention proposal was produced")
    return [(box, score_value, mass_value) for box, score_value, mass_value, _ in proposals]


def similarity_field(query, gallery):
    if query.ndim != 2 or gallery.ndim != 2:
        raise ValueError("Retrieval fields require 2D feature tensors")
    return F.normalize(query.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T


def counterfactual_effect(clean_query, masked_query, gallery):
    """Return clean-minus-counterfactual similarity fields and row RMS."""
    clean = similarity_field(clean_query, gallery)
    masked = similarity_field(masked_query, gallery)
    effect = clean - masked
    centered = effect - effect.mean(dim=-1, keepdim=True)
    rms = centered.square().mean(dim=-1).sqrt()
    return effect, rms


def _field_alignment(student_field, teacher_field, magnitude_weight, eps):
    student = student_field.float()
    teacher = teacher_field.detach().to(student.device, dtype=torch.float32)
    student_centered = student - student.mean(dim=-1, keepdim=True)
    teacher_centered = teacher - teacher.mean(dim=-1, keepdim=True)
    student_rms = (student_centered.square().mean(dim=-1) + eps**2).sqrt() - eps
    teacher_rms = teacher_centered.square().mean(dim=-1).sqrt()

    teacher_energy = teacher_rms.detach()
    mean_energy = teacher_energy.mean().clamp_min(eps)
    valid = teacher_energy > eps
    weights = (teacher_energy / mean_energy).clamp(0.1, 4.0) * valid
    weights = weights / weights.mean().clamp_min(eps)
    direction = 1.0 - F.cosine_similarity(
        student_centered, teacher_centered, dim=-1, eps=eps
    )
    scale = teacher_rms.detach().mean().clamp_min(eps)
    magnitude = F.smooth_l1_loss(
        student_rms / scale,
        teacher_rms / scale,
        reduction="none",
        beta=0.5,
    )
    loss = (weights * (direction + magnitude_weight * magnitude)).mean()
    return loss, {
        "cosine": (1.0 - direction).mean().detach(),
        "student_rms": student_rms.mean().detach(),
        "teacher_rms": teacher_rms.mean().detach(),
        "magnitude_ratio": (student_rms.mean() / teacher_rms.mean().clamp_min(eps)).detach(),
    }


def avcrd_loss(
    student_sketch,
    student_masked_sketch,
    student_photo,
    teacher_sketch,
    teacher_masked_sketch,
    teacher_photo,
    clean_weight=1.0,
    effect_weight=1.0,
    magnitude_weight=0.25,
    eps=1e-6,
    shuffle_effect=False,
):
    """Match clean and intervention-induced cross-modal retrieval fields.

    The clean term makes the unperturbed geometry identifiable. The effect term
    transfers the teacher's retrieval response to removing its verified sketch
    evidence. Both terms operate on continuous cosine fields without labels or
    ranking operators.
    """
    for name, value in {
        "clean_weight": clean_weight,
        "effect_weight": effect_weight,
        "magnitude_weight": magnitude_weight,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if clean_weight == 0 and effect_weight == 0:
        raise ValueError("At least one AVCRD field must be active")
    tensors = (
        student_sketch, student_masked_sketch, student_photo,
        teacher_sketch, teacher_masked_sketch, teacher_photo,
    )
    if any(x.ndim != 2 for x in tensors):
        raise ValueError("AVCRD features must be 2D")
    if student_sketch.shape[0] != teacher_sketch.shape[0] or student_photo.shape[0] != teacher_photo.shape[0]:
        raise ValueError("Teacher/student batch rows must align")
    if student_masked_sketch.shape[0] != student_sketch.shape[0] or teacher_masked_sketch.shape[0] != teacher_sketch.shape[0]:
        raise ValueError("Clean and counterfactual sketch batches must align")

    if student_photo.shape[0] < 2 or student_sketch.shape[0] < 2:
        raise ValueError("AVCRD requires at least two queries and photos")
    with torch.autocast(device_type=student_sketch.device.type, enabled=False):
        student_clean = similarity_field(student_sketch, student_photo)
        student_masked = similarity_field(student_masked_sketch, student_photo)
        with torch.no_grad():
            teacher_clean = similarity_field(
                teacher_sketch.detach().to(student_sketch.device),
                teacher_photo.detach().to(student_sketch.device),
            )
            teacher_masked = similarity_field(
                teacher_masked_sketch.detach().to(student_sketch.device),
                teacher_photo.detach().to(student_sketch.device),
            )
        clean_loss, clean_stats = _field_alignment(
            student_clean, teacher_clean, magnitude_weight, eps
        )
        teacher_effect = teacher_clean - teacher_masked
        if shuffle_effect:
            # Deterministic cyclic swap preserves the effect distribution while
            # breaking its association with the corresponding sketch.
            teacher_effect = teacher_effect.roll(1, dims=0)
        effect_loss, effect_stats = _field_alignment(
            student_clean - student_masked,
            teacher_effect,
            magnitude_weight,
            eps,
        )
        total = clean_weight * clean_loss + effect_weight * effect_loss
    return total, {
        "clean_loss": clean_loss.detach(),
        "effect_loss": effect_loss.detach(),
        "clean_cosine": clean_stats["cosine"],
        "effect_cosine": effect_stats["cosine"],
        "clean_student_rms": clean_stats["student_rms"],
        "clean_teacher_rms": clean_stats["teacher_rms"],
        "effect_student_rms": effect_stats["student_rms"],
        "effect_teacher_rms": effect_stats["teacher_rms"],
        "effect_magnitude_ratio": effect_stats["magnitude_ratio"],
    }


def erase_ink_batch(images, boxes, threshold=0.08, softness=0.12):
    """Vectorized ink erasure: [B,3,H,W] images and [B,4] pixel boxes."""
    if images.ndim != 4 or boxes.shape != (len(images), 4):
        raise ValueError("Expected aligned [B,3,H,W] images and [B,4] boxes")
    height, width = images.shape[-2:]
    boxes = boxes.to(images.device)
    if not ((boxes[:, :2] >= 0).all() and (boxes[:, 2] <= height).all()
            and (boxes[:, 3] <= width).all() and (boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("Invalid batched counterfactual boxes")
    rows = torch.arange(height, device=images.device)[None, :, None]
    cols = torch.arange(width, device=images.device)[None, None, :]
    region = ((rows >= boxes[:, 0, None, None]) & (rows < boxes[:, 2, None, None])
              & (cols >= boxes[:, 1, None, None]) & (cols < boxes[:, 3, None, None]))
    alpha = ink_alpha(images.float(), threshold, softness) * region
    white = (images.new_ones(1, 3, 1, 1) - images.new_tensor(CLIP_MEAN).view(1,3,1,1)) / images.new_tensor(CLIP_STD).view(1,3,1,1)
    return (images.float() * (1 - alpha[:,None]) + white.float() * alpha[:,None]).to(images.dtype)


def random_mass_matched_proposals(ink_map, proposals, window, image_size, generator):
    """Equal-area random windows, matched approximately by raster ink mass.

    For each attention proposal, sample among up to eight windows whose mass is
    within 20% of the proposal's mass. If none qualifies, sample among the eight
    closest windows. Actual mass error is exported; exact mass equality is not
    assumed. A random shortlist receives the same teacher verification budget.
    """
    grid = ink_map.shape[0]
    mass = _window_sums(ink_map.float(), window)
    used = set()
    used_boxes = []
    result = []
    for salient_box, _, target_mass in proposals:
        candidates=[]
        for row in range(mass.shape[0]):
            for col in range(mass.shape[1]):
                box=_grid_box(row,col,window,grid,image_size)
                if ((row,col) in used or _overlap_fraction(box,salient_box) > 0.25
                        or any(_overlap_fraction(box,other)>0.25 for other in used_boxes)):
                    continue
                error=abs(float(mass[row,col])-target_mass)/max(target_mass,1e-6)
                candidates.append((error,row,col,box))
        if not candidates:
            raise ValueError("Image grid cannot supply matched random proposals; use a smaller window")
        candidates.sort(key=lambda item:item[:3])
        eligible=[item for item in candidates if item[0]<=.2][:8] or candidates[:8]
        item=eligible[int(torch.randint(len(eligible),(),generator=generator))]
        used.add((item[1],item[2]))
        used_boxes.append(item[3])
        result.append((item[3],float(mass[item[1],item[2]]),item[0]))
    return result
