"""Stroke-graph extraction and causal evidence pooling for sketch-photo KD."""

from contextlib import AbstractContextManager
import math

import torch
from torch import nn
from torch.nn import functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TARGET_NAMES = ("verified", "local", "random")


def _as_batch_first(sequence, batch_first):
    return sequence if batch_first else sequence.transpose(0, 1)


class FinalBlockInputCapture(AbstractContextManager):
    """Capture the residual stream entering the final visual Transformer block."""

    def __init__(self, visual):
        self.visual = visual
        self.values = []
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Capture cannot be nested on itself")
        self.values.clear()

        def hook(_module, args):
            self.values.append(args[0])

        self.handle = self.visual.transformer.resblocks[-1].register_forward_pre_hook(hook)
        return self

    def __exit__(self, *_args):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return False

    def residual(self):
        if len(self.values) != 1:
            raise RuntimeError("Expected exactly one visual forward in the capture")
        return self.values[0]


def patch_residuals(visual, sequence):
    """Return final-block input patch tokens as [batch, patches, width]."""
    batch_first = getattr(visual.transformer, "batch_first", False)
    x = _as_batch_first(sequence, batch_first)
    patch_count = visual.positional_embedding.shape[0] - 1
    if x.shape[1] < patch_count + 1:
        raise ValueError("Visual sequence is shorter than CLS plus image patches")
    return x[:, 1 : 1 + patch_count]


def projected_patch_features(visual, sequence):
    """Read final-block input patches in the deployed visual projection space."""
    patches = patch_residuals(visual, sequence)
    patches = visual.ln_post(patches)
    if visual.proj is not None:
        patches = patches @ visual.proj
    return patches.float()


def cls_patch_attention(visual, sequence):
    """Mean-head final-block CLS-to-image-patch attention, excluding prompts."""
    block = visual.transformer.resblocks[-1]
    attn = block.attn
    if not isinstance(attn, nn.MultiheadAttention):
        raise TypeError("SGCD currently requires torch.nn.MultiheadAttention")
    if attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
        raise ValueError("Extra MultiheadAttention bias/zero tokens are unsupported")
    x = block.ln_1(sequence)
    batch_first = getattr(visual.transformer, "batch_first", False)
    x = _as_batch_first(x, batch_first)
    width, heads = attn.embed_dim, attn.num_heads
    head_width = width // heads
    if attn.in_proj_weight is None:
        wq, wk = attn.q_proj_weight, attn.k_proj_weight
    else:
        wq, wk, _wv = attn.in_proj_weight.chunk(3, dim=0)
    if attn.in_proj_bias is None:
        bq = bk = None
    else:
        bq, bk, _bv = attn.in_proj_bias.chunk(3)
    with torch.autocast(device_type=x.device.type, enabled=False):
        q = F.linear(x[:, :1].float(), wq.float(), None if bq is None else bq.float())
        k = F.linear(x.float(), wk.float(), None if bk is None else bk.float())
        q = q.reshape(len(x), 1, heads, head_width).transpose(1, 2)
        k = k.reshape(len(x), -1, heads, head_width).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(head_width)).softmax(-1)
    patch_count = visual.positional_embedding.shape[0] - 1
    return weights[:, :, 0, 1 : 1 + patch_count].mean(1)


def ink_strength(images, threshold=0.08, softness=0.12):
    """Soft dark-ink mass in [0,1] from CLIP-normalized images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected [batch,3,height,width] images")
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    gray = (images.float() * std + mean).clamp(0, 1).mean(1)
    return ((1.0 - gray - threshold) / softness).clamp(0, 1)


def patch_ink_mass(images, grid, threshold=0.08, softness=0.12):
    ink = ink_strength(images, threshold, softness)
    return F.adaptive_avg_pool2d(ink[:, None], (grid, grid))[:, 0].flatten(1)


def zhang_suen_thinning(binary, max_iterations=32):
    """Batch Zhang-Suen thinning for small raster sketch masks."""
    if binary.ndim != 3:
        raise ValueError("Expected [batch,height,width] binary masks")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    x = binary.bool()

    def neighbours(value):
        padded = F.pad(value.float(), (1, 1, 1, 1)).bool()
        return (
            padded[:, :-2, 1:-1],
            padded[:, :-2, 2:],
            padded[:, 1:-1, 2:],
            padded[:, 2:, 2:],
            padded[:, 2:, 1:-1],
            padded[:, 2:, :-2],
            padded[:, 1:-1, :-2],
            padded[:, :-2, :-2],
        )

    for _ in range(max_iterations):
        previous = x
        for substep in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbours(x)
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = sum((~ring[index] & ring[index + 1]).int()
                              for index in range(8))
            degree = sum(value.int() for value in ring[:-1])
            common = x & (degree >= 2) & (degree <= 6) & (transitions == 1)
            if substep == 0:
                removable = common & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                removable = common & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            x = x & ~removable
        if torch.equal(x, previous):
            break
    return x


def _trace_skeleton_paths(mask):
    """Trace ordered paths between skeleton endpoints/junctions for one image."""
    coordinates = {tuple(value) for value in torch.nonzero(mask, as_tuple=False).tolist()}
    if not coordinates:
        return []
    offsets = tuple((row, col) for row in (-1, 0, 1) for col in (-1, 0, 1)
                    if row or col)
    adjacency = {
        point: sorted((point[0] + dr, point[1] + dc) for dr, dc in offsets
                      if (point[0] + dr, point[1] + dc) in coordinates)
        for point in coordinates
    }
    keys = {point for point, neighbours in adjacency.items() if len(neighbours) != 2}
    visited = set()
    paths = []

    def edge(a, b):
        return tuple(sorted((a, b)))

    def walk(start, neighbour):
        result = [start]
        previous, current = start, neighbour
        visited.add(edge(previous, current))
        while True:
            result.append(current)
            if current in keys and current != start:
                break
            candidates = [value for value in adjacency[current]
                          if value != previous and edge(current, value) not in visited]
            if not candidates:
                break
            following = candidates[0]
            visited.add(edge(current, following))
            previous, current = current, following
        return result

    for start in sorted(keys):
        if not adjacency[start]:
            paths.append([start])
        for neighbour in adjacency[start]:
            if edge(start, neighbour) not in visited:
                paths.append(walk(start, neighbour))
    for start in sorted(coordinates):
        for neighbour in adjacency[start]:
            if edge(start, neighbour) not in visited:
                paths.append(walk(start, neighbour))
    return [path for path in paths if path]


def _split_paths(paths, maximum):
    paths = [list(path) for path in paths]
    while len(paths) < maximum:
        candidates = [(len(path), index) for index, path in enumerate(paths)
                      if len(path) >= 6]
        if not candidates:
            break
        _length, index = max(candidates)
        path = paths.pop(index)
        middle = len(path) // 2
        paths.extend((path[:middle + 1], path[middle:]))
    return sorted(paths, key=len, reverse=True)[:maximum]


def stroke_path_maps(images, output_grid, skeleton_grid=28, max_paths=6,
                     threshold=0.08, softness=0.12, binary_threshold=0.10):
    """Extract padded stroke-path maps [B,K,output_grid**2] from raster sketches."""
    if skeleton_grid < output_grid or max_paths < 1:
        raise ValueError("Invalid stroke graph geometry")
    ink_pixels = ink_strength(images, threshold, softness)
    coarse = F.adaptive_max_pool2d(ink_pixels[:, None],
                                   (skeleton_grid, skeleton_grid))[:, 0]
    skeleton = zhang_suen_thinning(coarse > binary_threshold)
    ink_patches = patch_ink_mass(images, output_grid, threshold, softness)
    maps = images.new_zeros(len(images), max_paths, output_grid * output_grid,
                            dtype=torch.float32)
    valid = torch.zeros(len(images), max_paths, dtype=torch.bool, device=images.device)
    for batch_index in range(len(images)):
        paths = _split_paths(_trace_skeleton_paths(skeleton[batch_index].cpu()), max_paths)
        signatures = set()
        kept = []
        for path in paths:
            values = torch.zeros(output_grid * output_grid, device=images.device)
            for row, column in path:
                target_row = min(output_grid - 1, row * output_grid // skeleton_grid)
                target_column = min(output_grid - 1, column * output_grid // skeleton_grid)
                values[target_row * output_grid + target_column] += 1
            signature = tuple((values > 0).cpu().tolist())
            if any(signature) and signature not in signatures:
                signatures.add(signature)
                kept.append(values)
        if not kept:
            kept = [torch.ones(output_grid * output_grid, device=images.device)]
        for path_index, values in enumerate(kept[:max_paths]):
            maps[batch_index, path_index] = normalize_evidence(
                values[None], ink_patches[batch_index:batch_index + 1]
            )[0]
            valid[batch_index, path_index] = True
    return maps, valid, skeleton


def path_priority_maps(path_maps, ink_mass, grid, valid=None, distance_scale=1.0):
    """Expand path cores into fixed-budget erasure priorities by spatial distance."""
    if path_maps.ndim != 3 or path_maps.shape[-1] != grid * grid:
        raise ValueError("Invalid path maps")
    if ink_mass.shape != (len(path_maps), grid * grid):
        raise ValueError("Invalid ink mass")
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(grid, device=path_maps.device),
        torch.arange(grid, device=path_maps.device), indexing="ij"
    ), dim=-1).reshape(-1, 2).float()
    result = torch.zeros_like(path_maps.float())
    for batch_index in range(len(path_maps)):
        for path_index in range(path_maps.shape[1]):
            if valid is not None and not bool(valid[batch_index, path_index]):
                continue
            support = path_maps[batch_index, path_index] > 0
            if not support.any():
                continue
            distance = torch.cdist(coordinates, coordinates[support], p=1).min(-1).values
            priority = torch.exp(-distance / distance_scale) * ink_mass[batch_index]
            result[batch_index, path_index] = priority + path_maps[batch_index, path_index]
    return result


def local_photo_correspondence(path_features, photo_patches, top_k=4):
    """Mean top-patch cosine from every sketch path to class-representative photos."""
    if path_features.ndim != 3 or photo_patches.ndim != 4:
        raise ValueError("Expected path [B,K,D] and photo patches [B,M,P,D]")
    path_features = F.normalize(path_features.float(), dim=-1)
    photo_patches = F.normalize(photo_patches.float(), dim=-1)
    similarities = torch.einsum("bkd,bmpd->bkmp", path_features, photo_patches)
    count = min(top_k, similarities.shape[-1])
    return similarities.topk(count, dim=-1).values.mean(-1).mean(-1)


def normalize_evidence(scores, ink_mass, eps=1e-8):
    """Normalize nonnegative scores on ink support with an ink-only fallback."""
    if scores.shape != ink_mass.shape:
        raise ValueError("Evidence and ink maps must have identical shapes")
    values = scores.float().clamp_min(0) * ink_mass.float().clamp_min(0).sqrt()
    fallback = ink_mass.float().clamp_min(0)
    empty_ink = fallback.sum(-1, keepdim=True) <= eps
    fallback = torch.where(empty_ink, torch.ones_like(fallback), fallback)
    values = torch.where(values.sum(-1, keepdim=True) > eps, values, fallback)
    return values / values.sum(-1, keepdim=True).clamp_min(eps)


def graph_smooth_evidence(weights, ink_mass, grid, steps=1, mix=0.25, eps=1e-8):
    """Diffuse evidence only through four-neighbour ink support on the patch lattice."""
    if weights.shape != ink_mass.shape or weights.shape[-1] != grid * grid:
        raise ValueError("Invalid evidence graph geometry")
    if steps < 0 or not 0 <= mix <= 1:
        raise ValueError("Graph steps and mix are out of range")
    x = weights.float().reshape(-1, 1, grid, grid)
    support = (ink_mass.float().reshape_as(x) > eps).float()
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    for _ in range(steps):
        neighbour_sum = F.conv2d(x * support, kernel, padding=1)
        neighbour_count = F.conv2d(support, kernel, padding=1)
        neighbour = neighbour_sum / neighbour_count.clamp_min(1)
        x = ((1 - mix) * x + mix * neighbour) * support
    x = x.flatten(1)
    fallback = normalize_evidence(torch.ones_like(x), ink_mass, eps)
    return torch.where(x.sum(-1, keepdim=True) > eps,
                       x / x.sum(-1, keepdim=True).clamp_min(eps), fallback)


def resize_evidence(weights, source_grid, target_grid, target_ink, graph_steps=1, graph_mix=0.25):
    if weights.shape[-1] != source_grid * source_grid:
        raise ValueError("Source evidence grid does not match")
    x = F.interpolate(weights.float().reshape(-1, 1, source_grid, source_grid),
                      size=(target_grid, target_grid), mode="area").flatten(1)
    x = normalize_evidence(x, target_ink)
    return graph_smooth_evidence(x, target_ink, target_grid, graph_steps, graph_mix)


def evidence_entropy(weights, eps=1e-8):
    values = weights.float().clamp_min(eps)
    return -(values * values.log()).sum(-1)


def erase_by_patch_evidence(images, weights, fraction, threshold=0.08, softness=0.12):
    """Whiten the highest-evidence ink patches using an approximately exact ink budget."""
    if not 0 < fraction < 1:
        raise ValueError("Erasure fraction must be in (0,1)")
    grid = math.isqrt(weights.shape[-1])
    if grid * grid != weights.shape[-1]:
        raise ValueError("Evidence must describe a square grid")
    ink = ink_strength(images, threshold, softness)
    masses = F.adaptive_avg_pool2d(ink[:, None], (grid, grid))[:, 0].flatten(1)
    coefficients = torch.zeros_like(masses)
    for row in range(len(images)):
        total = masses[row].sum()
        if total <= 1e-8:
            continue
        target = fraction * total
        order = torch.argsort(weights[row].float(), descending=True, stable=True)
        accumulated = masses.new_zeros(())
        for index in order.tolist():
            mass = masses[row, index]
            if mass <= 1e-8:
                continue
            remaining = target - accumulated
            if remaining <= 0:
                break
            coefficient = torch.minimum(mass.new_ones(()), remaining / mass)
            coefficients[row, index] = coefficient
            accumulated = accumulated + coefficient * mass
    mask = F.interpolate(coefficients.reshape(-1, 1, grid, grid),
                         size=images.shape[-2:], mode="nearest")[:, 0]
    alpha = (mask * ink).clamp(0, 1)
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    rgb = (images.float() * std + mean).clamp(0, 1)
    erased = rgb * (1 - alpha[:, None]) + alpha[:, None]
    erased = (erased - mean) / std
    removed = (mask * ink).sum((1, 2)) / ink.sum((1, 2)).clamp_min(1e-8)
    return erased.to(images.dtype), removed


def similarity_field(query, gallery):
    return F.normalize(query.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T


def _weighted_mean(values, weights, eps=1e-8):
    weights = weights.float().clamp_min(0)
    return (values.float() * weights).sum() / weights.sum().clamp_min(eps)


def hellinger_loss(student, teacher, confidence=None, eps=1e-8):
    if student.shape != teacher.shape:
        raise ValueError("Student and teacher evidence maps must match")
    per_sample = 0.5 * (student.float().clamp_min(eps).sqrt() -
                        teacher.detach().to(student.device).float().clamp_min(eps).sqrt()).square().sum(-1)
    if confidence is None:
        return per_sample.mean()
    return _weighted_mean(per_sample, confidence.to(student.device))


def centered_field_alignment(student_query, student_gallery, teacher_query, teacher_gallery,
                             confidence=None, eps=1e-8):
    student = similarity_field(student_query, student_gallery)
    with torch.no_grad():
        teacher = similarity_field(teacher_query.to(student.device), teacher_gallery.to(student.device))
    student = student - student.mean(-1, keepdim=True)
    teacher = teacher - teacher.mean(-1, keepdim=True)
    per_sample = 1 - F.cosine_similarity(student, teacher, dim=-1, eps=eps)
    loss = per_sample.mean() if confidence is None else _weighted_mean(per_sample, confidence.to(student.device))
    return loss, 1 - per_sample.detach().mean()


def counterfactual_field_alignment(clean_student, masked_student, student_gallery,
                                   clean_teacher, masked_teacher, teacher_gallery,
                                   confidence=None, magnitude_weight=0.0, eps=1e-8):
    student = similarity_field(clean_student, student_gallery) - similarity_field(masked_student, student_gallery)
    with torch.no_grad():
        teacher = (similarity_field(clean_teacher.to(student.device), teacher_gallery.to(student.device)) -
                   similarity_field(masked_teacher.to(student.device), teacher_gallery.to(student.device)))
    student_centered = student - student.mean(-1, keepdim=True)
    teacher_centered = teacher - teacher.mean(-1, keepdim=True)
    direction = 1 - F.cosine_similarity(student_centered, teacher_centered, dim=-1, eps=eps)
    student_rms = student_centered.square().mean(-1).sqrt()
    teacher_rms = teacher_centered.square().mean(-1).sqrt()
    scale = teacher_rms.mean().clamp_min(eps)
    magnitude = F.smooth_l1_loss(student_rms / scale, teacher_rms / scale,
                                 reduction="none", beta=0.5)
    per_sample = direction + magnitude_weight * magnitude
    loss = per_sample.mean() if confidence is None else _weighted_mean(per_sample, confidence.to(student.device))
    return loss, {
        "effect_cosine": (1 - direction).mean().detach(),
        "student_effect_rms": student_rms.mean().detach(),
        "teacher_effect_rms": teacher_rms.mean().detach(),
        "effect_magnitude_ratio": (student_rms.mean() / teacher_rms.mean().clamp_min(eps)).detach(),
    }


class ResidualMap(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class StrokeGraphEvidenceHead(nn.Module):
    """Student-predicted stroke evidence that causally contributes to retrieval."""

    def __init__(self, width, grid, bottleneck=128, beta=0.1, temperature=0.07,
                 graph_steps=1, graph_mix=0.25):
        super().__init__()
        if width < 1 or grid < 1 or bottleneck < 1 or beta < 0 or temperature <= 0:
            raise ValueError("Invalid stroke evidence head configuration")
        self.grid = grid
        self.beta = beta
        self.temperature = temperature
        self.graph_steps = graph_steps
        self.graph_mix = graph_mix
        key_width = min(bottleneck, width)
        self.key = nn.Linear(width, key_width, bias=False)
        self.query = nn.Linear(width, key_width, bias=False)
        self.evidence_adapter = ResidualMap(width, bottleneck)
        self.fusion = ResidualMap(width, bottleneck)

    def forward(self, native, dense, ink_mass):
        if dense.shape[1] != self.grid * self.grid or dense.shape[-1] != native.shape[-1]:
            raise ValueError("Dense feature geometry is incompatible with the evidence head")
        native = F.normalize(native.float(), dim=-1)
        dense = F.normalize(dense.float(), dim=-1)
        logits = torch.einsum("bnd,bd->bn", self.key(dense), self.query(native))
        logits = logits / (math.sqrt(self.key.out_features) * self.temperature)
        log_ink = ink_mass.float().clamp_min(1e-12).log().masked_fill(ink_mass <= 0, -torch.inf)
        empty = ~torch.isfinite(log_ink).any(-1, keepdim=True)
        log_ink = torch.where(empty, torch.zeros_like(log_ink), log_ink)
        weights = (logits + log_ink).softmax(-1)
        weights = graph_smooth_evidence(weights, ink_mass, self.grid,
                                        self.graph_steps, self.graph_mix)
        pooled = torch.einsum("bn,bnd->bd", weights, dense)
        evidence = F.normalize(pooled + self.evidence_adapter(pooled), dim=-1)
        correction = self.beta * self.fusion(evidence)
        descriptor = F.normalize(native + correction, dim=-1)
        return {
            "descriptor": descriptor,
            "native": native,
            "dense": dense,
            "weights": weights,
            "evidence": evidence,
            "correction": correction,
            "ink_mass": ink_mass.float(),
        }
