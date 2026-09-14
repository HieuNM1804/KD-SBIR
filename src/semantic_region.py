"""Crop-semantic distillation and region attention in the deployed CLIP space.

Teacher vectors are aligned by a FIXED semi-orthogonal map fitted on seen data.
The student keeps its native descriptor and adds a zero-initialized residual.
No pair ranking, contrastive objective, teacher attention hook or AV target.
"""
import math
from contextlib import AbstractContextManager

import torch
from torch import nn
from torch.nn import functional as F


def region_boxes(size, grid):
    if not 1 <= grid <= size:
        raise ValueError('Region grid must be between 1 and image size')
    edges = [round(i * size / grid) for i in range(grid + 1)]
    return [(edges[y], edges[x], edges[y+1], edges[x+1])
            for y in range(grid) for x in range(grid)]


def crop_regions(images, grid):
    """Crop AFTER the ordinary deterministic image transform; same target geometry."""
    if images.shape[-2] != images.shape[-1]:
        raise ValueError('Expected square transformed inputs')
    size = images.shape[-1]
    return torch.stack([F.interpolate(images[..., y0:y1, x0:x1].float(),
                       size=(size, size), mode='bilinear', align_corners=False,
                       antialias=True) for y0, x0, y1, x1 in region_boxes(size, grid)], dim=1)


def area_prior(size, patch_grid, region_grid):
    """Exact area overlap; e.g. a 7x7 patch lattice need not divide into 2x2."""
    edges = torch.linspace(0, size, patch_grid + 1)
    rows = []
    for y0, x0, y1, x1 in region_boxes(size, region_grid):
        dy = (torch.minimum(edges[1:], edges.new_tensor(y1)) -
              torch.maximum(edges[:-1], edges.new_tensor(y0))).clamp_min(0)
        dx = (torch.minimum(edges[1:], edges.new_tensor(x1)) -
              torch.maximum(edges[:-1], edges.new_tensor(x0))).clamp_min(0)
        rows.append((dy[:, None] * dx[None, :]).flatten())
    result = torch.stack(rows)
    return result / result.sum(-1, keepdim=True)


def content_prior(images, modality, grid):
    """Sketch ink MASS per region; photo regions have equal visibility prior."""
    b, _, size, _ = images.shape
    if modality == 'photo':
        return images.new_full((b, grid**2), 1 / grid**2, dtype=torch.float32)
    if modality != 'sketch':
        raise ValueError(modality)
    mean = images.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = images.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    ink = ((images.float() * std + mean).mean(1) < .8).float()
    masses = torch.stack([ink[:, y0:y1, x0:x1].sum((1, 2))
                         for y0, x0, y1, x1 in region_boxes(size, grid)], dim=1)
    total = masses.sum(1, keepdim=True)
    return torch.where(total > 0, masses / total.clamp_min(1),
                       torch.full_like(masses, 1 / grid**2))


def semantic_weights(full, crops, prior, temperature):
    """Agreement with teacher GLOBAL semantics, without category/test labels."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    scores = torch.einsum('bd,brd->br', F.normalize(full.float(), dim=-1),
                          F.normalize(crops.float(), dim=-1)) / temperature
    log_prior = prior.float().clamp_min(1e-30).log().masked_fill(prior == 0, -torch.inf)
    return (scores + log_prior).softmax(-1)


def fit_alignment(teacher, student):
    """Uncentered orthogonal Procrustes, teacher D >= student d. No learned W."""
    if teacher.shape[0] != student.shape[0] or teacher.shape[1] < student.shape[1]:
        raise ValueError('Alignment requires matched samples and teacher width >= student width')
    if teacher.shape[0] < student.shape[1]:
        raise ValueError('Need at least student-width calibration samples')
    t = F.normalize(teacher.double().cpu(), dim=-1)
    s = F.normalize(student.double().cpu(), dim=-1)
    u, _, vh = torch.linalg.svd(t.T @ s, full_matrices=False)
    return (u @ vh).float()


class DenseRegionCapture(AbstractContextManager):
    """CLIPSelf-style LOCAL final-block readout; normal CLS forward is unchanged."""
    def __init__(self, visual):
        self.visual = visual
        self.inputs = []

    def __enter__(self):
        self.handle = self.visual.transformer.resblocks[-1].register_forward_pre_hook(
            lambda _module, args: self.inputs.append(args[0]))
        return self

    def __exit__(self, *args):
        self.handle.remove()

    def dense_features(self):
        if len(self.inputs) != 1:
            raise RuntimeError('Expected exactly one visual forward in region capture')
        visual = self.visual
        block = visual.transformer.resblocks[-1]
        count = visual.positional_embedding.shape[0] - 1
        # OpenAI CLIP is sequence-first. Exclude CLS and appended visual prompts.
        x = self.inputs[0][1:1+count].permute(1, 0, 2)
        width = x.shape[-1]
        bias = block.attn.in_proj_bias
        v = F.linear(block.ln_1(x), block.attn.in_proj_weight[2*width:],
                     None if bias is None else bias[2*width:])
        x = x + block.attn.out_proj(v)
        x = x + block.mlp(block.ln_2(x))
        return (visual.ln_post(x) @ visual.proj).float()


class ResidualMap(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class SemanticRegionHead(nn.Module):
    def __init__(self, width, image_size, patch_grid, grid=2, bottleneck=64, beta=.5):
        super().__init__()
        self.grid, self.beta = grid, beta
        self.register_buffer('prior', area_prior(image_size, patch_grid, grid))
        key_width = min(128, width)
        self.key = nn.Linear(width, key_width, bias=False)
        self.queries = nn.Parameter(torch.zeros(grid**2, key_width))
        self.region_adapter = ResidualMap(width, bottleneck)
        self.gate = nn.Sequential(nn.Linear(2*width, bottleneck), nn.GELU(), nn.Linear(bottleneck, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.fusion = ResidualMap(width, bottleneck)

    def forward(self, native, dense, visibility, mode):
        if mode not in ('semantic', 'uniform', 'random', 'global'):
            raise ValueError(mode)
        native = F.normalize(native.float(), dim=-1)
        scores = torch.einsum('bnk,rk->brn', self.key(F.normalize(dense.float(), dim=-1)),
                              self.queries) / math.sqrt(self.queries.shape[-1])
        spatial = self.prior.clamp_min(1e-30).log().masked_fill(self.prior == 0, -torch.inf)
        attention = (scores + spatial[None]).softmax(-1)
        pooled = torch.einsum('brn,bnd->brd', attention, dense.float())
        regions = F.normalize(pooled + self.region_adapter(pooled), dim=-1)
        global_expanded = native[:, None].expand_as(regions)
        logits = self.gate(torch.cat((regions, global_expanded), dim=-1)).squeeze(-1)
        log_prior = visibility.clamp_min(1e-30).log().masked_fill(visibility == 0, -torch.inf)
        weights = (logits + log_prior).softmax(-1)
        source = native if mode == 'global' else torch.einsum('br,brd->bd', weights, regions)
        correction = self.beta * self.fusion(source)
        descriptor = F.normalize(native + correction, dim=-1)
        return {'descriptor': descriptor, 'native': native, 'regions': regions,
                'weights': weights, 'attention': attention, 'correction': correction}


def region_losses(output, teacher_global, crop_targets, gate_target, visibility, reference, args):
    target = F.normalize(teacher_global.float().detach(), dim=-1)
    reference = F.normalize(reference.float().detach(), dim=-1)
    regions = F.normalize(crop_targets.float().detach(), dim=-1)
    values = {
        'descriptor': (1 - (output['descriptor'] * target).sum(-1)).mean(),
        # SAME visibility weighting for semantic/uniform/random; only gate target changes.
        'region': ((1 - (output['regions'] * regions).sum(-1)) * visibility).sum(-1).mean(),
        'gate': (gate_target * (gate_target.clamp_min(1e-30).log() -
                               output['weights'].clamp_min(1e-30).log())).sum(-1).mean(),
        'reference': (1 - (output['native'] * reference).sum(-1)).mean(),
        'spread': F.relu(args.region_spread_fraction * target.std(0, correction=0) -
                         output['descriptor'].std(0, correction=0)).square().mean() * target.shape[-1],
    }
    if args.region_mode == 'global':
        values['region'] = values['region'] * 0
        values['gate'] = values['gate'] * 0
    total = sum(getattr(args, 'lambda_' + k) * v for k, v in values.items())
    return total, values
