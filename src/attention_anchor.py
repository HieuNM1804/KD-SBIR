"""Attention-weighted visual vocabulary descriptors, used in training AND retrieval.

Tokens are last-block projected V vectors (concatenated heads, then W_O without
output bias). Weights are mean-head CLS attention over image patches, conditioned
on attending a patch. No spatial teacher/student correspondence is required.
"""
from contextlib import AbstractContextManager
import math
import torch
from torch import nn
from torch.nn import functional as F


def token_evidence(attn, query, key, value, patch_count):
    if not isinstance(attn, nn.MultiheadAttention):
        raise TypeError('Anchor capture requires torch MultiheadAttention')
    if attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
        raise ValueError('Extra attention tokens are unsupported')
    if attn.training and attn.dropout:
        raise ValueError('Stochastic attention dropout is incompatible with cached targets')
    if not attn.batch_first:
        query, key, value = [x.transpose(0, 1) for x in (query, key, value)]
    if not 0 < patch_count < key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError('Invalid patch span')
    width, heads = attn.embed_dim, attn.num_heads
    dim = width // heads
    wq, wk, wv = (attn.in_proj_weight.chunk(3) if attn.in_proj_weight is not None else
                  (attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight))
    bq, bk, bv = ((None,)*3 if attn.in_proj_bias is None else attn.in_proj_bias.chunk(3))
    with torch.autocast(device_type=query.device.type, enabled=False):
        def linear(x, w, b):
            return F.linear(x.float(), w.float(), None if b is None else b.float())
        batch = query.shape[0]
        q = linear(query[:, :1], wq, bq).reshape(batch, 1, heads, dim).transpose(1, 2)
        k = linear(key, wk, bk).reshape(batch, -1, heads, dim).transpose(1, 2)
        a = (q @ k.transpose(-1, -2) / math.sqrt(dim)).softmax(-1)
        a = a[:, :, 0, 1:1+patch_count].mean(1)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
        v = linear(value[:, 1:1+patch_count], wv, bv)
        return F.linear(v, attn.out_proj.weight.float(), None), a


class AnchorCapture(AbstractContextManager):
    def __init__(self, visual):
        self.attn = visual.transformer.resblocks[-1].attn
        self.patch_count = visual.positional_embedding.shape[0] - 1
        self.values = []
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError('Capture already active')
        self.values.clear()
        def hook(module, args, kwargs):
            if any(kwargs.get(k) is not None for k in ('attn_mask', 'key_padding_mask')) or kwargs.get('is_causal', False):
                raise ValueError('Masked attention is unsupported')
            qkv = [args[i] if len(args) > i else kwargs[n] for i,n in enumerate(('query','key','value'))]
            self.values.append(token_evidence(module, *qkv, self.patch_count))
        self.handle = self.attn.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        self.handle = None
        return False


def anchor_distribution(tokens, attention, anchors, center, temperature, pooling='attention'):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Anchor temperature must be positive and finite')
    if tokens.ndim != 3 or attention.shape != tokens.shape[:2] or tokens.shape[-1] != anchors.shape[-1]:
        raise ValueError('Invalid anchor token/attention/anchor shapes')
    if pooling not in ('attention', 'uniform'):
        raise ValueError('Unknown anchor pooling')
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        features = F.normalize(F.normalize(tokens.float(), dim=-1) - center.float(), dim=-1)
        logits = features @ F.normalize(anchors.float(), dim=-1).T / temperature
        assignments = logits.softmax(-1)
        weights = attention.float() if pooling == 'attention' else torch.ones_like(attention, dtype=torch.float32)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        p = (weights[..., None] * assignments).sum(1).clamp_min(1e-12)
        return p / p.sum(-1, keepdim=True)


def hellinger_descriptor(probabilities):
    p = probabilities.float().clamp_min(1e-12)
    return (p / p.sum(-1, keepdim=True)).sqrt()


def anchor_kd(student, teacher):
    if student.ndim != 2 or student.shape != teacher.shape:
        raise ValueError('Anchor KD requires matching [batch, anchors] probabilities')
    with torch.autocast(device_type=student.device.type, enabled=False):
        target = teacher.detach().to(device=student.device, dtype=torch.float32)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
        return F.kl_div(student.float().clamp_min(1e-12).log(), target, reduction='batchmean')


class AnchorRouter(nn.Module):
    """Shared student vocabulary. All learned parameters are used at inference."""
    def __init__(self, width, count=256, temperature=.1, pooling='attention', seed=42):
        super().__init__()
        if count < 2:
            raise ValueError('Need at least two anchors')
        generator = torch.Generator().manual_seed(seed)
        self.anchors = nn.Parameter(F.normalize(torch.randn(count, width, generator=generator), dim=-1))
        self.center = nn.Parameter(torch.zeros(width))
        self.temperature, self.pooling = temperature, pooling

    def forward(self, tokens, attention):
        return anchor_distribution(tokens, attention, self.anchors, self.center, self.temperature, self.pooling)

    def descriptor(self, tokens, attention):
        return hellinger_descriptor(self(tokens, attention))


@torch.no_grad()
def fit_vocabulary(tokens, count, iterations=25, seed=42):
    """Deterministic spherical k-means on a balanced, seen-only token sample.

    Fit on CPU: index_add is deterministic here. No validation images/classes
    or external text prototypes enter the vocabulary.
    """
    x = F.normalize(tokens.detach().float().cpu(), dim=-1)
    if x.ndim != 2 or len(x) < count or count < 2 or iterations < 1 or not torch.isfinite(x).all():
        raise ValueError('Invalid vocabulary fitting sample/count/iterations')
    center = x.mean(0)
    x = F.normalize(x-center, dim=-1)
    generator = torch.Generator().manual_seed(seed)
    anchors = x[torch.randperm(len(x), generator=generator)[:count]].clone()
    for _ in range(iterations):
        assignments = torch.cat([(part @ anchors.T).argmax(-1) for part in x.split(2048)])
        sums = torch.zeros_like(anchors).index_add_(0, assignments, x)
        counts = torch.bincount(assignments, minlength=count)
        active = (counts > 0) & (sums.norm(dim=-1) > 1e-8)
        anchors[active] = F.normalize(sums[active], dim=-1)
    return anchors, center, {'fit_tokens': len(x), 'occupied_anchors': int((counts > 0).sum()),
                             'iterations': iterations}
