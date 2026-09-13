"""Differentiable patch contribution to the CLS attention output."""

from contextlib import AbstractContextManager
import math
import torch
from torch import nn
from torch.nn import functional as F


def region_patch_weights(patch_count, region_grid, device=None):
    """Fraction of each square patch inside each normalized image region.

    Columns sum to one. Fractional overlaps preserve contributions when, e.g.,
    a 7x7 student and 16x16 teacher are both partitioned into 2x2 regions.
    Regions and patches are flattened in row-major order.
    """
    side = math.isqrt(patch_count)
    if side * side != patch_count or not 1 <= region_grid <= side:
        raise ValueError('Regional AV requires a square patch grid and 1 <= region_grid <= grid size')
    patches = torch.arange(side, device=device, dtype=torch.float32)
    regions = torch.arange(region_grid, device=device, dtype=torch.float32)
    overlap = (torch.minimum((regions[:, None] + 1) / region_grid, (patches[None, :] + 1) / side)
               - torch.maximum(regions[:, None] / region_grid, patches[None, :] / side)).clamp_min(0) * side
    return torch.einsum('ai,bj->abij', overlap, overlap).reshape(region_grid**2, patch_count)


def patch_attention_output(attn, query, key, value, patch_count, region_grid=None):
    """Sum A[CLS,patch] V_patch W_O across heads; excludes output bias.

    The softmax denominator retains ALL keys (CLS, image and prompt tokens).
    Only the CLS row is computed, in FP32. Does not replace the model forward.
    """
    if not isinstance(attn, nn.MultiheadAttention):
        raise TypeError("Patch output KD requires nn.MultiheadAttention")
    if attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
        raise ValueError("Extra MHA bias/zero tokens are unsupported")
    if attn.training and attn.dropout:
        raise ValueError("Stochastic attention dropout is incompatible with cached KD")
    if not attn.batch_first:
        query, key, value = (x.transpose(0, 1) for x in (query, key, value))
    if key.shape[1] != value.shape[1] or not 0 < patch_count <= key.shape[1] - 1:
        raise ValueError("Invalid image patch span")
    width, heads = attn.embed_dim, attn.num_heads
    dim = width // heads
    if attn.in_proj_weight is None:
        wq, wk, wv = attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight
    else:
        wq, wk, wv = attn.in_proj_weight.chunk(3, dim=0)
    bq, bk, bv = (
        (None,) * 3 if attn.in_proj_bias is None else attn.in_proj_bias.chunk(3)
    )
    with torch.autocast(device_type=query.device.type, enabled=False):

        def linear(x, w, b):
            return F.linear(x.float(), w.float(), None if b is None else b.float())

        batch = query.shape[0]
        q = linear(query[:, :1], wq, bq).reshape(batch, 1, heads, dim).transpose(1, 2)
        k = linear(key, wk, bk).reshape(batch, -1, heads, dim).transpose(1, 2)
        v = (
            linear(value[:, 1 : 1 + patch_count], wv, bv)
            .reshape(batch, patch_count, heads, dim)
            .transpose(1, 2)
        )
        a = (q @ k.transpose(-1, -2) / math.sqrt(dim)).softmax(-1)
        if region_grid is None:
            pooled = (a[..., 1 : 1 + patch_count] @ v).reshape(batch, width)
        else:
            regions = region_patch_weights(patch_count, region_grid, query.device)
            pooled = torch.einsum('bhp,rp,bhpd->brhd',
                                  a[:, :, 0, 1 : 1 + patch_count], regions, v)
            pooled = pooled.reshape(batch, region_grid**2, width)
        return F.linear(pooled, attn.out_proj.weight.float(), None)


class PatchOutputCapture(AbstractContextManager):
    """Scoped hook on the final attention block; captured values retain gradients."""

    def __init__(self, visual, region_grid=None):
        self.attn = visual.transformer.resblocks[-1].attn
        self.patch_count = visual.positional_embedding.shape[0] - 1
        self.region_grid = region_grid
        self.values = []
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Capture cannot be nested on itself")
        self.values.clear()

        def hook(module, args, kwargs):
            if (
                kwargs.get("attn_mask") is not None
                or kwargs.get("key_padding_mask") is not None
                or kwargs.get("is_causal", False)
            ):
                raise ValueError("Masked visual attention is unsupported")
            q, k, v = [
                args[i] if len(args) > i else kwargs[name]
                for i, name in enumerate(("query", "key", "value"))
            ]
            self.values.append(
                patch_attention_output(module, q, k, v, self.patch_count, self.region_grid)
            )

        self.handle = self.attn.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        self.handle = None
        return False


def make_projector(input_dim, output_dim, seed):
    # Auxiliary heads must not perturb the baseline's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Linear(input_dim, output_dim, bias=False)


def feature_cosine_kd(student, teacher, projector):
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError("Expected matching feature batches")
    with torch.autocast(device_type=student.device.type, enabled=False):
        predicted = projector(student.float())
        target = teacher.detach().to(device=predicted.device, dtype=torch.float32)
        return (1 - F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)).mean()


def regional_cosine_kd(student, teacher, projector):
    """Mean cosine KD over images and regions; one shared projector for all regions."""
    if student.ndim != 3 or teacher.ndim != 3 or student.shape[:2] != teacher.shape[:2]:
        raise ValueError('Regional AV requires matching [batch, regions, width] tensors')
    return feature_cosine_kd(student.flatten(0, 1), teacher.flatten(0, 1), projector)


def relational_av_kd(student_sketch, student_photo, teacher_sketch, teacher_photo,
                     temperature=0.07):
    """Mean teacher->student KL for sketch->photo and photo->sketch.

    Each space is normalized independently; no cross-model projector or T^2
    scaling. Rows/columns must reference the same samples in teacher/student.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("AV temperature must be finite and positive")
    tensors = (student_sketch, student_photo, teacher_sketch, teacher_photo)
    if any(x.ndim != 2 for x in tensors):
        raise ValueError("AV relations require 2D feature batches")
    if (student_sketch.shape[0] != teacher_sketch.shape[0]
            or student_photo.shape[0] != teacher_photo.shape[0]
            or min(student_sketch.shape[0], student_photo.shape[0]) < 2):
        raise ValueError("Matching teacher/student batches with at least two samples required")
    with torch.autocast(device_type=student_sketch.device.type, enabled=False):
        s = F.normalize(student_sketch.float(), dim=-1) @ F.normalize(student_photo.float(), dim=-1).T
        with torch.no_grad():
            ts, tp = (x.detach().to(student_sketch.device, dtype=torch.float32)
                      for x in (teacher_sketch, teacher_photo))
            t = F.normalize(ts, dim=-1) @ F.normalize(tp, dim=-1).T
        return 0.5 * (
            F.kl_div(F.log_softmax(s / temperature, dim=-1),
                     F.softmax(t / temperature, dim=-1), reduction="batchmean")
            + F.kl_div(F.log_softmax(s.T / temperature, dim=-1),
                       F.softmax(t.T / temperature, dim=-1), reduction="batchmean")
        )
