"""Differentiable patch contribution to the CLS attention output."""

from contextlib import AbstractContextManager
import math
import torch
from torch import nn
from torch.nn import functional as F


def patch_attention_output(attn, query, key, value, patch_count):
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
        pooled = (a[..., 1 : 1 + patch_count] @ v).reshape(batch, width)
        return F.linear(pooled, attn.out_proj.weight.float(), None)


class PatchOutputCapture(AbstractContextManager):
    """Scoped hook on the final attention block; captured values retain gradients."""

    def __init__(self, visual):
        self.attn = visual.transformer.resblocks[-1].attn
        self.patch_count = visual.positional_embedding.shape[0] - 1
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
                patch_attention_output(module, q, k, v, self.patch_count)
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
