"""Read stroke evidence from frozen CLIP attention without a learned head."""

import torch
from torch.nn import functional as F

from src.stroke_graph import (
    cls_patch_attention,
    normalize_evidence,
    projected_patch_features,
)


def native_prompt_evidence(visual, sequence, native, ink_mass):
    """Keep the native descriptor; gradients pass through tokens to prompts.

    The attention weights are the mean final-block CLS-to-patch attention,
    renormalized on ink support. This is a localization readout, not a claim
    that attention alone measures a patch's causal retrieval contribution.
    """
    raw_attention = cls_patch_attention(visual, sequence)
    weights = normalize_evidence(raw_attention, ink_mass)
    dense = F.normalize(projected_patch_features(visual, sequence), dim=-1)
    pooled = torch.einsum("bn,bnd->bd", weights, dense)
    return {
        "descriptor": native,
        "native": native,
        "dense": dense,
        "weights": weights,
        "evidence": F.normalize(pooled, dim=-1),
        "correction": torch.zeros_like(native),
        "ink_mass": ink_mass.float(),
    }
