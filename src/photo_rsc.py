"""Photo-only KD-gradient self-challenging; no trainable state or extra loss."""

import math

import torch
from torch.nn import functional as F

from src.losses import loss_fn


def validate_rsc(probability, drop_fraction):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("--student_rsc_prob must be in [0, 1].")
    if not math.isfinite(drop_fraction) or not 0 <= drop_fraction < 1:
        raise ValueError("--student_rsc_drop must be in [0, 1).")


def photo_rsc_features(args, features, training=True):
    """Mask photo embedding coordinates selected by |d(existing KD)/d(photo)|.

    The probe uses detached FP32 embeddings, so it neither traverses the
    encoder nor accumulates parameter gradients. The final KD backward still
    traverses the original photo/sketch graphs. Each photo is selected with
    independent probability p; at most floor(D * drop_fraction) coordinates
    with strictly positive sensitivity are removed. Evaluation is an identity.
    """
    probability = getattr(args, "student_rsc_prob", 0.0)
    drop_fraction = getattr(args, "student_rsc_drop", 0.1)
    validate_rsc(probability, drop_fraction)
    photo = features[0]
    mask = torch.ones_like(photo, dtype=torch.bool)
    count = min(int(photo.shape[-1] * drop_fraction), photo.shape[-1] - 1)
    if (
        not training or not torch.is_grad_enabled() or not features[4]
        or probability == 0 or count <= 0
        or (args.lambda_domain <= 0 and args.lambda_modality <= 0)
    ):
        return features, mask

    selected = (
        torch.ones(photo.shape[0], device=photo.device, dtype=torch.bool)
        if probability == 1
        else torch.rand(photo.shape[0], device=photo.device) < probability
    )
    if not selected.any():
        return features, mask

    # Avoid AMP scaling/underflow when ranking sensitivities, including when
    # the encoder emits FP16. This is NOT an extra optimized objective.
    with torch.autocast(device_type=photo.device.type, enabled=False):
        probe = [x.detach().float() if torch.is_tensor(x) else x for x in features]
        probe[0].requires_grad_(True)
        probe_loss, _ = loss_fn(args, probe)
        importance = torch.autograd.grad(probe_loss, probe[0], create_graph=False)[0].abs()
        if not torch.isfinite(importance).all():
            raise RuntimeError("Non-finite photo RSC probe gradient.")
        # Stable ordering makes ties reproducible; zero gradients never drop.
        indices = importance.argsort(dim=-1, descending=True, stable=True)[:, :count]
        remove = (importance.gather(1, indices) > 0) & selected[:, None]
        mask.scatter_(1, indices, ~remove)
        masked = photo.float() * mask
        # Degenerate sparse features must not become a zero retrieval vector.
        valid = masked.norm(dim=-1) > 1e-12
        mask = mask | ~valid[:, None]
        changed = (~mask).any(dim=-1, keepdim=True)
        if not changed.any():
            return features, mask
        challenged = F.normalize(photo.float() * mask, dim=-1)
        challenged = torch.where(changed, challenged, photo.float())
    return (challenged, *features[1:]), mask
