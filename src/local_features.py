"""Scoped final-token capture. Original encoder forward/inference is untouched."""
import math
import torch
from torch.nn import functional as F


def area_pool(patches, target_grid):
    source = math.isqrt(patches.shape[1])
    if source*source != patches.shape[1] or not 1 <= target_grid <= source:
        raise ValueError("Pooling requires a square patch grid and no upsampling.")
    if target_grid == source:
        return patches.float()
    # Exact overlap, deterministic CUDA backward; unlike adaptive_avg_pool2d.
    with torch.autocast(device_type=patches.device.type, enabled=False):
        edges = torch.arange(target_grid+1, device=patches.device, dtype=torch.float32)
        edges = edges * source/target_grid
        cells = torch.arange(source, device=patches.device, dtype=torch.float32)
        weights = (torch.minimum(edges[1:, None], cells[None]+1)
                   - torch.maximum(edges[:-1, None], cells[None])).clamp_min(0)
        weights = weights / (source/target_grid)
        x = patches.float().reshape(-1, source, source, patches.shape[-1]).permute(0, 3, 1, 2)
        x = weights @ x @ weights.t()
        return x.permute(0, 2, 3, 1).reshape(-1, target_grid**2, patches.shape[-1])


def project_patches(patches, visual):
    """Apply final LN and projection independently to tokens (a hypothesis)."""
    with torch.autocast(device_type=patches.device.type, enabled=False):
        layer = visual.ln_post
        x = F.layer_norm(patches.float(), layer.normalized_shape,
                         layer.weight.float() if layer.weight is not None else None,
                         layer.bias.float() if layer.bias is not None else None, layer.eps)
        if visual.proj is not None:
            x = x @ visual.proj.float()
        return x


class FinalPatches:
    def __init__(self, visual, batch_first=False, grid=None):
        self.visual, self.batch_first, self.grid = visual, batch_first, grid
        self.values = []
        self.handle = None

    def __enter__(self):
        count = self.visual.positional_embedding.shape[0]-1
        def capture(_module, _inputs, output):
            x = output if self.batch_first else output.transpose(0, 1)
            # Both implementations append prompt tokens after the image tokens.
            patches = x[:, 1:count+1].float()
            if self.grid is not None:
                patches = area_pool(patches, self.grid)
            self.values.append(patches)
        self.handle = self.visual.transformer.resblocks[-1].register_forward_hook(capture)
        return self

    def __exit__(self, *_exception):
        self.handle.remove()
        self.handle = None


def spatial_structure_loss(ct, student_patches):
    from src.structural_losses import structure
    grid = math.isqrt(ct.shape[-1])
    cs = structure(area_pool(student_patches, grid))
    return (ct.detach().float()-cs).square().mean()
