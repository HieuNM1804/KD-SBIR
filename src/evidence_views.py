"""Deterministic image-space interventions; input is a normalized CPU tensor."""
import math
import torch
from torch.nn import functional as F


def regions(size, area):
    width = max(1, min(size - 1, round(size * math.sqrt(area))))
    end = size - width
    return [(y, x, y + width, x + width) for y, x in
            ((0, 0), (0, end), (end, 0), (end, end), (end // 2, end // 2))]


def intervene(image, box, fill="mean", donor=None):
    # Generate on CPU in both target preparation and student training. This
    # avoids CPU/GPU reduction differences changing the cached teacher view.
    if image.device.type != "cpu":
        raise ValueError("Evidence views must be generated on CPU")
    y0, x0, y1, x1 = box
    patch = image[:, y0:y1, x0:x1]
    if patch.numel() == 0:
        raise ValueError("Empty evidence region")
    result = image.clone()
    if fill == "mean":
        replacement = patch.mean(dim=(-2, -1), keepdim=True)
    elif fill == "donor" and donor is not None:
        replacement = donor[:, y0:y1, x0:x1]
    else:
        raise ValueError("donor fill requires the recorded donor image")
    result[:, y0:y1, x0:x1] = replacement
    crop = F.interpolate(patch[None], size=image.shape[-2:], mode="bilinear",
                         align_corners=False)[0]
    return result, crop
