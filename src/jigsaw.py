"""Conditional cross-modal jigsaw components for teacher pretraining.

The objective follows SpLIP's conditional cross-modal jigsaw formulation.  The
paper does not release its training implementation, so the permutation bank and
image tiling below are explicit, deterministic reproduction choices.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_permutation_bank(grid_size, num_permutations, seed):
    """Return unique, non-identity tile permutations generated deterministically."""
    tile_count = grid_size**2
    if grid_size < 2:
        raise ValueError("Jigsaw grid size must be at least 2.")
    if num_permutations < 2:
        raise ValueError("Jigsaw needs at least two permutation classes.")
    if num_permutations >= math.factorial(tile_count):
        raise ValueError(
            "The requested permutation count must leave out the identity class."
        )

    generator = torch.Generator().manual_seed(seed)
    identity = tuple(range(tile_count))
    permutations = []
    seen = {identity}
    while len(permutations) < num_permutations:
        candidate = tuple(torch.randperm(tile_count, generator=generator).tolist())
        if candidate not in seen:
            seen.add(candidate)
            permutations.append(candidate)
    return torch.tensor(permutations, dtype=torch.long)


def apply_jigsaw(images, permutation_bank, permutation_labels):
    """Apply one bank permutation to each image while preserving image shape."""
    if images.ndim != 4:
        raise ValueError("Jigsaw input must have shape [batch, channels, height, width].")
    if permutation_bank.ndim != 2:
        raise ValueError("Permutation bank must have shape [classes, tiles].")
    batch_size, channels, height, width = images.shape
    labels = permutation_labels.to(images.device).long()
    if labels.shape != (batch_size,):
        raise ValueError("Every image needs one permutation label.")
    if labels.min().item() < 0 or labels.max().item() >= len(permutation_bank):
        raise ValueError("Permutation label is outside the permutation bank.")

    tile_count = permutation_bank.shape[1]
    grid_size = math.isqrt(tile_count)
    if grid_size**2 != tile_count:
        raise ValueError("The permutation bank must describe a square grid.")

    work_height = math.ceil(height / grid_size) * grid_size
    work_width = math.ceil(width / grid_size) * grid_size
    work = images.float()
    if (work_height, work_width) != (height, width):
        work = F.interpolate(
            work,
            size=(work_height, work_width),
            mode="bilinear",
            align_corners=False,
        )

    tile_height = work_height // grid_size
    tile_width = work_width // grid_size
    tiles = (
        work.reshape(
            batch_size,
            channels,
            grid_size,
            tile_height,
            grid_size,
            tile_width,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch_size, tile_count, channels, tile_height, tile_width)
    )
    selected = permutation_bank.to(images.device)[labels]
    gather_index = selected[:, :, None, None, None].expand_as(tiles)
    shuffled_tiles = torch.gather(tiles, dim=1, index=gather_index)
    shuffled = (
        shuffled_tiles.reshape(
            batch_size,
            grid_size,
            grid_size,
            channels,
            tile_height,
            tile_width,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch_size, channels, work_height, work_width)
    )
    if (work_height, work_width) != (height, width):
        shuffled = F.interpolate(
            shuffled,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return shuffled.to(images.dtype)


class ConditionalJigsawSolver(nn.Module):
    """Small MLP that predicts a permutation from a pair of image features."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_permutations,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden = nn.Linear(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_permutations)

    def forward(self, conditioning_features, shuffled_sketch_features):
        if conditioning_features.shape != shuffled_sketch_features.shape:
            raise ValueError("Both jigsaw inputs must have the same feature shape.")
        pair = torch.cat(
            [conditioning_features.float(), shuffled_sketch_features.float()],
            dim=-1,
        )
        hidden = F.gelu(self.hidden(pair))
        return self.classifier(self.dropout(hidden))
