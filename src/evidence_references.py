"""Balanced train-only reference selection and architecture-independent scores."""
from pathlib import Path
import torch
from torch.nn import functional as F


def labels_for(paths, categories):
    lookup = {name: i for i, name in enumerate(categories)}
    return torch.tensor([lookup[Path(p).parent.name] for p in paths])


def balanced_indices(labels, count, seed):
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        if count and len(indices) < count:
            raise ValueError(f"Class {label.item()} has fewer than {count} samples")
        order = torch.randperm(len(indices), generator=generator)
        selected.extend(indices[order[:count or len(indices)]].tolist())
    return torch.tensor(selected, dtype=torch.long)


def prototypes(features, labels, classes):
    unit = F.normalize(features.float(), dim=-1)
    output = []
    for c in range(classes):
        current = unit[labels.to(unit.device) == c]
        if not len(current):
            raise ValueError(f"Missing reference class {c}")
        output.append(F.normalize(current.mean(0), dim=0))
    return torch.stack(output).detach()


def scores(features, references):
    with torch.autocast(device_type=features.device.type, enabled=False):
        return F.normalize(features.float(), dim=-1) @ references.detach().float().t()


def margin(values, label):
    others = values.clone()
    others[..., label] = -torch.inf
    return values[..., label] - others.max(dim=-1).values
