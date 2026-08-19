import math
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class ModalityQKVLoRA(nn.Module):
    """Low-rank QKV update with independent photo and sketch parameters."""

    _MODALITIES = ("photo", "sketch")

    def __init__(self, width, rank, alpha, targets, device, seed):
        super().__init__()
        self.width = width
        self.rank = rank
        self.scale = alpha / rank
        self.targets = tuple(targets)
        self.active_modality = None

        for modality in self._MODALITIES:
            for target in self.targets:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                down_value = torch.empty(rank, width, dtype=torch.float32)
                nn.init.kaiming_uniform_(
                    down_value,
                    a=math.sqrt(5),
                    generator=generator,
                )
                down = nn.Parameter(down_value.to(device=device))
                up = nn.Parameter(
                    torch.zeros(width, rank, device=device, dtype=torch.float32)
                )
                self.register_parameter(f"{modality}_{target}_down", down)
                self.register_parameter(f"{modality}_{target}_up", up)
                seed += 1

    def forward(self, pretrained_qkv):
        if self.active_modality is None:
            return pretrained_qkv

        updates = []
        for target in "qkv":
            if target not in self.targets:
                update = torch.zeros(
                    self.width,
                    self.width,
                    device=pretrained_qkv.device,
                    dtype=pretrained_qkv.dtype,
                )
            else:
                down = getattr(
                    self,
                    f"{self.active_modality}_{target}_down",
                )
                up = getattr(
                    self,
                    f"{self.active_modality}_{target}_up",
                )
                update = (up @ down).to(dtype=pretrained_qkv.dtype)
                update = update * self.scale
            updates.append(update)

        return pretrained_qkv + torch.cat(updates, dim=0)


class TeacherLoRAController:
    """Install and switch modality-specific LoRA banks in a CLIP ViT."""

    def __init__(self, visual, rank, alpha, depth, targets, seed):
        blocks = list(visual.transformer.resblocks)
        if depth == -1:
            selected = list(enumerate(blocks))
        else:
            selected = list(enumerate(blocks[-depth:], start=len(blocks) - depth))

        self.layers = []
        for layer_index, block in selected:
            attention = block.attn
            if attention.in_proj_weight is None:
                raise RuntimeError(
                    "Teacher LoRA requires MultiheadAttention.in_proj_weight."
                )
            width = attention.embed_dim
            lora = ModalityQKVLoRA(
                width=width,
                rank=rank,
                alpha=alpha,
                targets=targets,
                device=attention.in_proj_weight.device,
                seed=seed + layer_index * 16,
            )
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                lora,
            )
            self.layers.append((layer_index, lora))

    @property
    def depth(self):
        return len(self.layers)

    def parameters(self):
        for _, layer in self.layers:
            yield from layer.parameters()

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def requires_grad_(self, requires_grad):
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)
        return self

    def state_dict(self):
        return {
            f"layer_{layer_index}.{name}": value.detach().cpu()
            for layer_index, layer in self.layers
            for name, value in layer.state_dict().items()
        }

    def set_modality(self, modality):
        if modality not in ModalityQKVLoRA._MODALITIES:
            raise ValueError(f"Unsupported teacher modality: {modality}")
        for _, layer in self.layers:
            layer.active_modality = modality

    def clear_modality(self):
        for _, layer in self.layers:
            layer.active_modality = None

    @contextmanager
    def use(self, modality):
        self.set_modality(modality)
        try:
            yield
        finally:
            self.clear_modality()


def install_teacher_lora(teacher, rank, alpha, depth, targets, seed=42):
    visual = teacher.visual
    layer_count = len(visual.transformer.resblocks)
    if depth == -1:
        effective_depth = layer_count
    elif 1 <= depth <= layer_count:
        effective_depth = depth
    else:
        raise ValueError(
            f"teacher_lora_depth must be -1 or in [1, {layer_count}], got {depth}."
        )

    return TeacherLoRAController(
        visual=visual,
        rank=rank,
        alpha=alpha,
        depth=effective_depth,
        targets=targets,
        seed=seed,
    )
