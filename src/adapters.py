import torch
import torch.nn as nn
from torch.nn import functional as F


def _normal_parameter(shape, std, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.empty(*shape, dtype=torch.float32)
    nn.init.normal_(value, std=std, generator=generator)
    return nn.Parameter(value.to(device=device))


def _zero_parameter(shape, device):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=device))


class BottleneckAdapter(nn.Module):
    """Residual adapter whose zero-initialized expansion starts as identity."""

    def __init__(
        self,
        width,
        bottleneck,
        std,
        seed,
        dropout=0.0,
        scale=1.0,
        device=None,
    ):
        super().__init__()
        if width < 1:
            raise ValueError("Adapter width must be positive.")
        if bottleneck < 1:
            raise ValueError("Adapter bottleneck must be positive.")
        if std <= 0:
            raise ValueError("Adapter initialization std must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("Adapter dropout must be in [0, 1).")
        if scale <= 0:
            raise ValueError("Adapter scale must be positive.")

        self.width = width
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.scale = scale
        self.down_weight = _normal_parameter(
            (bottleneck, width), std, seed, device
        )
        self.down_bias = _zero_parameter((bottleneck,), device)
        self.up_weight = _zero_parameter((width, bottleneck), device)
        self.up_bias = _zero_parameter((width,), device)

    @staticmethod
    def _cast(parameter, reference):
        return parameter.to(device=reference.device, dtype=reference.dtype)

    def forward(self, x):
        # Parameter-free normalization prevents scale drift without introducing
        # another set of affine LayerNorm parameters.
        residual = F.layer_norm(x, (self.width,))
        residual = F.linear(
            residual,
            self._cast(self.down_weight, x),
            self._cast(self.down_bias, x),
        )
        residual = F.gelu(residual, approximate="tanh")
        residual = F.dropout(residual, p=self.dropout, training=self.training)
        residual = F.linear(
            residual,
            self._cast(self.up_weight, x),
            self._cast(self.up_bias, x),
        )
        return x + self.scale * residual


class ModalityOutputAdapters(nn.Module):
    """One independent output adapter for each image modality."""

    _MODALITIES = ("photo", "sketch")

    def __init__(
        self,
        width,
        bottleneck,
        std,
        seed,
        dropout=0.0,
        scale=1.0,
        device=None,
    ):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for modality_index, modality in enumerate(self._MODALITIES):
            self.adapters[modality] = BottleneckAdapter(
                width=width,
                bottleneck=bottleneck,
                std=std,
                seed=seed + modality_index,
                dropout=dropout,
                scale=scale,
                device=device,
            )

    def for_modality(self, modality):
        if modality not in self._MODALITIES:
            raise ValueError(f"Unsupported adapter modality: {modality}")
        return self.adapters[modality]

    def forward(self, x, modality):
        return self.for_modality(modality)(x)

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
