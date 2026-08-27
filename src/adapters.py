import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """A simple down-project, activate, up-project residual adapter."""

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
        self.scale = scale
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, width)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.normal_(self.down.weight, std=std, generator=generator)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if device is not None:
            self.to(device=device)

    def forward(self, x):
        # Frozen CLIP/DFN encoders may emit FP16; train the small adapter in FP32.
        x = x.float()
        residual = self.up(
            self.dropout(self.activation(self.down(self.norm(x))))
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
