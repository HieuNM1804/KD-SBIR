"""Frozen loaders for the five official TinyCLIP checkpoints.

This module deliberately exposes image encoders only. It creates no optimizer,
prompt parameter, loss, or training state.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TinyCLIPSpec:
    key: str
    name: str
    kind: str
    directory: str
    repository: str | None
    revision: str | None
    filename: str | None
    image_size: int
    patch_size: int
    vision_width: int
    vision_layers: int
    text_width: int
    text_layers: int
    projection_dim: int = 512


MODEL_SPECS = {
    "8m": TinyCLIPSpec(
        "8m", "TinyCLIP-ViT-8M-16-Text-3M", "huggingface",
        "tinyclip8m_student", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c", None,
        224, 16, 256, 10, 256, 3,
    ),
    "22m": TinyCLIPSpec(
        "22m", "TinyCLIP-ViT-22M-32-Text-10M", "auto_pruned",
        "tinyclip22m_student", None, None,
        "TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt",
        224, 32, 370, 12, 509, 12,
    ),
    "40m": TinyCLIPSpec(
        "40m", "TinyCLIP-ViT-40M-32-Text-19M", "huggingface",
        "tinyclip40m_student", "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "886b932a36b8fa6c18a8e423a67ca21af5316af8", None,
        224, 32, 512, 12, 512, 6,
    ),
    "45m": TinyCLIPSpec(
        "45m", "TinyCLIP-ViT-45M-32-Text-18M", "auto_pruned",
        "tinyclip45m_student", None, None,
        "TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt",
        224, 32, 549, 12, 510, 12,
    ),
    "61m": TinyCLIPSpec(
        "61m", "TinyCLIP-ViT-61M-32-Text-29M", "huggingface",
        "tinyclip61m_student", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
        "94cbbea5c7949cfe7bdafde64bcea5e403f59852", None,
        224, 32, 640, 12, 512, 9,
    ),
}


class FrozenImageEncoder(nn.Module):
    """Uniform image-only facade for HF and auto-pruned TinyCLIP models."""

    def __init__(self, model, spec, encoder_kind):
        super().__init__()
        self.model = model
        self.spec = spec
        self.encoder_kind = encoder_kind

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def forward(self, images):
        images = images.to(dtype=self.dtype)
        if self.encoder_kind == "huggingface":
            outputs = self.model.vision_model(pixel_values=images)
            return self.model.visual_projection(outputs.pooler_output)
        return self.model.encode_image(images, normalized=False)

    def vision_parameter_count(self):
        if self.encoder_kind == "huggingface":
            modules = (self.model.vision_model, self.model.visual_projection)
            return sum(p.numel() for module in modules for p in module.parameters())
        return sum(
            p.numel() for p in self.model.image_encoder_without_ddp.parameters()
        )


def _model_location(spec, models_root):
    if models_root is None:
        if spec.kind == "huggingface":
            return spec.repository
        raise FileNotFoundError(
            f"{spec.name} uses an official .pt checkpoint; pass --models-root."
        )
    directory = Path(models_root) / spec.directory
    if spec.kind == "huggingface":
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing TinyCLIP config: {directory / 'config.json'}")
        return str(directory)
    checkpoint = directory / spec.filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing TinyCLIP checkpoint: {checkpoint}")
    return checkpoint


def _load_state_dict(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no usable state_dict: {path}")
    return state


def _load_huggingface(spec, location, local_only):
    from transformers import CLIPModel

    kwargs = {"local_files_only": local_only}
    if not local_only:
        kwargs["revision"] = spec.revision
    model = CLIPModel.from_pretrained(location, **kwargs)
    vision = model.config.vision_config
    text = model.config.text_config
    actual = {
        "image_size": int(vision.image_size),
        "patch_size": int(vision.patch_size),
        "vision_width": int(vision.hidden_size),
        "vision_layers": int(vision.num_hidden_layers),
        "text_width": int(text.hidden_size),
        "text_layers": int(text.num_hidden_layers),
        "projection_dim": int(model.config.projection_dim),
    }
    expected = {
        "image_size": spec.image_size,
        "patch_size": spec.patch_size,
        "vision_width": spec.vision_width,
        "vision_layers": spec.vision_layers,
        "text_width": spec.text_width,
        "text_layers": spec.text_layers,
        "projection_dim": spec.projection_dim,
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected {spec.name} architecture: {actual}; expected {expected}")
    return model


def _load_auto_pruned(spec, checkpoint):
    from src.tinyclip_vendor.model import CLIP, load_pruned_model, prune_model

    model = CLIP(
        embed_dim=spec.projection_dim,
        vision_cfg={"image_size": spec.image_size, "layers": 12, "width": 768,
                    "patch_size": spec.patch_size},
        text_cfg={"context_length": 77, "vocab_size": 49408, "width": 512,
                  "heads": 8, "layers": 12},
        mask_image=True,
        mask_text=True,
    )
    load_pruned_model(model, _load_state_dict(checkpoint))
    model = prune_model(model)
    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    actual = {
        "image_size": int(image_size),
        "patch_size": int(model.visual.conv1.kernel_size[0]),
        "vision_width": int(model.visual.conv1.weight.shape[0]),
        "vision_layers": len(model.visual.transformer.resblocks),
        "text_width": int(model.text_encoder_without_ddp.token_embedding.embedding_dim),
        "text_layers": len(model.text_encoder_without_ddp.transformer.resblocks),
        "projection_dim": int(model.visual.proj.shape[1]),
    }
    expected = {
        "image_size": spec.image_size,
        "patch_size": spec.patch_size,
        "vision_width": spec.vision_width,
        "vision_layers": spec.vision_layers,
        "text_width": spec.text_width,
        "text_layers": spec.text_layers,
        "projection_dim": spec.projection_dim,
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected {spec.name} architecture: {actual}; expected {expected}")
    return model


def load_frozen_image_encoder(key, models_root, device, precision="auto"):
    """Load one official checkpoint with every parameter frozen."""

    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown TinyCLIP key {key!r}; choose from {tuple(MODEL_SPECS)}")
    spec = MODEL_SPECS[key]
    location = _model_location(spec, models_root)
    model = (
        _load_huggingface(spec, location, local_only=models_root is not None)
        if spec.kind == "huggingface"
        else _load_auto_pruned(spec, location)
    )
    if precision == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 inference requires CUDA; use --precision fp32 on CPU.")
        dtype = torch.float16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    model.requires_grad_(False).eval().to(device=device, dtype=dtype)
    encoder = FrozenImageEncoder(model, spec, spec.kind).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Inference model unexpectedly contains trainable parameters.")
    return encoder
