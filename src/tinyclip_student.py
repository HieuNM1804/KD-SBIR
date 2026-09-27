"""Frozen auto-pruned TinyCLIP ViT-22M/32 student with visual prompts."""

import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn


TINYCLIP_MODEL_NAME = "TinyCLIP-ViT-22M-32-Text-10M"
TINYCLIP_CHECKPOINT_FILENAME = (
    "TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt"
)
TINYCLIP_CHECKPOINT_URL = (
    "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/"
    f"checkpoints/{TINYCLIP_CHECKPOINT_FILENAME}"
)
TINYCLIP_OUTPUT_DIM = 512
TINYCLIP_IMAGE_SIZE = 224
TINYCLIP_PATCH_SIZE = 32
TINYCLIP_BASE_VISION_WIDTH = 768
TINYCLIP_BASE_TEXT_WIDTH = 512
TINYCLIP_LAYERS = 12


class PromptedTinyCLIPVision(nn.Module):
    """Run a physically pruned TinyCLIP vision tower with deep prompts."""

    def __init__(self, visual):
        super().__init__()
        # ``visual`` remains owned by the complete TinyCLIP model. A plain
        # reference prevents duplicate parameters and checkpoint keys here.
        object.__setattr__(self, "_visual", visual)
        # TinyCLIP prunes ``conv1.weight`` in place; PyTorch's cached
        # ``out_channels`` attribute still contains the unpruned width.
        self.width = int(visual.conv1.weight.shape[0])
        self.layers = len(visual.transformer.resblocks)

    def forward(self, images, prompt=None, compound_prompts=None):
        visual = self._visual
        hidden = visual.conv1(images.to(visual.conv1.weight.device))
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = hidden.permute(0, 2, 1)
        class_token = visual.class_embedding.to(hidden.dtype)
        class_token = class_token + torch.zeros(
            hidden.shape[0],
            1,
            hidden.shape[-1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        hidden = torch.cat((class_token, hidden), dim=1)
        hidden = hidden + visual.positional_embedding.to(hidden.dtype)

        prompt_length = 0
        if prompt is not None:
            expected_width = hidden.shape[2]
            if prompt.ndim != 2 or prompt.shape[1] != expected_width:
                raise ValueError(
                    "Visual prompt must have shape "
                    f"[n_ctx, {expected_width}], got {tuple(prompt.shape)}."
                )
            prompt_length = int(prompt.shape[0])
            first_prompt = prompt.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            first_prompt = first_prompt.unsqueeze(0).expand(
                hidden.shape[0], -1, -1
            )
            hidden = torch.cat((hidden, first_prompt), dim=1)

        hidden = visual.ln_pre(hidden)
        hidden = hidden.permute(1, 0, 2)
        compound_prompts = compound_prompts or []
        for layer_index, layer in enumerate(visual.transformer.resblocks):
            deep_index = layer_index - 1
            if prompt_length and 0 <= deep_index < len(compound_prompts):
                deep_prompt = compound_prompts[deep_index].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                expected = (prompt_length, hidden.shape[2])
                if tuple(deep_prompt.shape) != expected:
                    raise ValueError(
                        f"Deep visual prompt must have shape {expected}, "
                        f"got {tuple(deep_prompt.shape)}."
                    )
                deep_prompt = deep_prompt.unsqueeze(1).expand(
                    -1, hidden.shape[1], -1
                )
                hidden = torch.cat(
                    (hidden[:-prompt_length], deep_prompt), dim=0
                )
            hidden = layer(hidden)

        hidden = hidden.permute(1, 0, 2)
        pooled = visual.ln_post(hidden[:, 0, :])
        return pooled @ visual.proj


class TinyCLIPStudent(nn.Module):
    """CLIP-compatible facade used by the existing KD implementation."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.visual = PromptedTinyCLIPVision(model.visual)
        object.__setattr__(self, "_tokenizer", tokenizer)
        self.context_length = int(
            model.text_encoder_without_ddp.context_length
        )

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    @property
    def visual_width(self):
        return self.visual.width

    @property
    def visual_layers(self):
        return self.visual.layers

    def tokenize(self, texts):
        return self._tokenizer(list(texts), context_length=self.context_length)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def encode_image(self, images):
        return self.visual(images.to(dtype=self.dtype))


def _checkpoint_path():
    explicit = os.environ.get("TINYCLIP_MODEL_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            path = path / TINYCLIP_CHECKPOINT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"TINYCLIP_MODEL_PATH does not resolve to a file: {path}"
            )
        return path

    kaggle_copy = (
        Path("/kaggle/working/tinyclip22m_student")
        / TINYCLIP_CHECKPOINT_FILENAME
    )
    if kaggle_copy.is_file():
        return kaggle_copy
    raise FileNotFoundError(
        "TinyCLIP ViT-22M/32 checkpoint was not found. Set "
        "TINYCLIP_MODEL_PATH to the official auto-pruned checkpoint."
    )


def _load_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("TinyCLIP checkpoint has no usable state_dict.")
    return state


def load_tinyclip_student(backbone):
    if backbone != TINYCLIP_MODEL_NAME:
        raise ValueError(
            f"This branch supports --backbone {TINYCLIP_MODEL_NAME!r}, "
            f"got {backbone!r}."
        )

    from src.tinyclip_vendor.model import (
        CLIP,
        load_pruned_model,
        prune_model,
    )
    from src.tinyclip_vendor.tokenizer import tokenize

    model = CLIP(
        embed_dim=TINYCLIP_OUTPUT_DIM,
        vision_cfg={
            "image_size": TINYCLIP_IMAGE_SIZE,
            "layers": TINYCLIP_LAYERS,
            "width": TINYCLIP_BASE_VISION_WIDTH,
            "patch_size": TINYCLIP_PATCH_SIZE,
        },
        text_cfg={
            "context_length": 77,
            "vocab_size": 49408,
            "width": TINYCLIP_BASE_TEXT_WIDTH,
            "heads": 8,
            "layers": TINYCLIP_LAYERS,
        },
        mask_image=True,
        mask_text=True,
    )
    load_pruned_model(model, _load_checkpoint(_checkpoint_path()))
    model = prune_model(model).eval()

    actual = {
        "vision_width": int(model.visual.conv1.weight.shape[0]),
        "vision_layers": len(model.visual.transformer.resblocks),
        "vision_patch_size": int(model.visual.conv1.kernel_size[0]),
        "text_width": int(
            model.text_encoder_without_ddp.token_embedding.embedding_dim
        ),
        "projection_dim": int(model.visual.proj.shape[1]),
    }
    expected = {
        "vision_width": 370,
        "vision_layers": 12,
        "vision_patch_size": 32,
        "text_width": 509,
        "projection_dim": 512,
    }
    if actual != expected:
        raise RuntimeError(
            f"Unexpected TinyCLIP ViT-22M/32 architecture: {actual}; "
            f"expected {expected}."
        )

    if torch.cuda.is_available():
        model = model.to(dtype=torch.float16)
    return TinyCLIPStudent(model, tokenize)
