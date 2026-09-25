"""Frozen TinyCLIP student with modality prompts supplied by ``src.model``."""

import os
from pathlib import Path

import torch
import torch.nn as nn


TINYCLIP_MODEL_NAME = "TinyCLIP-ViT-40M-32-Text-19M"
TINYCLIP_REPOSITORY = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
TINYCLIP_REVISION = "886b932a36b8fa6c18a8e423a67ca21af5316af8"
TINYCLIP_OUTPUT_DIM = 512
TINYCLIP_IMAGE_SIZE = 224
TINYCLIP_PATCH_SIZE = 32


class PromptedTinyCLIPVision(nn.Module):
    """Run the Hugging Face TinyCLIP ViT with replace-per-layer prompts."""

    def __init__(self, vision_model, projection):
        super().__init__()
        # The complete CLIP model owns these frozen modules. Keeping only
        # non-registering references here avoids duplicate checkpoint keys.
        object.__setattr__(self, "_vision_model", vision_model)
        object.__setattr__(self, "_projection", projection)
        self.width = int(vision_model.config.hidden_size)
        self.layers = len(vision_model.encoder.layers)

    def forward(self, images, prompt=None, compound_prompts=None):
        vision_model = self._vision_model
        hidden = vision_model.embeddings(images)
        prompt_length = 0
        if prompt is not None:
            if prompt.ndim != 2 or prompt.shape[1] != hidden.shape[2]:
                raise ValueError(
                    f"Visual prompt must have shape [n_ctx, {hidden.shape[2]}], "
                    f"got {tuple(prompt.shape)}."
                )
            prompt_length = prompt.shape[0]
            first_prompt = prompt.to(device=hidden.device, dtype=hidden.dtype)
            first_prompt = first_prompt.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            hidden = torch.cat((hidden, first_prompt), dim=1)
        hidden = vision_model.pre_layrnorm(hidden)

        compound_prompts = compound_prompts or []
        for layer_index, layer in enumerate(vision_model.encoder.layers):
            deep_index = layer_index - 1
            if prompt_length and 0 <= deep_index < len(compound_prompts):
                deep_prompt = compound_prompts[deep_index].to(
                    device=hidden.device, dtype=hidden.dtype
                )
                expected = (prompt_length, hidden.shape[2])
                if tuple(deep_prompt.shape) != expected:
                    raise ValueError(
                        f"Deep visual prompt must have shape {expected}, "
                        f"got {tuple(deep_prompt.shape)}."
                    )
                deep_prompt = deep_prompt.unsqueeze(0).expand(
                    hidden.shape[0], -1, -1
                )
                hidden = torch.cat((hidden[:, :-prompt_length], deep_prompt), dim=1)
            hidden = layer(
                hidden,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )[0]

        pooled = vision_model.post_layernorm(hidden[:, 0, :])
        return self._projection(pooled)


class TinyCLIPStudent(nn.Module):
    """Small CLIP-compatible facade used by the existing KD model."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.visual = PromptedTinyCLIPVision(
            model.vision_model, model.visual_projection
        )
        object.__setattr__(self, "_tokenizer", tokenizer)
        self.context_length = int(model.config.text_config.max_position_embeddings)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def visual_width(self):
        return self.visual.width

    @property
    def visual_layers(self):
        return self.visual.layers

    def tokenize(self, texts):
        encoded = self._tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def encode_text(self, tokens):
        outputs = self.model.text_model(input_ids=tokens)
        return self.model.text_projection(outputs.pooler_output)

    def encode_image(self, images):
        return self.visual(images.to(dtype=self.dtype))


def _model_source():
    explicit = os.environ.get("TINYCLIP_MODEL_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"TINYCLIP_MODEL_PATH is not a directory: {path}")
        return str(path), None, True

    kaggle_copy = Path("/kaggle/working/tinyclip_student")
    if kaggle_copy.is_dir():
        return str(kaggle_copy), None, True

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    return TINYCLIP_REPOSITORY, TINYCLIP_REVISION, offline


def load_tinyclip_student(backbone):
    if backbone != TINYCLIP_MODEL_NAME:
        raise ValueError(
            f"This branch supports --backbone {TINYCLIP_MODEL_NAME!r}, "
            f"got {backbone!r}."
        )
    source, revision, local_only = _model_source()
    from transformers import AutoTokenizer, CLIPModel

    kwargs = {"local_files_only": local_only}
    if revision is not None:
        kwargs["revision"] = revision
    model = CLIPModel.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    vision = model.config.vision_config
    actual = {
        "image_size": int(vision.image_size),
        "patch_size": int(vision.patch_size),
        "hidden_size": int(vision.hidden_size),
        "layers": int(vision.num_hidden_layers),
        "projection_dim": int(model.config.projection_dim),
    }
    expected = {
        "image_size": TINYCLIP_IMAGE_SIZE,
        "patch_size": TINYCLIP_PATCH_SIZE,
        "hidden_size": 512,
        "layers": 12,
        "projection_dim": TINYCLIP_OUTPUT_DIM,
    }
    if actual != expected:
        raise RuntimeError(
            f"Unexpected TinyCLIP architecture: {actual}; expected {expected}."
        )
    model = model.to(dtype=torch.float16).eval()
    return TinyCLIPStudent(model, tokenizer)
