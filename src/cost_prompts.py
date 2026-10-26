"""Student prompt forwards reused from the inference-speed branch (c5cea3b)."""
import torch
from torch import nn


def _random_parameter(rows, width, seed):
    if rows == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameter = torch.empty(rows, width)
    nn.init.normal_(parameter, std=0.02, generator=generator)
    return nn.Parameter(parameter)


class IndependentVisualPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx_visual,
        visual_width,
        seed,
        prompt_depth,
    ):
        super().__init__()
        self.ctx = _random_parameter(n_ctx_visual, visual_width, seed)
        self.compound_prompts = nn.ParameterList(
            [
                _random_parameter(
                    n_ctx_visual,
                    visual_width,
                    seed + layer_index,
                )
                for layer_index in range(1, prompt_depth)
                if n_ctx_visual > 0
            ]
        )

    def forward(self):
        return self.ctx, list(self.compound_prompts)


class PromptedHFVision(nn.Module):
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
            layer_output = layer(
                hidden,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            # Transformers releases differ here: older CLIPEncoderLayer
            # versions return ``(hidden_states,)``, while newer versions
            # return the tensor directly when attentions are disabled.
            hidden = (
                layer_output[0]
                if isinstance(layer_output, (tuple, list))
                else layer_output
            )
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(
                    "TinyCLIP encoder layer returned an unexpected hidden "
                    f"state shape: {getattr(hidden, 'shape', None)}."
                )

        pooled = vision_model.post_layernorm(hidden[:, 0, :])
        return self._projection(pooled)


class PromptedStudent(nn.Module):
    def __init__(self, visual, width, layers, n_ctx, depth, seed):
        super().__init__()
        if not 1 <= depth <= layers:
            raise ValueError(f"Prompt depth {depth} must be in [1, {layers}]")
        self.visual = visual
        self.photo_visual_prompt = IndependentVisualPromptLearner(n_ctx, width, seed+201, depth)
        self.sketch_visual_prompt = IndependentVisualPromptLearner(n_ctx, width, seed+202, depth)

    def forward(self, images, modality="sketch"):
        if modality not in ("sketch", "photo"):
            raise ValueError(modality)
        prompts = self.photo_visual_prompt if modality == "photo" else self.sketch_visual_prompt
        ctx, deep = prompts()
        features = self.visual(images, ctx, deep)
        return features / features.norm(dim=-1, keepdim=True)
