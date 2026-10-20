"""Main's independent visual prompts with existing TinyCLIP branch forward paths.
HF forward copied from student-tinyclip-vit40m32 (7b6ee8e).
Pruned forward copied from student-tinyclip-vit22m32 (648efe0).
Prompt learner copied from main (741ea27). No training or teacher needed.
"""
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


class PromptedPrunedVision(nn.Module):
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


def wrap_student(raw, kind, args, device):
    if kind == "huggingface":
        visual = PromptedHFVision(raw.model.vision_model, raw.model.visual_projection)
        # The original wrapper has non-registering references; own the visual modules here.
        visual.add_module("vision_owner", raw.model.vision_model)
        visual.add_module("projection_owner", raw.model.visual_projection)
        width, layers = visual.width, visual.layers
    elif kind == "auto_pruned":
        tower = raw.model.image_encoder_without_ddp.visual
        visual = PromptedPrunedVision(tower)
        visual.add_module("visual_owner", tower)
        width, layers = visual.width, visual.layers
    else:
        visual = raw
        width = visual.conv1.weight.shape[0]
        layers = len(visual.transformer.resblocks)
    model = PromptedStudent(visual, width, layers, args.n_ctx_visual, args.prompt_depth, args.seed)
    # Retain main's FP32 prompts with fp16/fp32 backbone, casting prompts inside visual forward.
    return model.eval().requires_grad_(False).to(device)
