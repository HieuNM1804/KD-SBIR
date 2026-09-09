"""Image-conditioned soft text prompts derived from visual patch tokens."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _replace_context_embeddings(token_embeddings, contexts):
    context_count = contexts.shape[1]
    if token_embeddings.shape[1] <= context_count + 1:
        raise ValueError("Token sequence is too short for the soft context.")
    return torch.cat(
        (
            token_embeddings[:, :1],
            contexts,
            token_embeddings[:, 1 + context_count :],
        ),
        dim=1,
    )


def encode_openai_soft_prompt(model, token_ids, contexts):
    """Encode continuous prompt tokens with the vendored OpenAI CLIP tower."""
    x = model.token_embedding(token_ids).type(model.dtype)
    x = _replace_context_embeddings(
        x,
        contexts.to(device=x.device, dtype=x.dtype),
    )
    x = x + model.positional_embedding.type(x.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    eot_indices = token_ids.argmax(dim=-1)
    return x[torch.arange(len(x), device=x.device), eot_indices] @ (
        model.text_projection
    )


def encode_openclip_soft_prompt(model, token_ids, contexts):
    """Encode continuous prompt tokens with an OpenCLIP text tower."""
    cast_dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(token_ids).to(cast_dtype)
    x = _replace_context_embeddings(
        x,
        contexts.to(device=x.device, dtype=x.dtype),
    )
    x = x + model.positional_embedding.to(x.dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    eot_indices = token_ids.argmax(dim=-1)
    x = x[torch.arange(len(x), device=x.device), eot_indices]
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection
    return x


class PatchToTextContexts(nn.Module):
    """Project all image patches into a fixed number of soft text tokens."""

    def __init__(
        self,
        visual_width,
        text_width,
        context_tokens,
        seed,
        gate_init=0.1,
    ):
        super().__init__()
        if min(visual_width, text_width, context_tokens) < 1:
            raise ValueError("Patch/text widths and context_tokens must be positive.")
        if gate_init <= 0:
            raise ValueError("gate_init must be positive.")

        self.visual_width = visual_width
        self.context_tokens = context_tokens
        self.patch_norm = nn.LayerNorm(visual_width)
        self.patch_projection = nn.Linear(visual_width, text_width)
        self.context_norm = nn.LayerNorm(text_width)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.normal_(
            self.patch_projection.weight,
            std=visual_width**-0.5,
            generator=generator,
        )
        nn.init.zeros_(self.patch_projection.bias)

    def forward(self, patch_features, base_context):
        if patch_features.ndim != 3:
            raise ValueError("Patch features must have shape [B, N, D].")
        if patch_features.shape[-1] != self.visual_width:
            raise ValueError(
                f"Expected patch width {self.visual_width}, got "
                f"{patch_features.shape[-1]}."
            )
        if patch_features.shape[1] < self.context_tokens:
            raise ValueError(
                "context_tokens cannot exceed the number of image patches."
            )

        # This is the requested visual-to-text initialization path:
        # last-layer patch tokens -> learned projection -> M spatial pools.
        projected = self.patch_projection(
            self.patch_norm(patch_features.float())
        )
        pooled = F.adaptive_avg_pool1d(
            projected.transpose(1, 2),
            self.context_tokens,
        ).transpose(1, 2)
        residual = self.context_norm(pooled)
        return base_context.unsqueeze(0) + self.gate * residual


class ImageConditionedTextPromptLearner(nn.Module):
    """Generate true-class text features from the current image's patches."""

    _MODALITIES = ("photo", "sketch")

    def __init__(
        self,
        text_model,
        tokenizer,
        classnames,
        visual_width,
        context_tokens,
        seed,
        text_backend,
        gate_init=0.1,
        encode_chunk_size=64,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if text_backend not in {"openai", "open_clip"}:
            raise ValueError("text_backend must be 'openai' or 'open_clip'.")
        if encode_chunk_size < 1:
            raise ValueError("encode_chunk_size must be positive.")
        self.context_tokens = context_tokens
        self.text_backend = text_backend
        self.encode_chunk_size = encode_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        text_width = text_model.token_embedding.weight.shape[1]

        initial_context = self._initial_context(
            text_model,
            tokenizer,
            context_tokens,
            text_width,
            seed,
        )
        self.base_context = nn.Parameter(initial_context)
        self.context_generator = PatchToTextContexts(
            visual_width=visual_width,
            text_width=text_width,
            context_tokens=context_tokens,
            seed=seed + 1,
            gate_init=gate_init,
        )
        for modality in self._MODALITIES:
            self.register_buffer(
                f"{modality}_tokens",
                self._class_tokens(
                    tokenizer,
                    classnames,
                    context_tokens,
                    modality,
                ),
                persistent=False,
            )

    @staticmethod
    def _class_tokens(tokenizer, classnames, context_tokens, modality):
        placeholders = " ".join(["X"] * context_tokens)
        prompts = [
            (
                f"{placeholders} a {modality} of a "
                f"{name.replace('_', ' ')}."
            )
            for name in classnames
        ]
        tokens = tokenizer(prompts)
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        return tokens.long()

    @staticmethod
    def _initial_context(
        text_model,
        tokenizer,
        context_tokens,
        text_width,
        seed,
    ):
        tokens = tokenizer(["an image of a"])
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        device = text_model.token_embedding.weight.device
        with torch.no_grad():
            embeddings = text_model.token_embedding(tokens.to(device))[0, 1:]
            nonzero = tokens[0, 1:].ne(0)
            embeddings = embeddings[nonzero.to(embeddings.device)]
            if len(embeddings):
                embeddings = embeddings[:-1]
            initial = torch.empty(
                context_tokens,
                text_width,
                device=device,
                dtype=embeddings.dtype,
            )
            copied = min(context_tokens, len(embeddings))
            if copied:
                initial[:copied].copy_(embeddings[:copied])
            if copied < context_tokens:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                tail = torch.empty(
                    context_tokens - copied,
                    text_width,
                    dtype=torch.float32,
                )
                nn.init.normal_(tail, std=0.02, generator=generator)
                initial[copied:].copy_(
                    tail.to(device=device, dtype=initial.dtype)
                )
        return initial.float().cpu()

    def _encode(self, text_model, token_ids, contexts):
        encoder = (
            encode_openai_soft_prompt
            if self.text_backend == "openai"
            else encode_openclip_soft_prompt
        )

        def encode_context(current_contexts):
            return encoder(text_model, token_ids, current_contexts)

        if self.gradient_checkpointing and contexts.requires_grad:
            return checkpoint(
                encode_context,
                contexts,
                use_reentrant=False,
            )
        return encode_context(contexts)

    def forward(self, text_model, patch_features, class_labels, modality):
        if modality not in self._MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}.")
        labels = class_labels.long().reshape(-1)
        if len(labels) != len(patch_features):
            raise ValueError("Every image needs one class label.")
        token_bank = getattr(self, f"{modality}_tokens")
        if labels.numel() and (
            labels.min().item() < 0 or labels.max().item() >= len(token_bank)
        ):
            raise ValueError("A class label is outside the text token bank.")

        contexts = self.context_generator(
            patch_features,
            self.base_context,
        )
        tokens = token_bank[labels.to(token_bank.device)].to(
            patch_features.device
        )
        features = torch.cat(
            [
                self._encode(
                    text_model,
                    tokens[start : start + self.encode_chunk_size],
                    contexts[start : start + self.encode_chunk_size],
                )
                for start in range(0, len(tokens), self.encode_chunk_size)
            ],
            dim=0,
        )
        return F.normalize(features.float(), dim=-1), contexts

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
