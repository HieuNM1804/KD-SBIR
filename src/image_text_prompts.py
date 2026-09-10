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


def deterministic_adaptive_average_tokens(features, output_tokens):
    """Adaptive-average token sequences with deterministic GEMM backward."""
    if features.ndim != 3:
        raise ValueError("Features must have shape [B, N, D].")
    patch_count = features.shape[1]
    if not 1 <= output_tokens <= patch_count:
        raise ValueError("output_tokens must be in [1, patch_count].")

    # These are the same start/end bins used by adaptive average pooling.
    # Expressing the pooling as a fixed linear map avoids CUDA's
    # adaptive_avg_pool2d backward kernel, which has no deterministic mode.
    pooling = features.new_zeros(output_tokens, patch_count)
    for output_index in range(output_tokens):
        start = output_index * patch_count // output_tokens
        end = (
            (output_index + 1) * patch_count + output_tokens - 1
        ) // output_tokens
        pooling[output_index, start:end] = 1.0 / (end - start)
    return F.linear(features.transpose(1, 2), pooling).transpose(1, 2)


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
            raise ValueError(
                "Patch/text widths and context_tokens must be positive."
            )
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
        pooled = deterministic_adaptive_average_tokens(
            projected,
            self.context_tokens,
        )
        residual = self.context_norm(pooled)
        return base_context.unsqueeze(0) + self.gate * residual


class PartQueryPatchToTextContexts(nn.Module):
    """Extract unordered semantic parts with learned queries over all patches.

    Unlike fixed spatial averaging, every context token has its own learned
    query and can follow the same semantic part across sketch/photo layout
    changes. Keys and queries are cosine-normalized; a learned bounded scale
    controls attention sharpness without relying on a CUDA attention kernel.
    """

    def __init__(
        self,
        visual_width,
        text_width,
        context_tokens,
        seed,
        gate_init=0.1,
        attention_temperature=0.07,
    ):
        super().__init__()
        if min(visual_width, text_width, context_tokens) < 1:
            raise ValueError("Patch/text widths and context_tokens must be positive.")
        if gate_init <= 0:
            raise ValueError("gate_init must be positive.")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")

        self.visual_width = visual_width
        self.context_tokens = context_tokens
        self.patch_norm = nn.LayerNorm(visual_width)
        self.key_projection = nn.Linear(visual_width, text_width)
        self.value_projection = nn.Linear(visual_width, text_width)
        self.query_norm = nn.LayerNorm(text_width)
        self.context_norm = nn.LayerNorm(text_width)
        self.part_queries = nn.Parameter(
            torch.empty(context_tokens, text_width)
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(float(1.0 / attention_temperature)).log()
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        for projection in (self.key_projection, self.value_projection):
            nn.init.normal_(
                projection.weight,
                std=visual_width**-0.5,
                generator=generator,
            )
            nn.init.zeros_(projection.bias)
        nn.init.normal_(
            self.part_queries,
            std=text_width**-0.5,
            generator=generator,
        )

    def forward(self, patch_features, base_context, return_attention=False):
        if patch_features.ndim != 3:
            raise ValueError("Patch features must have shape [B, N, D].")
        if patch_features.shape[-1] != self.visual_width:
            raise ValueError(
                f"Expected patch width {self.visual_width}, got "
                f"{patch_features.shape[-1]}."
            )
        if base_context.shape != (
            self.context_tokens,
            self.part_queries.shape[-1],
        ):
            raise ValueError("base_context has an incompatible shape.")

        patches = self.patch_norm(patch_features.float())
        keys = F.normalize(self.key_projection(patches), dim=-1)
        values = self.value_projection(patches)
        queries = F.normalize(
            self.query_norm(self.part_queries.float()), dim=-1
        )
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("md,bnd->bmn", queries, keys) * scale
        attention = logits.softmax(dim=-1)
        parts = torch.einsum("bmn,bnd->bmd", attention, values)
        contexts = base_context.unsqueeze(0) + self.gate * (
            self.context_norm(parts)
        )
        if return_attention:
            return contexts, attention
        return contexts


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
        context_generator="part_query",
        part_attention_temperature=0.07,
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
        self.context_generator_name = context_generator
        text_width = text_model.token_embedding.weight.shape[1]

        initial_context = self._initial_context(
            text_model,
            tokenizer,
            context_tokens,
            text_width,
            seed,
        )
        self.base_context = nn.Parameter(initial_context)
        if context_generator == "adaptive_average":
            generator_class = PatchToTextContexts
            generator_kwargs = {}
        elif context_generator == "part_query":
            generator_class = PartQueryPatchToTextContexts
            generator_kwargs = {
                "attention_temperature": part_attention_temperature,
            }
        else:
            raise ValueError(
                "context_generator must be adaptive_average or part_query."
            )
        self.context_generator = generator_class(
            visual_width=visual_width,
            text_width=text_width,
            context_tokens=context_tokens,
            seed=seed + 1,
            gate_init=gate_init,
            **generator_kwargs,
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

    def forward(
        self,
        text_model,
        patch_features,
        class_labels,
        modality,
        return_attention=False,
    ):
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

        if return_attention:
            if self.context_generator_name != "part_query":
                raise RuntimeError(
                    "Attention maps require the part_query context generator."
                )
            contexts, attention = self.context_generator(
                patch_features,
                self.base_context,
                return_attention=True,
            )
        else:
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
        output = (F.normalize(features.float(), dim=-1), contexts)
        if return_attention:
            return (*output, attention)
        return output

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
