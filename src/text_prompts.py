import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def _normal_parameter(shape, std, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.empty(shape, dtype=torch.float32)
    nn.init.normal_(value, std=std, generator=generator)
    return nn.Parameter(value)


class MultiAspectPromptGenerator(nn.Module):
    """Turn a variable-length set of image patches into text contexts."""

    def __init__(
        self,
        visual_width,
        text_width,
        latent_width,
        aspects,
        context_tokens,
        heads,
        dropout,
        gate_init,
        seed,
        initial_context,
    ):
        super().__init__()
        if min(visual_width, text_width, latent_width) < 1:
            raise ValueError("Prompt widths must be positive.")
        if aspects < 1 or context_tokens < 1:
            raise ValueError("Aspects and context tokens must be positive.")
        if heads < 1 or latent_width % heads:
            raise ValueError("latent_width must be divisible by heads.")
        if not 0 <= dropout < 1:
            raise ValueError("Prompt dropout must be in [0, 1).")
        if gate_init <= 0:
            raise ValueError("Prompt gate initialization must be positive.")
        if initial_context.shape != (context_tokens, text_width):
            raise ValueError(
                "initial_context must have shape "
                f"[{context_tokens}, {text_width}]."
            )

        self.visual_width = visual_width
        self.text_width = text_width
        self.latent_width = latent_width
        self.aspects = aspects
        self.context_tokens = context_tokens

        self.patch_norm = nn.LayerNorm(visual_width)
        self.patch_projection = nn.Linear(visual_width, latent_width)
        self.queries = _normal_parameter(
            (aspects, latent_width), std=0.02, seed=seed
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.aspect_norm = nn.LayerNorm(latent_width)
        self.aspect_mlp = nn.Sequential(
            nn.Linear(latent_width, latent_width * 2),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(latent_width * 2, latent_width),
        )
        self.context_projection = nn.Linear(
            latent_width, context_tokens * text_width
        )
        self.base_context = nn.Parameter(
            initial_context.float().unsqueeze(0).repeat(aspects, 1, 1)
        )
        self.gates = nn.Parameter(
            torch.full((aspects, 1, 1), float(gate_init))
        )

        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        nn.init.normal_(
            self.patch_projection.weight,
            std=visual_width**-0.5,
            generator=generator,
        )
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.normal_(
            self.context_projection.weight,
            std=0.02,
            generator=generator,
        )
        nn.init.zeros_(self.context_projection.bias)
        with torch.no_grad():
            # Break aspect symmetry without moving far from the CLIP phrase.
            noise = torch.empty_like(self.base_context, device="cpu")
            nn.init.normal_(noise, std=1e-4, generator=generator)
            self.base_context.add_(noise.to(self.base_context.device))

    def forward(self, patch_features):
        if patch_features.ndim != 3:
            raise ValueError("Patch features must have shape [B, N, D].")
        if patch_features.shape[-1] != self.visual_width:
            raise ValueError(
                f"Expected patch width {self.visual_width}, got "
                f"{patch_features.shape[-1]}."
            )

        # Patch inputs are a frozen condition; the learned text-side module is
        # deliberately the only recipient of this objective's gradients.
        patches = self.patch_projection(
            self.patch_norm(patch_features.detach().float())
        )
        queries = self.queries.unsqueeze(0).expand(len(patches), -1, -1)
        attended, attention = self.cross_attention(
            queries,
            patches,
            patches,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended + self.aspect_mlp(self.aspect_norm(attended))
        residual = self.context_projection(attended).reshape(
            len(patches),
            self.aspects,
            self.context_tokens,
            self.text_width,
        )
        contexts = self.base_context.unsqueeze(0) + self.gates.unsqueeze(0) * residual
        return contexts, attention

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def attention_diversity_loss(attention):
    """Penalize different aspect queries attending to the same patch pattern."""
    if attention.ndim != 3:
        raise ValueError("Attention must have shape [B, R, N].")
    if attention.shape[1] == 1:
        return attention.new_zeros((), dtype=torch.float32)
    normalized = F.normalize(attention.float(), dim=-1)
    gram = normalized @ normalized.transpose(-1, -2)
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).unsqueeze(0)
    off_diagonal = gram * (1.0 - identity)
    return off_diagonal.square().sum() / (
        gram.shape[0] * gram.shape[1] * (gram.shape[1] - 1)
    )


def multi_aspect_similarity(query_features, gallery_aspects, aspect_temperature):
    """Smoothly aggregate query similarity across a photo's text aspects."""
    if aspect_temperature <= 0:
        raise ValueError("Aspect temperature must be greater than zero.")
    if query_features.ndim != 2 or gallery_aspects.ndim != 3:
        raise ValueError(
            "Expected query [B, D] and gallery aspects [G, R, D]."
        )
    queries = F.normalize(query_features.float(), dim=-1)
    aspects = F.normalize(gallery_aspects.float(), dim=-1)
    similarities = torch.einsum("bd,grd->bgr", queries, aspects)
    # Subtract log(R) so duplicating an identical aspect does not shift scores.
    return aspect_temperature * (
        torch.logsumexp(similarities / aspect_temperature, dim=-1)
        - math.log(gallery_aspects.shape[1])
    )


def multi_aspect_infonce_loss(
    query_features,
    gallery_aspects,
    targets,
    instance_temperature,
    aspect_temperature,
):
    if instance_temperature <= 0:
        raise ValueError("Instance temperature must be greater than zero.")
    if gallery_aspects.shape[0] != 100:
        raise RuntimeError("Fine-grained text training expects 100 photos.")
    logits = multi_aspect_similarity(
        query_features, gallery_aspects, aspect_temperature
    ) / instance_temperature
    return F.cross_entropy(logits, targets.long()), logits


def relational_logits_kd_loss(student_logits, teacher_logits, temperature):
    if temperature <= 0:
        raise ValueError("KD temperature must be greater than zero.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape.")
    teacher_probability = F.softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    student_log_probability = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    return F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * temperature**2


def _replace_context_embeddings(token_embeddings, contexts, context_tokens):
    if token_embeddings.shape[1] <= context_tokens + 1:
        raise ValueError("Token sequence is too short for the soft context.")
    return torch.cat(
        (
            token_embeddings[:, :1],
            contexts,
            token_embeddings[:, 1 + context_tokens :],
        ),
        dim=1,
    )


def encode_openai_soft_prompts(model, token_ids, contexts):
    """Encode soft contexts with the vendored OpenAI CLIP text tower."""
    context_tokens = contexts.shape[1]
    x = model.token_embedding(token_ids).type(model.dtype)
    x = _replace_context_embeddings(
        x, contexts.to(device=x.device, dtype=x.dtype), context_tokens
    )
    x = x + model.positional_embedding.type(x.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    eot_indices = token_ids.argmax(dim=-1)
    return x[torch.arange(len(x), device=x.device), eot_indices] @ model.text_projection


def encode_openclip_soft_prompts(model, token_ids, contexts):
    """Encode soft contexts with an OpenCLIP text tower."""
    from open_clip.transformer import text_global_pool

    context_tokens = contexts.shape[1]
    cast_dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(token_ids).to(cast_dtype)
    x = _replace_context_embeddings(
        x, contexts.to(device=x.device, dtype=x.dtype), context_tokens
    )
    x = x + model.positional_embedding.to(cast_dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    x = text_global_pool(
        x,
        token_ids,
        model.text_pool_type,
        eos_token_id=getattr(model, "text_eos_id", None),
    )
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection
    return x


class MultiAspectPhotoTextPrompts(nn.Module):
    """Photo-conditioned prompt bank for seen and unseen class names."""

    def __init__(
        self,
        text_model,
        tokenizer,
        seen_classnames,
        unseen_classnames,
        visual_width,
        latent_width=512,
        aspects=4,
        context_tokens=4,
        heads=8,
        dropout=0.1,
        gate_init=0.1,
        seed=42,
        openclip=False,
        encode_chunk_size=100,
    ):
        super().__init__()
        if encode_chunk_size < 1:
            raise ValueError("encode_chunk_size must be positive.")
        self.aspects = aspects
        self.context_tokens = context_tokens
        self.openclip = openclip
        self.encode_chunk_size = encode_chunk_size
        text_width = text_model.token_embedding.weight.shape[1]

        initial_context = self._initial_context(
            text_model, tokenizer, context_tokens, text_width
        )
        self.generator = MultiAspectPromptGenerator(
            visual_width=visual_width,
            text_width=text_width,
            latent_width=latent_width,
            aspects=aspects,
            context_tokens=context_tokens,
            heads=heads,
            dropout=dropout,
            gate_init=gate_init,
            seed=seed,
            initial_context=initial_context,
        )
        self.register_buffer(
            "seen_tokens",
            self._class_tokens(tokenizer, seen_classnames, context_tokens),
            persistent=False,
        )
        self.register_buffer(
            "unseen_tokens",
            self._class_tokens(tokenizer, unseen_classnames, context_tokens),
            persistent=False,
        )

    @staticmethod
    def _class_tokens(tokenizer, classnames, context_tokens):
        placeholders = " ".join(["X"] * context_tokens)
        prompts = [
            f"{placeholders} {name.replace('_', ' ')}."
            for name in classnames
        ]
        tokens = tokenizer(prompts)
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        return tokens.long()

    @staticmethod
    def _initial_context(text_model, tokenizer, context_tokens, text_width):
        phrase = "a photo of a"
        tokens = tokenizer([phrase])
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        device = text_model.token_embedding.weight.device
        with torch.no_grad():
            embeddings = text_model.token_embedding(tokens.to(device))[0, 1:]
            # Ignore padding and EOT. The standard M=4 case exactly uses the
            # four pretrained word embeddings from "a photo of a".
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
                nn.init.normal_(initial[copied:], std=0.02)
        return initial.float().cpu()

    def _tokens_for(self, categories, split):
        if split not in {"seen", "unseen"}:
            raise ValueError("split must be 'seen' or 'unseen'.")
        token_bank = self.seen_tokens if split == "seen" else self.unseen_tokens
        return token_bank[categories.long().to(token_bank.device)]

    def forward(self, text_model, patch_features, categories, split="seen"):
        contexts, attention = self.generator(patch_features)
        batch_size, aspects, context_tokens, text_width = contexts.shape
        tokens = self._tokens_for(categories, split).to(patch_features.device)
        tokens = tokens[:, None, :].expand(-1, aspects, -1).reshape(
            batch_size * aspects, -1
        )
        contexts = contexts.reshape(
            batch_size * aspects, context_tokens, text_width
        )
        encode = (
            encode_openclip_soft_prompts
            if self.openclip
            else encode_openai_soft_prompts
        )
        features = torch.cat(
            [
                encode(
                    text_model,
                    tokens[start : start + self.encode_chunk_size],
                    contexts[start : start + self.encode_chunk_size],
                )
                for start in range(0, len(tokens), self.encode_chunk_size)
            ],
            dim=0,
        )
        features = F.normalize(features.float(), dim=-1)
        return features.reshape(batch_size, aspects, -1), attention

    def trainable_parameter_count(self):
        return self.generator.trainable_parameter_count()
