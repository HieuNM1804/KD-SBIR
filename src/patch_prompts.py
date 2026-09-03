import torch
import torch.nn as nn
from torch.nn import functional as F


def _normal_parameter(shape, std, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.empty(shape, dtype=torch.float32)
    nn.init.normal_(value, std=std, generator=generator)
    return nn.Parameter(value)


class SharedPatchToPromptProjector(nn.Module):
    """Map photo or sketch patch tokens to one bank of soft text tokens."""

    def __init__(
        self,
        visual_width,
        text_width,
        latent_width,
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
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive.")
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
        self.context_tokens = context_tokens
        self.patch_norm = nn.LayerNorm(visual_width)
        self.patch_projection = nn.Linear(visual_width, latent_width)
        self.prompt_queries = _normal_parameter(
            (context_tokens, latent_width), std=0.02, seed=seed
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(latent_width)
        self.query_mlp = nn.Sequential(
            nn.Linear(latent_width, latent_width * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(latent_width * 4, latent_width),
        )
        self.context_projection = nn.Linear(latent_width, text_width)
        self.base_context = nn.Parameter(initial_context.float().clone())
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

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

    def forward(self, patch_features):
        if patch_features.ndim != 3:
            raise ValueError("Patch features must have shape [B, N, D].")
        if patch_features.shape[-1] != self.visual_width:
            raise ValueError(
                f"Expected patch width {self.visual_width}, got "
                f"{patch_features.shape[-1]}."
            )

        # The projector is the only trainable recipient of this objective.
        # Keeping image patches detached protects the visual baseline while
        # prompt retrieval is evaluated as an independent representation.
        patches = self.patch_projection(
            self.patch_norm(patch_features.detach().float())
        )
        queries = self.prompt_queries.unsqueeze(0).expand(
            len(patches), -1, -1
        )
        attended, attention = self.cross_attention(
            queries,
            patches,
            patches,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended + self.query_mlp(self.query_norm(attended))
        residual = self.context_projection(attended)
        contexts = self.base_context.unsqueeze(0) + self.gate * residual
        return contexts, attention

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def _replace_context_embeddings(token_embeddings, contexts):
    context_tokens = contexts.shape[1]
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
    """Encode continuous context tokens with the vendored CLIP text tower."""
    x = model.token_embedding(token_ids).type(model.dtype)
    x = _replace_context_embeddings(
        x, contexts.to(device=x.device, dtype=x.dtype)
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


class SharedImageConditionedPrompt(nn.Module):
    """One prompt learner shared by photo and sketch inputs."""

    def __init__(
        self,
        text_model,
        tokenizer,
        seen_classnames,
        unseen_classnames,
        visual_width,
        latent_width=256,
        context_tokens=8,
        heads=8,
        dropout=0.1,
        gate_init=0.1,
        seed=42,
        encode_chunk_size=256,
    ):
        super().__init__()
        if encode_chunk_size < 1:
            raise ValueError("encode_chunk_size must be positive.")
        self.context_tokens = context_tokens
        self.encode_chunk_size = encode_chunk_size
        text_width = text_model.token_embedding.weight.shape[1]
        initial_context = self._initial_context(
            text_model,
            tokenizer,
            context_tokens,
            text_width,
            seed,
        )
        self.projector = SharedPatchToPromptProjector(
            visual_width=visual_width,
            text_width=text_width,
            latent_width=latent_width,
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
                generator = torch.Generator(device="cpu").manual_seed(seed + 2)
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

    def _tokens_for(self, categories, split):
        if split not in {"seen", "unseen"}:
            raise ValueError("split must be 'seen' or 'unseen'.")
        token_bank = self.seen_tokens if split == "seen" else self.unseen_tokens
        return token_bank[categories.long().to(token_bank.device)]

    def forward(self, text_model, patch_features, categories, split="seen"):
        contexts, attention = self.projector(patch_features)
        tokens = self._tokens_for(categories, split).to(patch_features.device)
        features = torch.cat(
            [
                encode_openai_soft_prompts(
                    text_model,
                    tokens[start : start + self.encode_chunk_size],
                    contexts[start : start + self.encode_chunk_size],
                )
                for start in range(0, len(tokens), self.encode_chunk_size)
            ],
            dim=0,
        )
        return F.normalize(features.float(), dim=-1), attention

    def trainable_parameter_count(self):
        return self.projector.trainable_parameter_count()


def shared_prompt_infonce_loss(
    sketch_prompt_features,
    photo_prompt_features,
    target_photo_indices,
    temperature=0.07,
):
    """Match every sketch prompt to its exact photo prompt in the gallery."""
    if temperature <= 0:
        raise ValueError("Prompt temperature must be greater than zero.")
    if photo_prompt_features.shape[0] != 100:
        raise RuntimeError("Prompt training expects a 100-photo gallery.")
    sketches = F.normalize(sketch_prompt_features.float(), dim=-1)
    photos = F.normalize(photo_prompt_features.float(), dim=-1)
    targets = target_photo_indices.to(sketches.device).long()
    if targets.shape[0] != sketches.shape[0]:
        raise RuntimeError("Every sketch prompt needs one photo target.")
    logits = sketches @ photos.t() / temperature
    return F.cross_entropy(logits, targets)
