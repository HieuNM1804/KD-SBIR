import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.resblocks = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        with torch.no_grad():
            prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class VisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, type='photo'):
        super().__init__()
        if cfg.n_ctx <= 0:
            raise ValueError("n_ctx must be greater than 0 for visual prompting.")

        dtype = clip_model.dtype
        visual_width = clip_model.visual.ln_pre.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        assert (
            cfg.max_size == clip_imsize
        ), f"cfg_imsize ({cfg.max_size}) must equal to clip_imsize ({clip_imsize})"

        # Preserve the baseline RNG position so later teacher-adapter weights
        # keep the same initialization after removing text-to-visual projectors.
        text_width = clip_model.ln_final.weight.shape[0]
        if cfg.n_ctx > 4:
            text_ctx_rng_compat = torch.empty(cfg.n_ctx, text_width, dtype=dtype)
            nn.init.normal_(text_ctx_rng_compat)
        nn.Linear(text_width, visual_width)
        nn.Linear(text_width, visual_width)

        visual_ctx = torch.empty(cfg.n_ctx, visual_width, dtype=dtype)
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + (0 if type == 'photo' else 1))
        nn.init.normal_(visual_ctx, std=0.02, generator=generator)
        self.ctx = nn.Parameter(visual_ctx)

    def forward(self):
        return self.ctx
