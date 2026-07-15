import torch
import torch.nn as nn

from clip import clip

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, type='photo'):
        super().__init__()
        self.cfg = cfg
        n_ctx = cfg.n_ctx
        ctx_init = "a photo/sketch of "
            
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.max_size
        
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors)
        
        self.proj = nn.Linear(ctx_dim, 768)

        # Keep the exact RNG sequence of the recorded shallow-prompt benchmark.
        # The original implementation initialized this projection even when
        # prompt_depth=1, then discarded it because there were no deep prompts.
        # Removing that unused allocation changes the sketch prompt projection
        # initialization and makes seed=42 reproduce a different run.
        _benchmark_rng_compat = nn.Linear(ctx_dim, 768)

        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self):
        return self.proj(self.ctx)
