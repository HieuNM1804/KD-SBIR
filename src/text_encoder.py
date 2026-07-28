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

    def forward(self, tokenized_text, prompt_embeddings=None):
        # L = 77 tokens, D = 512 (embedding dimension), N = 64 (batch size)
        # tokenized_text : [N, L]
        if prompt_embeddings is None:
            prompt_embeddings = self.token_embedding(tokenized_text)
        x = prompt_embeddings.type(self.dtype)  # [N, L, D]
        x = x + self.positional_embedding.type(self.dtype)  # [N, L, D]
        x = x.permute(1, 0, 2)  # [N, L, D] -> [L, N, D]
        x = self.resblocks(x)  # [L, N, D]
        x = x.permute(1, 0, 2)  # [L, N, D] -> [N, L, D]
        x = self.ln_final(x).type(self.dtype)  # [N, L, D]

        x = (
            x[
                torch.arange(x.shape[0], device=x.device),
                tokenized_text.argmax(dim=-1),
            ]
            @ self.text_projection  # [512, 512]
        )  # [N, D]

        return x  # [N, D]
