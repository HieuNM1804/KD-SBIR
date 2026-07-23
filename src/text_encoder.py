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
        if prompt_embeddings is None:
            prompt_embeddings = self.token_embedding(tokenized_text)
        x = prompt_embeddings.type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the end-of-text embedding.
        x = (
            x[
                torch.arange(x.shape[0], device=x.device),
                tokenized_text.argmax(dim=-1),
            ]
            @ self.text_projection
        )

        return x
