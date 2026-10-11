"""AdamW for the trainable visual prompts of teacher and student."""

import torch


def build_prompt_optimizer(parameters, lr, weight_decay, betas, eps):
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("AdamW requires trainable visual prompt parameters.")
    return torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay,
        betas=tuple(betas), eps=eps,
    )
