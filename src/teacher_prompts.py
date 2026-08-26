import torch
import torch.nn as nn

from src.adapters import ModalityBottleneckAdapters


def _random_prompt(rows, width, std, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.empty(rows, width, dtype=torch.float32)
    nn.init.normal_(value, std=std, generator=generator)
    return nn.Parameter(value.to(device=device))


class ModalityVisualPrompts(nn.Module):
    """Independent deep visual prompts for photo and sketch teacher paths."""

    _MODALITIES = ("photo", "sketch")

    def __init__(self, width, n_ctx, depth, std, seed, device):
        super().__init__()
        self.n_ctx = n_ctx
        self.depth = depth
        self.prompts = nn.ModuleDict()
        for modality_index, modality in enumerate(self._MODALITIES):
            layer_prompts = nn.ParameterList()
            for layer_index in range(depth):
                layer_prompts.append(
                    _random_prompt(
                        rows=n_ctx,
                        width=width,
                        std=std,
                        seed=seed + modality_index * depth + layer_index,
                        device=device,
                    )
                )
            self.prompts[modality] = layer_prompts

    def for_layer(self, modality, layer_index, batch_size, dtype, device):
        if modality not in self._MODALITIES:
            raise ValueError(f"Unsupported teacher modality: {modality}")
        prompt = self.prompts[modality][layer_index]
        return prompt.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )


class TeacherPromptController(nn.Module):
    """Run a frozen OpenCLIP ViT with modality prompts and adapters."""

    def __init__(
        self,
        visual,
        n_ctx,
        depth,
        std,
        seed,
        adapter_bottleneck=0,
        adapter_depth=0,
        adapter_std=0.02,
        adapter_dropout=0.0,
        adapter_scale=1.0,
        adapter_seed=None,
    ):
        super().__init__()
        blocks = list(visual.transformer.resblocks)
        layer_count = len(blocks)
        if depth == -1:
            depth = layer_count
        if not 1 <= depth <= layer_count:
            raise ValueError(
                "teacher_prompt_depth must be -1 or in "
                f"[1, {layer_count}], got {depth}."
            )
        if n_ctx < 1:
            raise ValueError("teacher_n_ctx_visual must be at least 1.")

        width = visual.conv1.out_channels
        object.__setattr__(self, "_visual", visual)
        self.depth = depth
        self.n_ctx = n_ctx
        self.prompt_learner = ModalityVisualPrompts(
            width=width,
            n_ctx=n_ctx,
            depth=depth,
            std=std,
            seed=seed,
            device=visual.conv1.weight.device,
        )
        self.adapter_learner = None
        if adapter_bottleneck > 0:
            if adapter_depth == -1:
                adapter_depth = layer_count
            if not 1 <= adapter_depth <= layer_count:
                raise ValueError(
                    "teacher_adapter_depth must be -1 or in "
                    f"[1, {layer_count}], got {adapter_depth}."
                )
            self.adapter_learner = ModalityBottleneckAdapters(
                width=width,
                bottleneck=adapter_bottleneck,
                depth=adapter_depth,
                std=adapter_std,
                seed=seed + 10_000 if adapter_seed is None else adapter_seed,
                dropout=adapter_dropout,
                scale=adapter_scale,
                device=visual.conv1.weight.device,
            )

    def prompt_parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.prompt_learner.parameters()
        )

    def adapter_parameter_count(self):
        if self.adapter_learner is None:
            return 0
        return self.adapter_learner.trainable_parameter_count()

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, images, modality):
        visual = self._visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(dtype=x.dtype)
        class_token = class_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + visual.positional_embedding.to(dtype=x.dtype)
        x = visual.patch_dropout(x)

        prompt = self.prompt_learner.for_layer(
            modality,
            layer_index=0,
            batch_size=x.shape[0],
            dtype=x.dtype,
            device=x.device,
        )
        x = visual.ln_pre(torch.cat((x, prompt), dim=1))

        batch_first = visual.transformer.batch_first
        if not batch_first:
            x = x.transpose(0, 1).contiguous()

        for layer_index, block in enumerate(visual.transformer.resblocks):
            if 0 < layer_index < self.depth:
                prompt = self.prompt_learner.for_layer(
                    modality,
                    layer_index=layer_index,
                    batch_size=(x.shape[0] if batch_first else x.shape[1]),
                    dtype=x.dtype,
                    device=x.device,
                )
                if batch_first:
                    x = torch.cat((x[:, :-self.n_ctx], prompt), dim=1)
                else:
                    x = torch.cat(
                        (x[:-self.n_ctx], prompt.transpose(0, 1)), dim=0
                    )
            x = block(x)
            if self.adapter_learner is not None:
                x = self.adapter_learner.apply_layer(
                    x, modality, layer_index
                )

        if not batch_first:
            x = x.transpose(0, 1)
        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled


def build_teacher_prompt_controller(
    teacher,
    n_ctx,
    depth,
    std=0.02,
    seed=42,
    adapter_bottleneck=0,
    adapter_depth=0,
    adapter_std=0.02,
    adapter_dropout=0.0,
    adapter_scale=1.0,
    adapter_seed=None,
):
    return TeacherPromptController(
        visual=teacher.visual,
        n_ctx=n_ctx,
        depth=depth,
        std=std,
        seed=seed,
        adapter_bottleneck=adapter_bottleneck,
        adapter_depth=adapter_depth,
        adapter_std=adapter_std,
        adapter_dropout=adapter_dropout,
        adapter_scale=adapter_scale,
        adapter_seed=adapter_seed,
    )
