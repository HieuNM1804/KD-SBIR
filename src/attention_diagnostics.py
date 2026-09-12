"""Frozen encoder diagnostics. Pair attribution is NOT native cross-attention."""

import hashlib
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from src.model import IndependentVisualPromptLearner


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder:
    def __init__(self, model, prompts=None, controller=None, openclip=False):
        self.model, self.prompts, self.controller = model, prompts, controller
        self.visual = model.visual
        self.batch_first = (
            bool(getattr(self.visual.transformer, "batch_first", False))
            if openclip
            else False
        )
        self.grid = math.isqrt(self.visual.positional_embedding.shape[0] - 1)
        self.size = self.grid * self.visual.conv1.kernel_size[0]

    def encode(self, images, modality):
        images = images.to(
            device=self.visual.conv1.weight.device, dtype=self.visual.conv1.weight.dtype
        )
        if self.controller is not None:
            value = self.controller(images, modality)
        elif self.prompts is not None:
            ctx, compounds = self.prompts[modality]()
            value = self.visual(images, ctx, compounds)
        else:
            value = self.model.encode_image(images)
        return F.normalize(value.float(), dim=-1)


def student_encoders(model, checkpoint, seed=42):
    """Reject missing prompts or altered frozen backbone instead of demo fallback."""
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Explicit student checkpoint required: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    model.eval().requires_grad_(False)
    backbone = {
        k.removeprefix("model.clip_model."): v
        for k, v in state.items()
        if k.startswith("model.clip_model.")
    }
    if not backbone:
        raise ValueError(
            "Checkpoint has no model.clip_model weights; cannot verify the frozen baseline"
        )
    for name, value in model.state_dict().items():
        if name not in backbone or not torch.equal(
            value.cpu(), backbone[name].to(value.dtype)
        ):
            raise ValueError(f"Checkpoint backbone differs from the base CLIP: {name}")
    prompts, config = {}, {}
    width = model.visual.ln_pre.normalized_shape[0]
    for modality in ("photo", "sketch"):
        prefix = f"model.{modality}_visual_prompt."
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if "ctx" not in sub or sub["ctx"].ndim != 2 or sub["ctx"].shape[1] != width:
            raise ValueError(f"Missing/incompatible trained {modality} prompts")
        depth = 1 + sum(k.startswith("compound_prompts.") for k in sub)
        if depth > len(model.visual.transformer.resblocks):
            raise ValueError("Prompt depth exceeds student depth")
        learner = IndependentVisualPromptLearner(
            sub["ctx"].shape[0], width, seed, depth
        )
        learner.load_state_dict(sub, strict=True)
        prompts[modality] = (
            learner.to(model.visual.conv1.weight.device).eval().requires_grad_(False)
        )
        config[modality] = {"n_ctx": sub["ctx"].shape[0], "depth": depth}
    return (
        Encoder(model),
        Encoder(model, prompts=prompts),
        {
            "checkpoint": str(checkpoint),
            "sha256": file_hash(checkpoint),
            "prompts": config,
            "frozen_backbone_verified": True,
        },
    )


def teacher_encoder(cache_path, mode, dataset, device):
    import open_clip
    from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, TEACHER_CACHE_FORMAT_VERSION
    from src.teacher_prompts import TeacherPromptController

    payload, meta = None, None
    if mode == "tuned":
        if not cache_path or not Path(cache_path).is_file():
            raise FileNotFoundError(
                "--teacher_cache_path must explicitly name the teacher cache used for training"
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        meta = payload.get("metadata", {})
        if meta.get("format_version") != TEACHER_CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported teacher cache format")
        if meta.get("dataset") != dataset or meta.get("max_size") != 224:
            raise ValueError("Teacher cache dataset or input size mismatch")
        if (
            meta.get("teacher_model") != DFN5B_MODEL
            or meta.get("teacher_pretrained") != DFN5B_PRETRAINED
        ):
            raise ValueError("Teacher cache backbone/pretraining mismatch")
        if meta.get("pretrain_epochs", 0) < 1 or not payload.get(
            "teacher_prompt_state_dict"
        ):
            raise ValueError(
                "Tuned teacher requires trained prompts, not random initialization"
            )
    elif mode != "raw":
        raise ValueError(mode)
    model = (
        open_clip.create_model(
            DFN5B_MODEL,
            pretrained=DFN5B_PRETRAINED,
            precision="fp16" if device.type == "cuda" else "fp32",
            device=device,
        )
        .eval()
        .requires_grad_(False)
    )
    controller = None
    if payload is not None:
        controller = TeacherPromptController(
            model.visual,
            meta["teacher_n_ctx_visual"],
            meta["teacher_prompt_depth"],
            meta["teacher_prompt_std"],
            meta["teacher_prompt_seed"],
        )
        controller.load_state_dict(payload["teacher_prompt_state_dict"], strict=True)
        controller.eval().requires_grad_(False)
    return Encoder(model, controller=controller, openclip=True), {
        "mode": mode,
        "cache": cache_path if mode == "tuned" else None,
        "sha256": file_hash(cache_path) if mode == "tuned" else None,
        "metadata": meta,
        "note": "Explicit cache validated; correspondence with student training run must be checked by user.",
    }


def pair_attribution(encoder, sketch, photo):
    """Signed gradient-times-activation at input to final block's attention.

    Detach ONLY that layer-normalized activation and require its gradient. For
    this partial derivative upstream gradients are unnecessary. This preserves
    values/output and avoids storing a 32-block teacher backward graph.
    Both maps explain the SAME cosine score for this pair, in their own model.
    """
    layer = encoder.visual.transformer.resblocks[-1].ln_1
    activations = []

    def capture(_module, _inputs, output):
        leaf = output.detach().requires_grad_(True)
        activations.append(leaf)
        return leaf

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            s = encoder.encode(sketch, "sketch")
            p = encoder.encode(photo, "photo")
            similarity = (s * p).sum()
            if len(activations) != 2:
                raise RuntimeError("Expected one final attention input per image")
            gradients = torch.autograd.grad(similarity, activations)
        maps = []
        for activation, gradient in zip(activations, gradients):
            relevance = (activation.detach().float() * gradient.detach().float()).sum(
                -1
            )
            vector = relevance[0] if encoder.batch_first else relevance[:, 0]
            maps.append(
                vector[1 : 1 + encoder.grid**2]
                .reshape(encoder.grid, encoder.grid)
                .cpu()
            )
        return maps, float(similarity.detach())
    finally:
        handle.remove()


def normalize_map(raw):
    raw = raw.detach().float()
    span = raw.max() - raw.min()
    return (
        ((raw - raw.min()) / span).numpy()
        if span > 1e-12
        else torch.zeros_like(raw).numpy()
    )


def rollout(attentions, patches, prompt_depth=0):
    """Forward layer order: R_l = A_l R_(l-1). Reset replaced prompt rows.

    Tracks influence of original image tokens, excluding independent prompt
    sources, so it is a heuristic image-path rollout rather than full attribution.
    """
    n = attentions[0].shape[-1]
    result = torch.eye(n)
    for index, weights in enumerate(attentions):
        if 0 < index < prompt_depth and n > patches + 1:
            result = result.clone()
            result[patches + 1 :] = 0
        a = weights[0].float()
        if a.shape != (n, n):
            raise ValueError("Rollout requires constant token count")
        a = a + torch.eye(n)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    return result[0, 1 : patches + 1]


def native_attention(encoder, image, modality, method="last"):
    blocks = list(encoder.visual.transformer.resblocks)
    selected = blocks[-1:] if method == "last" else blocks
    weights, handles = [], []

    def pre(_module, args, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    def post(_module, _args, output):
        if not isinstance(output, tuple) or output[1] is None or output[1].ndim != 3:
            raise RuntimeError("Expected averaged MHA weights [batch,query,key]")
        weights.append(output[1].detach().float().cpu())

    try:
        for block in selected:
            if not isinstance(block.attn, torch.nn.MultiheadAttention):
                raise TypeError(
                    "Native attention diagnostics require nn.MultiheadAttention"
                )
            handles.extend(
                [
                    block.attn.register_forward_pre_hook(pre, with_kwargs=True),
                    block.attn.register_forward_hook(post),
                ]
            )
        with torch.no_grad():
            encoder.encode(image, modality)
        if len(weights) != len(selected):
            raise RuntimeError("Not all attention blocks were captured")
        n = encoder.grid**2
        depth = (
            encoder.controller.depth
            if encoder.controller
            else (
                1 + len(encoder.prompts[modality].compound_prompts)
                if encoder.prompts
                else 0
            )
        )
        vector = (
            weights[-1][0, 0, 1 : 1 + n]
            if method == "last"
            else rollout(weights, n, depth)
        )
        return vector.reshape(encoder.grid, encoder.grid)
    finally:
        for handle in handles:
            handle.remove()
