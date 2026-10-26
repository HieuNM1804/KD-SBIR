"""Local-only image towers with the project's modality-specific deep prompts."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.cost_prompts import PromptedHFVision, PromptedStudent
from src.teacher_prompts import TeacherPromptController


SPECS = {
    "dfn5b_h14": dict(
        name="DFN5B ViT-H/14", directory="dfn5b_h14",
        filename="open_clip_pytorch_model.bin", repository="apple/DFN5B-CLIP-ViT-H-14",
        revision="11738501a1db6d5e0a3451a71ba100be02e577e6",
        sha256="d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae",
        patch=14, width=1280, layers=32, mlp=5120, dim=1024,
    ),
    "clip_b32": dict(
        name="OpenAI CLIP ViT-B/32", directory="clip_b32", filename="ViT-B-32.pt",
        url="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        sha256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        patch=32, width=768, layers=12, mlp=3072, dim=512,
    ),
    "tinyclip_61m": dict(
        name="TinyCLIP ViT-61M/32", directory="tinyclip61m_student",
        filename="model.safetensors",
        repository="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
        revision="94cbbea5c7949cfe7bdafde64bcea5e403f59852",
        patch=32, width=640, layers=12, mlp=2560, dim=512,
    ),
}


class PromptedTeacher(nn.Module):
    def __init__(self, visual, n_ctx, depth, seed):
        super().__init__()
        self.visual = visual
        self.controller = TeacherPromptController(visual, n_ctx, depth, 0.02, seed)

    def forward(self, images, modality="sketch"):
        return F.normalize(self.controller(images, modality).float(), dim=-1)


def read_payload(path):
    # Only explicitly supplied project checkpoints/caches are opened. No downloads.
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def restore_prompts(model, path, teacher=False):
    """Restore only prompt tensors; frozen towers come from the pinned originals.

    A teacher cache contains teacher_prompt_state_dict. Student Lightning
    checkpoints contain model.{sketch,photo}_visual_prompt.*. Both modalities
    are required; shape errors are fatal rather than silently timing other sizes.
    """
    if not path:
        return "seeded initialization (cost only; no trained accuracy claim)"
    payload = read_payload(Path(path))
    state = payload.get("teacher_prompt_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"No prompt state in {path}")
    while state:
        prefix = next((p for p in ("module.", "model.")
                       if all(k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {k[len(prefix):]: v for k, v in state.items()}
    if teacher:
        selected = {}
        for key, value in state.items():
            for prefix in ("teacher_prompts.", "controller.", ""):
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    if stripped.startswith("prompt_learner."):
                        selected[stripped] = value
                        break
        # Strict loading checks missing AND extra layers, not just tensor widths.
        model.controller.load_state_dict(selected, strict=True)
    else:
        for modality in ("sketch", "photo"):
            prefix = modality + "_visual_prompt."
            selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            getattr(model, modality + "_visual_prompt").load_state_dict(selected, strict=True)
    return str(Path(path).resolve())


def load_model(key, models_root, n_ctx, depth, seed=42, prompt_path=None):
    """Load on CPU, discard text tower, retain both prompt banks exactly once."""
    spec = SPECS[key]
    path = Path(models_root) / spec["directory"] / spec["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundled model: {path}")
    if key == "dfn5b_h14":
        import open_clip
        raw = open_clip.create_model("ViT-H-14-quickgelu", pretrained=None, device="cpu")
        open_clip.load_checkpoint(raw, str(path), strict=True)
        tower = raw.visual
        actual = (tower.conv1.kernel_size[0], tower.conv1.weight.shape[0],
                  len(tower.transformer.resblocks), tower.transformer.resblocks[0].mlp.c_fc.out_features,
                  tower.proj.shape[1])
        model = PromptedTeacher(tower, n_ctx, depth, seed)
    elif key == "clip_b32":
        from clip.model import build_model
        state = dict(torch.jit.load(str(path), map_location="cpu").state_dict())
        raw = build_model(state).float()
        tower = raw.visual
        actual = (tower.conv1.kernel_size[0], tower.conv1.weight.shape[0],
                  len(tower.transformer.resblocks), tower.transformer.resblocks[0].mlp.c_fc.out_features,
                  tower.proj.shape[1])
        model = PromptedStudent(tower, spec["width"], spec["layers"], n_ctx, depth, seed)
    else:
        from transformers import CLIPModel
        raw = CLIPModel.from_pretrained(str(path.parent), local_files_only=True,
                                       attn_implementation="sdpa")
        cfg = raw.config.vision_config
        actual = (cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers,
                  cfg.intermediate_size, raw.config.projection_dim)
        if cfg.image_size != 224:
            raise ValueError(f"Expected TinyCLIP input size 224, got {cfg.image_size}")
        tower = PromptedHFVision(raw.vision_model, raw.visual_projection)
        # PromptedHFVision uses non-registering references, just as the training branch.
        tower.add_module("vision_owner", raw.vision_model)
        tower.add_module("projection_owner", raw.visual_projection)
        model = PromptedStudent(tower, spec["width"], spec["layers"], n_ctx, depth, seed)
    expected = tuple(spec[k] for k in ("patch", "width", "layers", "mlp", "dim"))
    if tuple(actual) != expected:
        raise ValueError(f"Unexpected {key} architecture: {actual}; expected {expected}")
    prompt_source = restore_prompts(model, prompt_path, teacher=key == "dfn5b_h14")
    return model.float().eval().requires_grad_(False), prompt_source


def set_precision(model, key, device, precision):
    """Match the main branches: mixed FP16 for OpenAI/OpenCLIP, FP32 prompts."""
    model.float().to(device).eval().requires_grad_(False)
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("Use fp32 for CPU inference.")
        if key == "dfn5b_h14":
            from open_clip.model import convert_weights_to_lp
            convert_weights_to_lp(model.visual, dtype=torch.float16)
        elif key == "clip_b32":
            from clip.model import convert_weights
            convert_weights(model.visual)
        else:
            model.visual.half()
    elif precision != "fp32":
        raise ValueError(precision)
    return model


def parameter_counts(model, teacher=False):
    vision = sum(p.numel() for p in model.visual.parameters())
    if teacher:
        photo = sum(p.numel() for p in model.controller.prompt_learner.prompts["photo"].parameters())
        sketch = sum(p.numel() for p in model.controller.prompt_learner.prompts["sketch"].parameters())
    else:
        photo = sum(p.numel() for p in model.photo_visual_prompt.parameters())
        sketch = sum(p.numel() for p in model.sketch_visual_prompt.parameters())
    return dict(vision_parameters=vision, sketch_prompt_parameters=sketch,
                photo_prompt_parameters=photo, query_parameters=vision + sketch,
                deployed_parameters=vision + sketch + photo)


def vit_flops(spec, n_ctx, image_size=224):
    """Conv + linear projections + QK^T/AV, 2 FLOPs/MAC; no bias/elementwise ops.

    Prompt tokens persist through ALL transformer blocks even beyond prompt
    replacement depth. Count actual token length, not only prompted layers.
    """
    if image_size % spec["patch"]:
        raise ValueError("Input size must be divisible by patch size")
    patches = (image_size // spec["patch"]) ** 2
    tokens = patches + 1 + n_ctx
    w, h, layers = (spec[k] for k in ("width", "mlp", "layers"))
    counts = dict(
        patch_conv=2 * patches * 3 * spec["patch"] ** 2 * w,
        qkv_and_output=layers * 2 * tokens * 4 * w * w,
        attention_qk_av=layers * 4 * tokens * tokens * w,
        mlp=layers * 4 * tokens * w * h,
        embedding_projection=2 * w * spec["dim"],
    )
    return dict(tokens=tokens, total=sum(counts.values()), breakdown=counts)
