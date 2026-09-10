"""Offline-ready numerical smoke: actual student, 12 layers, ONLY joint KD.

Run with runpy from repository root. No pretrained model download, dataset,
or pytest needed. This is a numerical check, not a retrieval benchmark.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from types import SimpleNamespace

import torch
from torch.nn import functional as F

from clip.model import CLIP, convert_weights
from src.model import CustomCLIP, ZS_SBIR
from src.train import seed_everything


def config(**overrides):
    settings = dict(
        seed=42, prompt_depth=12, n_ctx_visual=3,
        lambda_domain=0.0, lambda_modality=0.0, lambda_joint_geometry=1.0,
        joint_cross_weight=0.5, geometry_diagnostics=True,
        kd_temperature=0.07, photo_text_kd_temperature=0.15,
        sketch_text_kd_temperature=0.02, teacher_pretrain_epochs=0,
        teacher_cache_path="", rebuild_teacher_cache=False,
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


def make_student(args, device):
    clip_model = CLIP(32, 16, 12, 64, 4, 77, 49408, 64, 1, 1).to(device)
    if device == "cuda":
        convert_weights(clip_model)
    student = CustomCLIP(args, clip_model, ["cat", "dog"], teacher=None).to(device)
    student.teacher_active = True
    if args.lambda_modality > 0:
        student._teacher_photo_text = F.normalize(torch.randn(2, 1024, device=device), dim=-1)
        student._teacher_sketch_text = F.normalize(torch.randn(2, 1024, device=device), dim=-1)
    return student


def run_smoke(device):
    seed_everything(42)
    args = config()
    student = make_student(args, device)
    wrapper = ZS_SBIR.__new__(ZS_SBIR)
    torch.nn.Module.__init__(wrapper)
    wrapper.args, wrapper.model = args, student
    logged = {}
    wrapper.log = lambda key, value, **kw: logged.update({key: value})
    wrapper.train()
    batch = (
        torch.randn(4, 3, 16, 16, device=device),
        torch.randn(4, 3, 16, 16, device=device),
        torch.randn(4, 1024, device=device, requires_grad=True),
        torch.randn(4, 1024, device=device, requires_grad=True),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    keys = tuple(student.state_dict())
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=0.01)
    before = {name: p.detach().clone() for name, p in student.named_parameters() if p.requires_grad}
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        loss = wrapper.training_step(batch, 0)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert logged["DOMAIN"] == 0 and logged["MODALITY"] == 0
    assert torch.equal(logged["JOINT"], loss)
    loss.backward()
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    assert len(before) == 24  # one shallow + 11 deep prompts, for both modalities
    assert batch[2].grad is None and batch[3].grad is None
    optimizer.step()
    for name, p in student.named_parameters():
        if name in before:
            assert not torch.equal(before[name], p), name
    assert tuple(student.state_dict()) == keys
    student.eval()
    with torch.inference_mode():
        enabled = student.extract_feature(batch[0], "photo")
        args.lambda_joint_geometry = 0
        disabled = student.extract_feature(batch[0], "photo")
    assert torch.equal(enabled, disabled)
    assert all(torch.isfinite(v).all() for v in logged.values())
    print(f"Joint KD alone: all 24 visual prompt tensors receive gradients and update on {device}.")
    print("Frozen backbone/teacher targets; unchanged inference/state keys: OK")


if __name__ == "__main__":
    torch.set_num_threads(2)
    run_smoke("cuda" if torch.cuda.is_available() else "cpu")
