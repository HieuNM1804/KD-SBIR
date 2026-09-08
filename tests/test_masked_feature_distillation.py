from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from clip.model import VisionTransformer
from src.losses import masked_feature_distillation_loss, loss_fn
from src.model import CustomCLIP, MFDProjector, ZS_SBIR
from src.train import add_masked_feature_distillation_args


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 512
        self.ln_pre = nn.LayerNorm(512)
        self.transformer = SimpleNamespace(layers=12)
        self.mask_ratios = []

    def forward(
        self,
        images,
        prompt,
        compound_prompts,
        mask_ratio=0.0,
    ):
        self.mask_ratios.append(mask_ratio)
        prompt_signal = prompt.mean() if prompt is not None else 0.0
        for compound_prompt in compound_prompts:
            prompt_signal = prompt_signal + compound_prompt.mean()
        return images + prompt_signal


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()
        self.dtype = torch.float32


def make_args(**overrides):
    values = {
        "n_ctx_visual": 2,
        "prompt_depth": 3,
        "seed": 42,
        "teacher_cache_path": "",
        "rebuild_teacher_cache": False,
        "teacher_pretrain_epochs": 0,
        "photo_mask_ratio": 0.75,
        "sketch_mask_ratio": 0.5,
        "lambda_mfd": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mfd_loss_is_zero_for_aligned_features():
    features = torch.randn(4, 1024)
    loss = masked_feature_distillation_loss(features, features * 7.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_mfd_loss_is_scale_invariant():
    student = torch.randn(4, 1024)
    teacher = torch.randn(4, 1024)
    reference = masked_feature_distillation_loss(student, teacher)
    scaled = masked_feature_distillation_loss(student * 3.0, teacher * 9.0)
    assert scaled.item() == pytest.approx(reference.item(), abs=1e-7)


def test_mae_random_masking_drops_patch_tokens_per_sample():
    patches = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    torch.manual_seed(7)
    kept, mask, restore_indices = VisionTransformer.random_masking(
        patches,
        mask_ratio=0.75,
    )

    assert kept.shape == (2, 2, 3)
    assert mask.shape == (2, 8)
    assert restore_indices.shape == (2, 8)
    assert torch.equal(mask.sum(dim=1), torch.tensor([6.0, 6.0]))
    assert set(mask.unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("mask_ratio", [-0.1, 1.0, 1.1])
def test_mae_random_masking_rejects_invalid_ratios(mask_ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        VisionTransformer.random_masking(torch.randn(2, 4, 8), mask_ratio)


def test_masked_visual_forward_keeps_cls_and_drops_only_patches():
    visual = VisionTransformer(
        input_resolution=8,
        patch_size=4,
        width=64,
        layers=1,
        heads=1,
        output_dim=16,
    ).eval()
    captured = []
    handle = visual.ln_pre.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    try:
        output = visual(torch.randn(2, 3, 8, 8), mask_ratio=0.5)
    finally:
        handle.remove()

    # Four image patches become two kept patches, plus the unmasked CLS token.
    assert captured[0].shape == (2, 3, 64)
    expected_cls = visual.class_embedding + visual.positional_embedding[0]
    assert torch.allclose(captured[0][:, 0], expected_cls.expand(2, -1))
    assert output.shape == (2, 16)


def test_zero_mask_ratio_uses_the_original_visual_path():
    visual = VisionTransformer(
        input_resolution=8,
        patch_size=4,
        width=64,
        layers=1,
        heads=1,
        output_dim=16,
    ).eval()
    images = torch.randn(2, 3, 8, 8)
    implicit = visual(images)
    explicit = visual(images, mask_ratio=0.0)
    assert torch.equal(implicit, explicit)


def test_mfd_projector_has_expected_shape():
    projector = MFDProjector()
    assert projector(torch.randn(3, 512)).shape == (3, 1024)


def test_photo_and_sketch_projectors_are_separate():
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )

    assert model.photo_mfd_projector is not model.sketch_mfd_projector
    assert (
        model.photo_mfd_projector.projection.weight.data_ptr()
        != model.sketch_mfd_projector.projection.weight.data_ptr()
    )


def test_loss_averages_modalities_and_applies_mfd_weight(monkeypatch):
    calls = []

    def fake_loss(student, teacher):
        calls.append((student, teacher))
        return student.new_tensor(2.0 if len(calls) == 1 else 4.0)

    monkeypatch.setattr("src.losses.masked_feature_distillation_loss", fake_loss)
    features = tuple(torch.randn(2, 1024) for _ in range(4))
    total, values = loss_fn(make_args(lambda_mfd=2.0), features)
    assert len(calls) == 2
    assert values["mfd_photo"].item() == 2.0
    assert values["mfd_sketch"].item() == 4.0
    assert values["mfd"].item() == 3.0
    assert total.item() == 6.0


def test_cli_defaults_and_modality_specific_ratios():
    parser = add_masked_feature_distillation_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.photo_mask_ratio == 0.75
    assert defaults.sketch_mask_ratio == 0.75
    assert defaults.lambda_mfd == 1.0

    custom = parser.parse_args(
        [
            "--photo_mask_ratio",
            "0.25",
            "--sketch_mask_ratio",
            "0.5",
            "--lambda_mfd",
            "3",
        ]
    )
    assert custom.photo_mask_ratio == 0.25
    assert custom.sketch_mask_ratio == 0.5
    assert custom.lambda_mfd == 3.0


def test_training_masks_student_but_inference_does_not():
    model = CustomCLIP(
        make_args(photo_mask_ratio=0.25, sketch_mask_ratio=0.75),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )
    batch = (
        torch.randn(2, 512),
        torch.randn(2, 512),
        torch.randn(2, 1024),
        torch.randn(2, 1024),
        torch.zeros(2, dtype=torch.long),
    )
    model(batch)
    model.extract_feature(torch.randn(2, 512), "sketch")
    assert model.clip_model.visual.mask_ratios == [0.25, 0.75, 0.0]


def test_gradients_reach_both_projectors_and_prompts_not_teacher():
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )
    teacher_photo = torch.randn(2, 1024, requires_grad=True)
    teacher_sketch = torch.randn(2, 1024, requires_grad=True)
    batch = (
        torch.randn(2, 512),
        torch.randn(2, 512),
        teacher_photo,
        teacher_sketch,
        torch.zeros(2, dtype=torch.long),
    )
    total, _ = loss_fn(model.cfg, model(batch))
    total.backward()

    assert model.photo_mfd_projector.projection.weight.grad is not None
    assert model.sketch_mfd_projector.projection.weight.grad is not None
    assert model.photo_visual_prompt.ctx.grad is not None
    assert model.sketch_visual_prompt.ctx.grad is not None
    assert teacher_photo.grad is None
    assert teacher_sketch.grad is None
    assert not any(
        parameter.requires_grad
        for parameter in model.clip_model.parameters()
    )


def test_inference_bypasses_both_projectors(monkeypatch):
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )

    def fail_if_called(_features):
        raise AssertionError("A train-time MFD projector was used at inference.")

    monkeypatch.setattr(model.photo_mfd_projector, "forward", fail_if_called)
    monkeypatch.setattr(model.sketch_mfd_projector, "forward", fail_if_called)
    output = model.extract_feature(torch.randn(2, 512), "sketch")
    assert output.shape == (2, 512)


def test_lightning_checkpoint_state_and_hyperparameters(monkeypatch):
    monkeypatch.setattr("src.model._load_clip_model", lambda _name: FakeCLIP())
    monkeypatch.setattr("src.model._load_teacher", lambda _args: object())
    args = make_args(lambda_mfd=2.0, backbone="ViT-B/32")
    model = ZS_SBIR(args, classnames=("cat",))

    assert model.hparams["mfd_loss"] == "mse"
    assert model.hparams["lambda_mfd"] == 2.0
    assert model.hparams["photo_mask_ratio"] == 0.75
    assert model.hparams["sketch_mask_ratio"] == 0.5
    assert model.hparams["projector_layout"] == "separate"
    assert "model.photo_mfd_projector.projection.weight" in model.state_dict()
    assert "model.sketch_mfd_projector.projection.weight" in model.state_dict()
