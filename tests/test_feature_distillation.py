from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.losses import feature_distillation_loss, loss_fn
from src.model import CustomCLIP, FeatureProjector, ZS_SBIR
from src.train import add_feature_distillation_args


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 512
        self.ln_pre = nn.LayerNorm(512)
        self.transformer = SimpleNamespace(layers=12)

    def forward(self, images, prompt, compound_prompts):
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
        "feature_loss": "mse",
        "lambda_fd": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("loss_type", ["mse", "cosine"])
def test_feature_loss_is_zero_for_aligned_features(loss_type):
    features = torch.randn(4, 1024)
    loss = feature_distillation_loss(features, features * 7.0, loss_type)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("loss_type", ["mse", "cosine"])
def test_feature_loss_is_scale_invariant(loss_type):
    student = torch.randn(4, 1024)
    teacher = torch.randn(4, 1024)
    reference = feature_distillation_loss(student, teacher, loss_type)
    scaled = feature_distillation_loss(student * 3.0, teacher * 9.0, loss_type)
    assert scaled.item() == pytest.approx(reference.item(), abs=1e-7)


def test_feature_projector_has_expected_shape():
    projector = FeatureProjector()
    assert projector(torch.randn(3, 512)).shape == (3, 1024)


def test_photo_and_sketch_projectors_are_separate():
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )

    assert model.photo_feature_projector is not model.sketch_feature_projector
    assert (
        model.photo_feature_projector.projection.weight.data_ptr()
        != model.sketch_feature_projector.projection.weight.data_ptr()
    )
    assert (
        model.photo_feature_projector.projection.bias.data_ptr()
        != model.sketch_feature_projector.projection.bias.data_ptr()
    )


def test_loss_selects_one_method_and_averages_modalities(monkeypatch):
    calls = []

    def fake_loss(student, teacher, loss_type):
        calls.append(loss_type)
        return student.new_tensor(2.0 if len(calls) == 1 else 4.0)

    monkeypatch.setattr("src.losses.feature_distillation_loss", fake_loss)
    features = tuple(torch.randn(2, 1024) for _ in range(4))
    total, values = loss_fn(make_args(feature_loss="cosine"), features)
    assert calls == ["cosine", "cosine"]
    assert values["fd_photo"].item() == 2.0
    assert values["fd_sketch"].item() == 4.0
    assert values["fd"].item() == 3.0
    assert total.item() == 3.0


def test_cli_defaults_and_choices():
    parser = add_feature_distillation_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.feature_loss == "mse"
    assert defaults.lambda_fd == 1.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--feature_loss", "both"])


@pytest.mark.parametrize("loss_type", ["mse", "cosine"])
def test_gradients_reach_both_projectors_and_prompts_not_teacher(loss_type):
    model = CustomCLIP(
        make_args(feature_loss=loss_type),
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

    assert model.photo_feature_projector.projection.weight.grad is not None
    assert model.photo_feature_projector.projection.bias.grad is not None
    assert model.sketch_feature_projector.projection.weight.grad is not None
    assert model.sketch_feature_projector.projection.bias.grad is not None
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
        raise AssertionError("The train-time projector was used during inference.")

    monkeypatch.setattr(
        model.photo_feature_projector,
        "forward",
        fail_if_called,
    )
    monkeypatch.setattr(
        model.sketch_feature_projector,
        "forward",
        fail_if_called,
    )
    output = model.extract_feature(torch.randn(2, 512), "sketch")
    assert output.shape == (2, 512)


def test_lightning_checkpoint_state_and_hyperparameters(monkeypatch):
    monkeypatch.setattr("src.model._load_clip_model", lambda _name: FakeCLIP())
    monkeypatch.setattr("src.model._load_teacher", lambda _args: object())
    args = make_args(feature_loss="cosine", lambda_fd=1.0, backbone="ViT-B/32")
    model = ZS_SBIR(args, classnames=("cat",))

    assert model.hparams["feature_loss"] == "cosine"
    assert model.hparams["lambda_fd"] == 1.0
    assert model.hparams["projector_layout"] == "separate"
    assert model.hparams["projector_input_dim"] == 512
    assert model.hparams["projector_output_dim"] == 1024
    assert (
        "model.photo_feature_projector.projection.weight"
        in model.state_dict()
    )
    assert (
        "model.sketch_feature_projector.projection.weight"
        in model.state_dict()
    )
