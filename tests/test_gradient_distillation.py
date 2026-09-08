from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.losses import (
    contrastive_embedding_gradients,
    gradient_distillation_loss,
    loss_fn,
    multi_positive_targets,
)
from src.model import CustomCLIP, GDProjector, ZS_SBIR
from src.train import add_gradient_distillation_args


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
        "task_temperature": 0.07,
        "gd_temperature": 0.07,
        "lambda_task": 1.0,
        "lambda_gd": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_multi_positive_targets_are_uniform_per_class():
    labels = torch.tensor([0, 0, 1, 2, 2])
    targets = multi_positive_targets(labels, labels, torch.float32)
    assert torch.equal(targets.sum(dim=-1), torch.ones(5))
    assert torch.equal(targets[0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(targets[2], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(targets[3], torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5]))


def test_analytic_gradients_match_autograd():
    torch.manual_seed(7)
    anchors = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
    candidates = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
    labels = torch.tensor([0, 0, 1, 2, 2])
    temperature = 0.2
    targets = multi_positive_targets(labels, labels, anchors.dtype)
    reference_loss = -(
        targets
        * F.log_softmax(anchors @ candidates.t() / temperature, dim=-1)
    ).sum(dim=-1).mean()
    expected_anchor, expected_candidate = torch.autograd.grad(
        reference_loss,
        (anchors, candidates),
    )

    actual_anchor, actual_candidate = contrastive_embedding_gradients(
        anchors,
        candidates,
        labels,
        labels,
        temperature,
    )
    assert torch.allclose(actual_anchor, expected_anchor, atol=1e-6)
    assert torch.allclose(actual_candidate, expected_candidate, atol=1e-6)


def test_gradient_distillation_is_zero_for_identical_spaces():
    photo = torch.randn(5, 1024)
    sketch = torch.randn(5, 1024)
    labels = torch.tensor([0, 0, 1, 2, 2])
    loss, terms = gradient_distillation_loss(
        photo,
        sketch,
        photo,
        sketch,
        labels,
        temperature=0.07,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert all(value.item() == pytest.approx(0.0, abs=1e-8) for value in terms.values())


def test_gd_projector_has_expected_shape():
    projector = GDProjector()
    assert projector(torch.randn(3, 512)).shape == (3, 1024)


def test_photo_and_sketch_projectors_are_separate():
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )
    assert model.photo_gd_projector is not model.sketch_gd_projector
    assert (
        model.photo_gd_projector.projection.weight.data_ptr()
        != model.sketch_gd_projector.projection.weight.data_ptr()
    )
    assert (
        model.photo_gd_projector.projection.bias.data_ptr()
        != model.sketch_gd_projector.projection.bias.data_ptr()
    )


def test_loss_combines_task_and_gradient_weights(monkeypatch):
    def fake_task(*_args):
        return torch.tensor(2.0)

    def fake_gd(*_args):
        return torch.tensor(3.0), {
            "gd_sketch_anchor": torch.tensor(0.5),
            "gd_photo_key": torch.tensor(0.5),
            "gd_photo_anchor": torch.tensor(1.0),
            "gd_sketch_key": torch.tensor(1.0),
        }

    monkeypatch.setattr("src.losses.soft_target_contrastive_loss", fake_task)
    monkeypatch.setattr("src.losses.gradient_distillation_loss", fake_gd)
    labels = torch.tensor([0, 1])
    features = (
        torch.randn(2, 512),
        torch.randn(2, 512),
        torch.randn(2, 1024),
        torch.randn(2, 1024),
        torch.randn(2, 1024),
        torch.randn(2, 1024),
        labels,
    )
    total, values = loss_fn(
        make_args(lambda_task=0.5, lambda_gd=4.0),
        features,
    )
    assert values["task"].item() == 2.0
    assert values["gd"].item() == 3.0
    assert total.item() == 13.0


def test_cli_defaults():
    parser = add_gradient_distillation_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.task_temperature == 0.07
    assert defaults.gd_temperature == 0.07
    assert defaults.lambda_task == 1.0
    assert defaults.lambda_gd == 1.0


def test_gradients_reach_projectors_and_prompts_not_teacher():
    torch.manual_seed(11)
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat", "dog"),
        teacher=object(),
    )
    teacher_photo = torch.randn(4, 1024, requires_grad=True)
    teacher_sketch = torch.randn(4, 1024, requires_grad=True)
    batch = (
        torch.randn(4, 512),
        torch.randn(4, 512),
        teacher_photo,
        teacher_sketch,
        torch.tensor([0, 1, 0, 1]),
    )
    total, _ = loss_fn(model.cfg, model(batch))
    total.backward()

    assert model.photo_gd_projector.projection.weight.grad is not None
    assert model.photo_gd_projector.projection.bias.grad is not None
    assert model.sketch_gd_projector.projection.weight.grad is not None
    assert model.sketch_gd_projector.projection.bias.grad is not None
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
        raise AssertionError("The train-time GD projector was used at inference.")

    monkeypatch.setattr(model.photo_gd_projector, "forward", fail_if_called)
    monkeypatch.setattr(model.sketch_gd_projector, "forward", fail_if_called)
    output = model.extract_feature(torch.randn(2, 512), "sketch")
    assert output.shape == (2, 512)


def test_lightning_checkpoint_state_and_hyperparameters(monkeypatch):
    monkeypatch.setattr("src.model._load_clip_model", lambda _name: FakeCLIP())
    monkeypatch.setattr("src.model._load_teacher", lambda _args: object())
    args = make_args(backbone="ViT-B/32")
    model = ZS_SBIR(args, classnames=("cat",))

    assert model.hparams["lambda_task"] == 1.0
    assert model.hparams["lambda_gd"] == 1.0
    assert model.hparams["task_temperature"] == 0.07
    assert model.hparams["gd_temperature"] == 0.07
    assert model.hparams["projector_layout"] == "separate"
    assert model.hparams["projector_input_dim"] == 512
    assert model.hparams["projector_output_dim"] == 1024
    assert "model.photo_gd_projector.projection.weight" in model.state_dict()
    assert "model.sketch_gd_projector.projection.weight" in model.state_dict()
