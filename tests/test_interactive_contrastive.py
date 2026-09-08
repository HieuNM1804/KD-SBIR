import math
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.losses import (
    interactive_contrastive_loss,
    loss_fn,
    multi_positive_contrastive_loss,
)
from src.model import CustomCLIP, ICLProjector, ZS_SBIR
from src.train import add_interactive_contrastive_args


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
        "icl_temperature": 0.07,
        "lambda_icl": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_multi_positive_loss_counts_every_same_class_candidate_as_positive():
    anchors = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    loss = multi_positive_contrastive_loss(
        anchors,
        candidates,
        torch.tensor([0]),
        torch.tensor([0, 0, 1]),
        logit_scale=10.0,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_multi_positive_loss_is_invariant_to_candidate_order():
    anchors = torch.randn(5, 8)
    candidates = torch.randn(7, 8)
    anchor_labels = torch.tensor([0, 1, 0, 2, 1])
    candidate_labels = torch.tensor([2, 0, 1, 0, 2, 1, 1])
    order = torch.tensor([6, 2, 4, 0, 5, 3, 1])

    reference = multi_positive_contrastive_loss(
        anchors,
        candidates,
        anchor_labels,
        candidate_labels,
        logit_scale=4.0,
    )
    permuted = multi_positive_contrastive_loss(
        anchors,
        candidates[order],
        anchor_labels,
        candidate_labels[order],
        logit_scale=4.0,
    )
    assert permuted.item() == pytest.approx(reference.item(), abs=1e-6)


def test_multi_positive_loss_rejects_anchor_without_positive():
    with pytest.raises(ValueError, match="at least one positive"):
        multi_positive_contrastive_loss(
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.tensor([0, 2]),
            torch.tensor([0, 1]),
            logit_scale=1.0,
        )


def test_interactive_loss_uses_cross_domain_teacher_features(monkeypatch):
    calls = []

    def fake_loss(anchors, candidates, *_args):
        calls.append((anchors, candidates))
        return anchors.new_tensor(2.0 if len(calls) == 1 else 4.0)

    monkeypatch.setattr("src.losses.multi_positive_contrastive_loss", fake_loss)
    photo = torch.randn(2, 4)
    sketch = torch.randn(2, 4)
    teacher_photo = torch.randn(2, 4)
    teacher_sketch = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    total, values = interactive_contrastive_loss(
        photo,
        sketch,
        teacher_photo,
        teacher_sketch,
        labels,
        logit_scale=1.0,
    )

    assert calls[0] == (sketch, teacher_photo)
    assert calls[1] == (photo, teacher_sketch)
    assert values["icl_sketch_to_photo"].item() == 2.0
    assert values["icl_photo_to_sketch"].item() == 4.0
    assert total.item() == 3.0


def test_loss_fn_applies_icl_weight():
    labels = torch.tensor([0, 1, 0, 1])
    features = (
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
        labels,
        torch.tensor(2.0),
    )
    raw, _ = interactive_contrastive_loss(*features)
    total, logged = loss_fn(make_args(lambda_icl=2.5), features)
    assert total.item() == pytest.approx(2.5 * raw.item())
    assert logged["icl"].item() == pytest.approx(raw.item())


def test_icl_projector_has_expected_shape():
    projector = ICLProjector()
    assert projector(torch.randn(3, 512)).shape == (3, 1024)


def test_photo_and_sketch_projectors_are_separate():
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )

    assert model.photo_icl_projector is not model.sketch_icl_projector
    assert (
        model.photo_icl_projector.projection.weight.data_ptr()
        != model.sketch_icl_projector.projection.weight.data_ptr()
    )
    assert (
        model.photo_icl_projector.projection.bias.data_ptr()
        != model.sketch_icl_projector.projection.bias.data_ptr()
    )


def test_cli_defaults():
    parser = add_interactive_contrastive_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.icl_temperature == 0.07
    assert defaults.lambda_icl == 1.0


def test_gradients_reach_both_projectors_prompts_and_logit_scale_not_teacher():
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

    assert model.photo_icl_projector.projection.weight.grad is not None
    assert model.photo_icl_projector.projection.bias.grad is not None
    assert model.sketch_icl_projector.projection.weight.grad is not None
    assert model.sketch_icl_projector.projection.bias.grad is not None
    assert model.icl_logit_scale.grad is not None
    assert model.photo_visual_prompt.ctx.grad is not None
    assert model.sketch_visual_prompt.ctx.grad is not None
    assert teacher_photo.grad is None
    assert teacher_sketch.grad is None
    assert not any(
        parameter.requires_grad
        for parameter in model.clip_model.parameters()
    )


def test_initial_logit_scale_matches_requested_temperature():
    model = CustomCLIP(
        make_args(icl_temperature=0.2),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )
    assert model.icl_logit_scale.item() == pytest.approx(math.log(5.0))


def test_inference_bypasses_both_projectors(monkeypatch):
    model = CustomCLIP(
        make_args(),
        FakeCLIP(),
        classnames=("cat",),
        teacher=object(),
    )

    def fail_if_called(_features):
        raise AssertionError("A train-time ICL projector was used at inference.")

    monkeypatch.setattr(model.photo_icl_projector, "forward", fail_if_called)
    monkeypatch.setattr(model.sketch_icl_projector, "forward", fail_if_called)
    output = model.extract_feature(torch.randn(2, 512), "sketch")
    assert output.shape == (2, 512)


def test_lightning_checkpoint_state_and_hyperparameters(monkeypatch):
    monkeypatch.setattr("src.model._load_clip_model", lambda _name: FakeCLIP())
    monkeypatch.setattr("src.model._load_teacher", lambda _args: object())
    args = make_args(
        icl_temperature=0.1,
        lambda_icl=2.0,
        backbone="ViT-B/32",
    )
    model = ZS_SBIR(args, classnames=("cat",))

    assert model.hparams["lambda_icl"] == 2.0
    assert model.hparams["icl_temperature"] == 0.1
    assert model.hparams["projector_layout"] == "separate"
    assert model.hparams["positive_policy"] == "same_class_multi_positive"
    assert model.hparams["projector_input_dim"] == 512
    assert model.hparams["projector_output_dim"] == 1024
    assert "model.photo_icl_projector.projection.weight" in model.state_dict()
    assert "model.sketch_icl_projector.projection.weight" in model.state_dict()
    assert "model.icl_logit_scale" in model.state_dict()
