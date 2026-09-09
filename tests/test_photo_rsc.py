from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from clip.model import CLIP, convert_weights
from src.losses import loss_fn
from src.model import CustomCLIP, ZS_SBIR
from src.photo_rsc import photo_rsc_features, validate_rsc
from src.train import seed_everything


def config(**overrides):
    values = dict(
        seed=42, prompt_depth=3, n_ctx_visual=3, lambda_modality=1.0,
        lambda_domain=3.0, kd_temperature=0.07, photo_text_kd_temperature=0.15,
        sketch_text_kd_temperature=0.02, teacher_pretrain_epochs=0,
        teacher_cache_path="", rebuild_teacher_cache=False,
        student_rsc_prob=1.0, student_rsc_drop=0.25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def embeddings(device="cpu"):
    torch.manual_seed(12)
    values = [F.normalize(torch.randn(4, 16, device=device), dim=-1) for _ in range(4)]
    values[0].requires_grad_()
    values[1].requires_grad_()
    texts = [F.normalize(torch.randn(3, 16, device=device), dim=-1) for _ in range(4)]
    return (*values, True, *texts)


@pytest.mark.parametrize("prob,drop", [(-0.1, .1), (1.1, .1), (.5, -1), (.5, 1), (float('nan'), .1), (.5, float('inf'))])
def test_invalid_options(prob, drop):
    with pytest.raises(ValueError):
        validate_rsc(prob, drop)


@pytest.mark.parametrize("options,training", [({"student_rsc_prob": 0}, True), ({"student_rsc_drop": 0}, True), ({}, False)])
def test_disabled_is_identity_without_rng_consumption(options, training):
    original = embeddings()
    state = torch.random.get_rng_state()
    result, mask = photo_rsc_features(config(**options), original, training)
    assert result is original and mask.all()
    assert torch.equal(state, torch.random.get_rng_state())


def test_exact_gradient_ranking_and_only_photo_changed():
    args, original = config(), embeddings()
    reference_loss, _ = loss_fn(args, original)
    gradient = torch.autograd.grad(reference_loss, original[0])[0].abs()
    expected = gradient.argsort(dim=-1, descending=True, stable=True)[:, :4]
    result, mask = photo_rsc_features(args, original)
    assert (~mask).sum(1).eq(4).all()
    assert (~mask).gather(1, expected).all()
    assert original[0].grad is None and original[1].grad is None
    assert all(a is b for a, b in zip(result[1:], original[1:]))
    torch.testing.assert_close(result[0].norm(dim=-1), torch.ones(4))
    loss_fn(args, result)[0].backward()
    assert original[0].grad[~mask].eq(0).all()
    assert original[0].grad[mask].abs().sum() > 0
    assert original[1].grad.abs().sum() > 0


def test_equal_teacher_student_zero_gradient_does_not_mask():
    photo = F.normalize(torch.ones(4, 16), dim=-1).requires_grad_()
    features = (photo, photo.detach(), photo.detach(), photo.detach(), True, None, None, None, None)
    result, mask = photo_rsc_features(config(lambda_modality=0), features)
    assert result is features and mask.all()


def test_sparse_features_do_not_become_zero():
    original = list(embeddings())
    original[0] = F.one_hot(torch.arange(4), num_classes=16).float().requires_grad_()
    result, mask = photo_rsc_features(config(student_rsc_drop=.99), original)
    assert (result[0].norm(dim=-1) > 0).all()
    assert torch.isfinite(result[0]).all()


@pytest.mark.parametrize("weights", [(3., 0.), (0., 1.)])
def test_each_kd_objective_can_supply_probe(weights):
    original = embeddings()
    args = config(lambda_domain=weights[0], lambda_modality=weights[1])
    result, mask = photo_rsc_features(args, original)
    assert (~mask).any()
    loss_fn(args, result)[0].backward()
    assert torch.isfinite(original[0].grad).all()


def test_inactive_teacher_and_objectives_bypass_probe():
    original = embeddings()
    inactive = (*original[:4], False, *original[5:])
    assert photo_rsc_features(config(), inactive)[0] is inactive
    assert photo_rsc_features(config(lambda_domain=0, lambda_modality=0), original)[0] is original


def test_probability_is_per_photo_and_seed_reproducible():
    original = embeddings()
    args = config(student_rsc_prob=.5)
    torch.manual_seed(42)
    expected = torch.rand(4) < .5
    torch.manual_seed(42)
    _, first = photo_rsc_features(args, original)
    assert torch.equal((~first).any(1), expected)
    torch.manual_seed(42)
    _, second = photo_rsc_features(args, original)
    assert torch.equal(first, second)
    with torch.no_grad():
        assert photo_rsc_features(args, original)[0] is original


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_real_training_step_backward_and_inference(device):
    seed_everything(42)
    args = config()
    backbone = CLIP(32, 16, 3, 64, 4, 77, 49408, 64, 1, 1).to(device)
    if device == "cuda":
        convert_weights(backbone)
    student = CustomCLIP(args, backbone, ["cat", "dog"], teacher=None).to(device)
    student.teacher_active = True
    student._teacher_sketch_text = F.normalize(torch.randn(2, 1024, device=device), dim=-1)
    student._teacher_photo_text = F.normalize(torch.randn(2, 1024, device=device), dim=-1)
    batch = (
        torch.randn(4, 3, 16, 16, device=device), torch.randn(4, 3, 16, 16, device=device),
        torch.randn(4, 1024, device=device), torch.randn(4, 1024, device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    state_keys = tuple(student.state_dict())
    teacher_before = batch[2].clone(), batch[3].clone()
    # Exercise the actual Lightning training_step without loading real weights.
    wrapper = ZS_SBIR.__new__(ZS_SBIR)
    torch.nn.Module.__init__(wrapper)
    wrapper.args, wrapper.model = args, student
    logged = {}
    wrapper.log = lambda key, value, **kw: logged.update({key: value})
    wrapper.train()
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        loss = wrapper.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert 0 < logged["rsc_drop_fraction"] <= .25
    for name, param in student.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    for prompts in (student.photo_visual_prompt, student.sketch_visual_prompt):
        assert sum(p.grad.abs().sum() for p in prompts.parameters()) > 0
    assert torch.equal(batch[2], teacher_before[0]) and torch.equal(batch[3], teacher_before[1])
    assert tuple(student.state_dict()) == state_keys
    before = student.photo_visual_prompt.ctx.detach().clone()
    torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=.01).step()
    assert not torch.equal(before, student.photo_visual_prompt.ctx)
    student.eval()
    with torch.inference_mode():
        enabled = student.extract_feature(batch[0], "photo")
        args.student_rsc_prob = 0
        disabled = student.extract_feature(batch[0], "photo")
    assert torch.equal(enabled, disabled)
