import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from src.joint_geometry import joint_geometry_loss, geometry_diagnostics
from src.losses import loss_fn, relational_kd_loss, image_text_kd_loss
from src.model import _load_teacher, default_teacher_cache_path

smoke = runpy.run_path(str(Path(__file__).with_name("geometry_smoke.py")))
torch.set_num_threads(2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_twelve_layer_standalone_training(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    smoke["run_smoke"](device)


def test_exact_geometry_in_different_widths_and_shared_rotation():
    torch.manual_seed(9)
    sketch, photo = torch.randn(5, 8), torch.randn(5, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    ts, tp = F.pad(sketch @ rotation, (0, 7)), F.pad(photo @ rotation, (0, 7))
    loss, parts = joint_geometry_loss(sketch, photo, ts, tp)
    assert loss < 1e-12
    assert all(v < 1e-12 for v in parts.values())


def test_cross_block_detects_independent_modality_rotation():
    torch.manual_seed(9)
    ts, tp = torch.randn(5, 8), torch.randn(5, 8)
    ss, sp = ts.clone(), -tp.clone()
    _, parts = joint_geometry_loss(ss, sp, ts, tp)
    assert parts["joint_ss"] < 1e-12 and parts["joint_pp"] < 1e-12
    assert parts["joint_sp"] > 0.1


def test_signed_cosines_and_block_means_match_reference():
    ss = torch.tensor([[1., 0], [-1., 0], [0., 1]])
    sp = torch.tensor([[0., 1], [1., 0], [0., -1]])
    ts = torch.tensor([[1., 0, 0], [0., 1, 0], [0., 0, 1]])
    tp = torch.tensor([[1., 0, 0], [0., -1, 0], [0., 0, 1]])
    gs = torch.cat([ss, sp]) @ torch.cat([ss, sp]).t()
    gt = torch.cat([ts, tp]) @ torch.cat([ts, tp]).t()
    error = (gs - gt).square()
    mask = ~torch.eye(3, dtype=torch.bool)
    cross = error[:3, 3:].mean()
    same = (error[:3, :3][mask].mean() + error[3:, 3:][mask].mean()) / 2
    loss, parts = joint_geometry_loss(ss, sp, ts, tp, 0.6)
    torch.testing.assert_close(loss, 0.6 * cross + 0.4 * same)
    torch.testing.assert_close(parts["joint_sp"], cross)
    assert error[:3, 3:].diag().sum() > 0  # cross diagonal cannot be dropped
    for alpha, expected in [(0., same), (1., cross)]:
        torch.testing.assert_close(joint_geometry_loss(ss, sp, ts, tp, alpha)[0], expected)


def test_collapsed_output_penalized_and_reported():
    teacher = torch.eye(4)
    student = torch.ones(4, 7)
    loss, parts = joint_geometry_loss(student, student, teacher, teacher)
    assert loss > 0.5
    stats = geometry_diagnostics(student, student, teacher, teacher, torch.tensor([0,1,2,3]))
    assert stats["student_sketch_variance"] == 0
    assert stats["student_photo_variance"] == 0
    assert stats["student_sp_std"] < 1e-6
    assert stats["teacher_sketch_variance"] > 0
    assert stats["student_centroid_gap"] == 0  # zero modality gap alone is NOT success


def test_teacher_stop_gradient_and_both_modalities_receive_gradients():
    torch.manual_seed(42)
    ss, sp = [torch.randn(4, 5, requires_grad=True) for _ in range(2)]
    ts, tp = [torch.randn(4, 9, requires_grad=True) for _ in range(2)]
    before = ts.detach().clone(), tp.detach().clone()
    joint_geometry_loss(ss, sp, ts, tp)[0].backward()
    assert ss.grad.abs().sum() > 0 and sp.grad.abs().sum() > 0
    assert ts.grad is None and tp.grad is None
    assert torch.equal(ts, before[0]) and torch.equal(tp, before[1])


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weight(alpha):
    x = torch.randn(4, 8)
    with pytest.raises(ValueError, match="joint_cross_weight"):
        joint_geometry_loss(x, x, x, x, alpha)


def test_invalid_shapes_and_singleton():
    x = torch.randn(4, 8)
    for features in [(x[:1],)*4, (x, x[:3], x, x), (x, x[:, :4], x, x),
                     (x, x, x[:, :4], x), (x[:,:,None],)*4, (x[:, :0],)*4]:
        with pytest.raises(ValueError):
            joint_geometry_loss(*features)


def test_single_class_diagnostics_have_explicit_zero_negative_count():
    x, y = torch.randn(4, 5), torch.randn(4, 7)
    stats = geometry_diagnostics(x, x, y, y, torch.zeros(4, dtype=torch.long))
    assert stats["different_class_count"] == 0
    assert stats["same_class_count"] == 16
    assert all(torch.isfinite(v) for v in stats.values())


def test_toy_embedding_optimization_improves_geometry_and_cross_modal_separation():
    torch.manual_seed(42)
    labels = torch.tensor([0, 1, 2, 3])
    targets = torch.eye(4)
    ss, sp = [torch.nn.Parameter(torch.randn(4, 6)) for _ in range(2)]
    optimizer = torch.optim.SGD([ss, sp], lr=2.0)
    first = joint_geometry_loss(ss, sp, targets, targets)[0].item()
    for _ in range(160):
        optimizer.zero_grad()
        loss = joint_geometry_loss(ss, sp, targets, targets)[0]
        loss.backward()
        optimizer.step()
    stats = geometry_diagnostics(ss, sp, targets, targets, labels)
    assert loss < first * 0.1
    assert stats["student_sp_same_class"] > stats["student_sp_different_class"] + 0.5


def test_disabled_objective_reproduces_main_loss_and_all_prompt_gradients():
    smoke["seed_everything"](42)
    args = smoke["config"](lambda_domain=3., lambda_modality=1., lambda_joint_geometry=0.)
    student = smoke["make_student"](args, "cpu").train()
    batch = (torch.randn(4,3,16,16), torch.randn(4,3,16,16),
             torch.randn(4,1024), torch.randn(4,1024), torch.tensor([0,1,0,1]))
    features = student(batch)
    p, s, tp, ts, _, st, pt, tt_s, tt_p = features
    expected = (args.lambda_domain * relational_kd_loss(s,p,ts,tp,args.kd_temperature)
                + args.lambda_modality * (
                    image_text_kd_loss(p,pt,tp,tt_p,args.photo_text_kd_temperature)
                    + image_text_kd_loss(s,st,ts,tt_s,args.sketch_text_kd_temperature)))
    expected.backward()
    grads = {n: p.grad.clone() for n,p in student.named_parameters() if p.requires_grad}
    student.zero_grad(set_to_none=True)
    actual, parts = loss_fn(args, student(batch))
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(parts) == {"domain_kd", "modality_kd"}
    for n,p in student.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], rtol=0, atol=0)


def test_joint_only_teacher_loading_and_inactive_teacher_rejection(monkeypatch):
    import src.model as module
    args = smoke["config"]()
    calls = []
    monkeypatch.setattr(module.open_clip, "create_model", lambda *a,**kw: calls.append(1) or torch.nn.Linear(2,2))
    teacher = _load_teacher(args)
    assert len(calls) == 1 and teacher is not None
    assert not any(p.requires_grad for p in teacher.parameters())
    x, t = torch.randn(4,8), torch.randn(4,12)
    features = (x,x,t,t,False,None,None,None,None)
    with pytest.raises(RuntimeError, match="teacher global"):
        loss_fn(args, features)
    args.lambda_joint_geometry = 0
    assert _load_teacher(args) is None


def test_reuse_main_global_cache_and_config_identity(tmp_path, monkeypatch):
    import src.model as module
    args = smoke["config"](
        root=str(tmp_path), dataset="sketchy_2", teacher_cache_dir=str(tmp_path),
        teacher_pretrain_epochs=0, teacher_n_ctx_visual=10, teacher_prompt_depth=12,
        teacher_prompt_std=0.02, teacher_prompt_seed=42,
        teacher_prompt_gradient_checkpointing=True, teacher_prompt_lr=0.03,
        teacher_momentum=0.9, teacher_weight_decay=0.001,
        teacher_pretrain_batch_size=64, lambda_teacher_retrieval=1.5,
        teacher_triplet_margin=0.2, teacher_scheduler_step_size=5, teacher_scheduler_gamma=0.1,
    )
    dataset = SimpleNamespace(max_size=224, all_categories=["cat","dog"],
        all_sketches_path=[str(tmp_path / f"s{i}.png") for i in range(4)],
        all_photo_paths=[str(tmp_path / f"p{i}.png") for i in range(4)])
    def set_features(sketch, photo):
        dataset.teacher_sketch_features, dataset.teacher_photo_features = sketch, photo
    dataset.set_teacher_features = set_features
    main_path = default_teacher_cache_path(args, dataset)
    args.lambda_joint_geometry = 0
    assert default_teacher_cache_path(args, dataset) == main_path
    args.lambda_joint_geometry = 1
    student = smoke["make_student"](args, "cpu")
    args.teacher_cache_path = main_path
    metadata = student._teacher_cache_metadata(dataset)
    ts, tp = torch.randn(4,1024).half(), torch.randn(4,1024).half()
    torch.save(dict(metadata=metadata, teacher_sketch_features=ts,
                    teacher_photo_features=tp, teacher_sketch_text=torch.randn(2,1024),
                    teacher_photo_text=torch.randn(2,1024)), main_path)
    monkeypatch.setattr(module.open_clip, "create_model", lambda *a,**kw: pytest.fail("teacher reloaded"))
    assert _load_teacher(args) is None
    student.persistent_teacher_cache = True
    student.cache_teacher_features(dataset, None, None, 2, 0, False)
    assert torch.equal(dataset.teacher_sketch_features, ts)
    assert torch.equal(dataset.teacher_photo_features, tp)
    assert student.teacher_active and student._teacher is None
    loss, _ = loss_fn(args, student((torch.randn(4,3,16,16),torch.randn(4,3,16,16),
                                     tp,ts,torch.tensor([0,1,0,1]))))
    loss.backward()
    assert student.photo_visual_prompt.compound_prompts[-1].grad.abs().sum() > 0
