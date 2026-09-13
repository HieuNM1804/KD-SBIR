import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from clip.model import CLIP, convert_weights
from src.attention_output_kd import (
    PatchOutputCapture,
    patch_attention_output,
    make_projector,
    feature_cosine_kd,
    relational_av_kd,
)
from src.attention_output_cache import validate_cache
from src.dataset import TrainDataset
from src.model import IndependentVisualPromptLearner, ZS_SBIR, DFN5B_OUTPUT_DIM


class AttentionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def student(self, device, width=64):
        model = (
            CLIP(32, 16, 3, width, 4, 16, 128, 64, 1, 1)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        if device == "cuda":
            convert_weights(model)
        prompts = IndependentVisualPromptLearner(3, width, 42, 3).to(device)
        return model, prompts

    def test_equivalence_to_explicit_per_head_avwo(self):
        for layout in (True, False):
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                attn = (
                    torch.nn.MultiheadAttention(32, 4, batch_first=layout)
                    .to(device)
                    .eval()
                )
                x = torch.randn(2, 8, 32, device=device, requires_grad=True)
                q = x if layout else x.transpose(0, 1)
                _, weights = attn(
                    q, q, q, need_weights=True, average_attn_weights=False
                )
                v = (
                    F.linear(x, attn.in_proj_weight[64:], attn.in_proj_bias[64:])
                    .reshape(2, 8, 4, 8)
                    .transpose(1, 2)
                )
                weighted = (weights[:, :, 0:1, 1:6] @ v[:, :, 1:6]).reshape(2, 32)
                expected = F.linear(weighted, attn.out_proj.weight, None)
                actual = patch_attention_output(attn, q, q, q, 5)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                grad1 = torch.autograd.grad(
                    expected.square().sum(), x, retain_graph=True
                )[0]
                grad2 = torch.autograd.grad(actual.square().sum(), x)[0]
                torch.testing.assert_close(grad1, grad2, atol=1e-6, rtol=1e-5)

    def test_capture_prompt_gradients_and_forward_unchanged(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            model, prompts = self.student(device, width=128)
            x = torch.randn(2, 3, 16, 16, device=device, dtype=model.dtype)
            ctx, deep = prompts()
            reference = model.visual(x, ctx, deep)
            with PatchOutputCapture(model.visual) as capture:
                output = model.visual(x, ctx, deep)
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            self.assertEqual(capture.values[0].shape, (2, 128))
            target = torch.randn(2, 96, device=device, requires_grad=True)
            proj = make_projector(128, 96, 5).to(device)
            dtype = torch.float16 if device == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device, dtype=dtype):
                loss = feature_cosine_kd(capture.values[0], target, proj)
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            for p in prompts.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()), 0)
            self.assertIsNone(target.grad)
            self.assertGreater(float(proj.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertFalse(
                model.visual.transformer.resblocks[-1].attn._forward_pre_hooks
            )

    def test_openclip_teacher_controller(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController

        visual = (
            VisionTransformer(16, 2, 96, 4, 3, 2, output_dim=48)
            .eval()
            .requires_grad_(False)
        )
        ctrl = (
            TeacherPromptController(visual, 3, 3, 0.02, 42).eval().requires_grad_(False)
        )
        x = torch.randn(2, 3, 16, 16)
        for forward in (lambda: ctrl(x, "sketch"), lambda: visual(x)):
            with torch.no_grad():
                reference = forward()
            with torch.no_grad(), PatchOutputCapture(visual) as cap:
                actual = forward()
            torch.testing.assert_close(reference, actual, atol=0, rtol=0)
            self.assertEqual(cap.values[0].shape, (2, 96))

    def test_projector_rng_and_cache_validation(self):
        state = torch.get_rng_state().clone()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        make_projector(64, 96, 123)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for before, after in zip(
            cuda_states, torch.cuda.get_rng_state_all() if cuda_states else []
        ):
            self.assertTrue(torch.equal(before, after))
        meta = {"version": 1}
        payload = {
            "metadata": meta,
            "sketch": torch.randn(2, 1280).half(),
            "photo": torch.randn(3, 1280).half(),
        }
        validate_cache(payload, meta, 2, 3)
        with self.assertRaises(ValueError):
            validate_cache(payload, {"version": 2}, 2, 3)
        payload["photo"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_cache(payload, meta, 2, 3)

    def test_dataset_pair_alignment_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("photo", "sketch"):
                (root / m / "cat").mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (16, 16), (i * 60, 0, 0)).save(
                        root / m / "cat" / f"{i}.png"
                    )
            ds = TrainDataset(
                SimpleNamespace(root=tmp, dataset="sketchy_2", seed=42, max_size=16)
            )
            sk = torch.arange(3).float().view(3, 1)
            ph = sk + 10
            ds.set_teacher_features(sk, ph)
            original = [ds[(2, i)] for i in range(3)]
            ds.set_attention_output_features(sk + 100, ph + 100)
            for i in range(3):
                current = ds[(2, i)]
                for old, new in zip(original[i][:4], current[:4]):
                    torch.testing.assert_close(old, new)
                torch.testing.assert_close(current[5], current[2] + 100)
                torch.testing.assert_close(current[6], current[3] + 100)

    def test_cache_build_restore_and_content_mismatch(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        from src.model import (
            DFN5B_MODEL,
            DFN5B_PRETRAINED,
            TEACHER_CACHE_FORMAT_VERSION,
        )
        from src.attention_output_cache import prepare_av_cache

        # Full DFN channel width with one tiny spatial block; no downloads.
        teacher = torch.nn.Module()
        teacher.visual = VisionTransformer(8, 4, 1280, 1, 20, 1, output_dim=1024).eval()
        prompts = TeacherPromptController(teacher.visual, 2, 1, 0.02, 42)
        meta = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "sketchy_2",
            "max_size": 8,
            "teacher_model": DFN5B_MODEL,
            "teacher_pretrained": DFN5B_PRETRAINED,
            "pretrain_epochs": 1,
            "teacher_n_ctx_visual": 2,
            "teacher_prompt_depth": 1,
            "teacher_prompt_std": 0.02,
            "teacher_prompt_seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("photo", "sketch"):
                (root / m / "cat").mkdir(parents=True)
                for i in range(2):
                    Image.new("RGB", (8, 8), (i * 80, 20, 50)).save(
                        root / m / "cat" / f"{i}.png"
                    )
            args = SimpleNamespace(
                root=tmp,
                dataset="sketchy_2",
                seed=42,
                max_size=8,
                teacher_cache_path=str(root / "teacher.pt"),
                av_cache_path="",
                workers=0,
                av_teacher_batch_size=2,
            )
            torch.save(
                {"metadata": meta, "teacher_prompt_state_dict": prompts.state_dict()},
                args.teacher_cache_path,
            )
            ds = TrainDataset(args)
            with patch(
                "open_clip.create_model",
                side_effect=lambda *a, **kw: teacher.to(kw["device"])
                .eval()
                .requires_grad_(False),
            ):
                prepare_av_cache(args, ds)
            self.assertEqual(ds.attention_sketch_features.shape, (2, 1280))
            self.assertFalse(ds.attention_sketch_features.requires_grad)
            restored = TrainDataset(args)
            with patch(
                "open_clip.create_model",
                side_effect=AssertionError("cache hit must not load teacher"),
            ):
                prepare_av_cache(args, restored)
            torch.testing.assert_close(
                ds.attention_sketch_features, restored.attention_sketch_features
            )
            Image.new("RGB", (8, 8), "red").save(root / "photo/cat/0.png")
            with self.assertRaises(ValueError):
                prepare_av_cache(args, restored)

    def test_training_step_ablation_paths_and_optimizer(self):
        # Real tiny prompted forward plus production training_step and losses;
        # avoid downloading large checkpoints in this numerical integration test.
        from src.losses import loss_fn
        from src.model import CustomCLIP

        for av, glob, objective in [(0, 0, "cosine"), (1, 0, "cosine"),
                                    (0, 1, "cosine"), (1, 1, "cosine"),
                                    (1, 0, "relational"), (1, 1, "relational")]:
            model, prompts = self.student("cpu")

            class Wrapper(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.clip_model = model
                    self.photo_visual_prompt = prompts
                    self.sketch_visual_prompt = IndependentVisualPromptLearner(
                        3, 64, 43, 3
                    )
                    if av and objective == "cosine":
                        self.av_projector = make_projector(64, 1280, 4)
                    if glob:
                        self.global_feature_projector = make_projector(32, 1024, 5)
                    self.stext = F.normalize(torch.randn(3, 32), dim=-1)
                    self.ttext = F.normalize(torch.randn(3, 1024), dim=-1)

                def forward(self, b):
                    p, s, tp, ts, _ = b
                    p = F.normalize(
                        self.clip_model.visual(p, *self.photo_visual_prompt()), dim=-1
                    )
                    s = F.normalize(
                        self.clip_model.visual(s, *self.sketch_visual_prompt()), dim=-1
                    )
                    return (
                        p,
                        s,
                        tp,
                        ts,
                        True,
                        self.stext,
                        self.stext,
                        self.ttext,
                        self.ttext,
                    )

            module = ZS_SBIR.__new__(ZS_SBIR)
            import pytorch_lightning as pl

            pl.LightningModule.__init__(module)
            module.model = Wrapper()
            module.lambda_av = av
            module.lambda_global_feature = glob
            module.av_objective = objective
            module.av_temperature = 0.07
            module.args = SimpleNamespace(
                seed=42,
                lambda_domain=3.0,
                lambda_modality=1.0,
                kd_temperature=0.07,
                lr=0.01,
                momentum=0.9,
                weight_decay=0.0005,
                photo_text_kd_temperature=0.15,
                sketch_text_kd_temperature=0.02,
            )
            batch = (
                torch.randn(3, 3, 16, 16),
                torch.randn(3, 3, 16, 16),
                torch.randn(3, 1024),
                torch.randn(3, 1024),
                torch.arange(3),
            )
            expected = (
                loss_fn(module.args, module(batch))[0] if not av and not glob else None
            )
            if av:
                batch += (torch.randn(3, 1280), torch.randn(3, 1280))
            with patch.object(module, "log"):
                loss = module.training_step(batch, 0)
            if expected is not None:
                torch.testing.assert_close(loss, expected, atol=0, rtol=0)
            loss.backward()
            for p in module.model.photo_visual_prompt.parameters():
                self.assertIsNotNone(p.grad)
            optimizer = module.configure_optimizers()[0][0]
            registered = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }
            if av and objective == "cosine":
                self.assertIn(id(module.model.av_projector.weight), registered)
            if objective == "relational":
                self.assertFalse(hasattr(module.model, "av_projector"))
            if glob:
                self.assertIn(
                    id(module.model.global_feature_projector.weight), registered
                )
            self.assertTrue(all(p.grad is None for p in model.parameters()))

            checkpoint = {}
            module.on_save_checkpoint(checkpoint)
            self.assertEqual(checkpoint["experiment_config"]["av_objective"], objective)

            if objective == "relational" and not glob:
                from src.av_gradient_audit import AVGradientAudit
                with tempfile.TemporaryDirectory() as tmp:
                    callback = AVGradientAudit()
                    callback.batch = batch
                    callback.indices = [0, 1, 2]
                    callback.records = []
                    callback.path = Path(tmp) / "audit.json"
                    trainer = SimpleNamespace(global_step=0, current_epoch=0)
                    before = [p.grad.clone() if p.grad is not None else None
                              for p in module.parameters()]
                    rng = torch.get_rng_state().clone()
                    callback.measure(trainer, module, "test")
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for p, old in zip(module.parameters(), before):
                        if old is None:
                            self.assertIsNone(p.grad)
                        else:
                            torch.testing.assert_close(p.grad, old, atol=0, rtol=0)
                    self.assertTrue(callback.path.is_file())
                    self.assertGreater(callback.records[0]["av_norm"], 0)

                    # Real Lightning lifecycle: diagnostics must leave the SGD
                    # update identical, and export start/first-step/end records.
                    from copy import deepcopy
                    from torch.utils.data import DataLoader, Dataset
                    import json

                    class FixedDataset(Dataset):
                        def __len__(self):
                            return 3

                        def __getitem__(self, key):
                            index = key[1] if isinstance(key, tuple) else key
                            return tuple(x[index] for x in batch)

                    results = []
                    for enabled in (False, True):
                        current = deepcopy(module)
                        current.zero_grad(set_to_none=True)
                        audit = AVGradientAudit()
                        trainer = pl.Trainer(
                            accelerator="cpu", devices=1, max_epochs=1,
                            limit_train_batches=1, limit_val_batches=0,
                            num_sanity_val_steps=0, logger=False,
                            enable_checkpointing=False, enable_progress_bar=False,
                            enable_model_summary=False, default_root_dir=tmp,
                            callbacks=[audit] if enabled else [],
                        )
                        trainer.fit(current, DataLoader(FixedDataset(), batch_size=3))
                        results.append(deepcopy(current.state_dict()))
                        if enabled:
                            report = json.loads(audit.path.read_text())
                            self.assertEqual([r["stage"] for r in report["measurements"]],
                                             ["start", "step_1", "end"])
                            trainer.save_checkpoint(str(Path(tmp) / "test.ckpt"))
                            saved = torch.load(Path(tmp) / "test.ckpt", weights_only=False)
                            self.assertEqual(saved["experiment_config"]["av_objective"], "relational")
                    for key in results[0]:
                        torch.testing.assert_close(results[0][key], results[1][key], atol=0, rtol=0)

    def test_relational_av_formula_gradients_and_invariance(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            sk = torch.randn(4, 16, device=device, requires_grad=True)
            ph = torch.randn(5, 16, device=device, requires_grad=True)
            ts = torch.randn(4, 24, device=device, requires_grad=True)
            tp = torch.randn(5, 24, device=device, requires_grad=True)
            with torch.autocast(device_type=device, enabled=device == "cuda", dtype=torch.float16):
                loss = relational_av_kd(sk, ph, ts, tp, 0.2)
            s = F.normalize(sk, dim=-1) @ F.normalize(ph, dim=-1).T / .2
            t = F.normalize(ts, dim=-1) @ F.normalize(tp, dim=-1).T / .2
            expected = .5 * (
                (t.softmax(-1) * (t.log_softmax(-1) - s.log_softmax(-1))).sum(-1).mean()
                + (t.T.softmax(-1) * (t.T.log_softmax(-1) - s.T.log_softmax(-1))).sum(-1).mean()
            )
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(loss, relational_av_kd(ph, sk, tp, ts, .2))
            loss.backward()
            self.assertGreater(sk.grad.norm().item(), 0)
            self.assertGreater(ph.grad.norm().item(), 0)
            self.assertIsNone(ts.grad)
            self.assertIsNone(tp.grad)
            # Embedding the same relations in a different width must give zero KL.
            zero = relational_av_kd(sk, ph, F.pad(sk, (0, 8)), F.pad(ph, (0, 8)), .2)
            self.assertLess(abs(zero.item()), 1e-5)
            for temperature in (0, -1, float("nan")):
                with self.assertRaises(ValueError):
                    relational_av_kd(sk, ph, ts, tp, temperature)


if __name__ == "__main__":
    unittest.main()
