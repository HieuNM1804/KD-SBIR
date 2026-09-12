import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import numpy as np
from PIL import Image
from clip.model import CLIP, convert_weights
from src.model import IndependentVisualPromptLearner
from src.teacher_prompts import TeacherPromptController
from src.attention_diagnostics import (
    Encoder,
    pair_attribution,
    native_attention,
    rollout,
    student_encoders,
    teacher_encoder,
)


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def student(self, device):
        model = (
            CLIP(32, 16, 3, 64, 4, 77, 49408, 64, 1, 1)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        if device == "cuda":
            convert_weights(model)
        prompts = {
            m: IndependentVisualPromptLearner(3, 64, 42 + i, 3)
            .to(device)
            .eval()
            .requires_grad_(False)
            for i, m in enumerate(("photo", "sketch"))
        }
        return Encoder(model, prompts=prompts)

    def test_pair_depends_on_partner_and_preserves_forward(self):
        from open_clip.transformer import VisionTransformer

        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            student = self.student(device)
            model = torch.nn.Module()
            model.visual = VisionTransformer(16, 2, 96, 4, 3, 2, output_dim=48)
            model.to(device).eval().requires_grad_(False)
            if device == "cuda":
                model.half()
            controller = (
                TeacherPromptController(model.visual, 3, 3, 0.02, 42)
                .eval()
                .requires_grad_(False)
            )
            teacher = Encoder(model, controller=controller, openclip=True)
            for encoder in (student, teacher):
                x = torch.randn(1, 3, 16, 16)
                y = torch.randn_like(x)
                other = torch.randn_like(x)
                with torch.no_grad():
                    expected = float(
                        (encoder.encode(x, "sketch") * encoder.encode(y, "photo")).sum()
                    )
                maps, score = pair_attribution(encoder, x, y)
                changed, _ = pair_attribution(encoder, x, other)
                self.assertAlmostEqual(expected, score, places=6)
                self.assertEqual(maps[0].shape, (encoder.grid, encoder.grid))
                self.assertGreater(maps[0].abs().sum().item(), 0)
                self.assertFalse(torch.allclose(maps[0], changed[0], atol=1e-8))
                self.assertTrue(all(p.grad is None for p in encoder.model.parameters()))
                with torch.no_grad():
                    actual = float(
                        (encoder.encode(x, "sketch") * encoder.encode(y, "photo")).sum()
                    )
                self.assertEqual(expected, actual)
                self.assertEqual(
                    len(encoder.visual.transformer.resblocks[-1].ln_1._forward_hooks), 0
                )
                for method in ("last", "rollout"):
                    heat = native_attention(encoder, x, "sketch", method)
                    self.assertEqual(heat.shape, maps[0].shape)
                    self.assertTrue(torch.isfinite(heat).all())
                self.assertTrue(
                    all(
                        len(b.attn._forward_hooks) == 0
                        for b in encoder.visual.transformer.resblocks
                    )
                )

    def test_rollout_order_and_prompt_reset(self):
        a = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
        b = torch.tensor([[0.7, 0.1, 0.2], [0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
        aa = (a + torch.eye(3)) / 2
        bb = (b + torch.eye(3)) / 2
        actual = rollout([a[None], b[None]], 1, 0)
        self.assertTrue(torch.allclose(actual, (bb @ aa)[0, 1:2]))
        aa[2] = 0
        self.assertTrue(
            torch.allclose(rollout([a[None], b[None]], 1, 2), (bb @ aa)[0, 1:2])
        )

    def test_checkpoint_strict_loading(self):
        student = self.student("cpu")
        state = {
            "model.clip_model." + k: v for k, v in student.model.state_dict().items()
        }
        for m, learner in student.prompts.items():
            state.update(
                {
                    f"model.{m}_visual_prompt." + k: v
                    for k, v in learner.state_dict().items()
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.ckpt"
            torch.save({"state_dict": state}, path)
            _, loaded, info = student_encoders(student.model, path)
            self.assertEqual(info["prompts"]["photo"]["depth"], 3)
            self.assertTrue(
                torch.equal(loaded.prompts["photo"].ctx, student.prompts["photo"].ctx)
            )
            del state["model.photo_visual_prompt.ctx"]
            torch.save({"state_dict": state}, path)
            with self.assertRaises(ValueError):
                student_encoders(student.model, path)
            with self.assertRaises(FileNotFoundError):
                student_encoders(student.model, Path(directory) / "missing.ckpt")

    def test_teacher_cache_required_and_restored(self):
        from open_clip.transformer import VisionTransformer
        from src.model import (
            DFN5B_MODEL,
            DFN5B_PRETRAINED,
            TEACHER_CACHE_FORMAT_VERSION,
        )

        model = torch.nn.Module()
        model.visual = VisionTransformer(16, 2, 96, 4, 3, 2, output_dim=48)
        prompts = TeacherPromptController(model.visual, 3, 3, 0.02, 42)
        meta = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "sketchy_2",
            "max_size": 224,
            "pretrain_epochs": 1,
            "teacher_model": DFN5B_MODEL,
            "teacher_pretrained": DFN5B_PRETRAINED,
            "teacher_n_ctx_visual": 3,
            "teacher_prompt_depth": 3,
            "teacher_prompt_std": 0.02,
            "teacher_prompt_seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            with self.assertRaises(FileNotFoundError):
                teacher_encoder(str(path), "tuned", "sketchy_2", torch.device("cpu"))
            torch.save(
                {"metadata": meta, "teacher_prompt_state_dict": prompts.state_dict()},
                path,
            )
            with patch("open_clip.create_model", return_value=model):
                encoder, info = teacher_encoder(
                    str(path), "tuned", "sketchy_2", torch.device("cpu")
                )
            for k, v in prompts.state_dict().items():
                self.assertTrue(torch.equal(v, encoder.controller.state_dict()[k]))
            self.assertEqual(info["mode"], "tuned")
            with self.assertRaises(ValueError):
                teacher_encoder(str(path), "tuned", "sketchy_1", torch.device("cpu"))

    def test_render_four_rows(self):
        import matplotlib.pyplot as plt
        from src.visualize_attention import build_composite

        image = Image.new("RGB", (32, 32), "white")
        maps = [[[torch.randn(4, 4), torch.randn(4, 4)]] for _ in range(3)]
        fig = build_composite(
            image,
            [image],
            maps,
            ["Base CLIP", "KD student", "Teacher DFN5B"],
            [[0.1], [0.2], [0.3]],
            [1],
            "test",
            "pair_grad",
            0.55,
        )
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(fig.axes[-1].get_ylabel(), "Teacher DFN5B")
        fig.savefig(Path(tempfile.gettempdir()) / "kd_sbir_pair_viz_test.png")
        plt.close(fig)

    def test_pipeline_writes_teacher_scores_and_maps(self):
        import argparse
        import json
        from src.visualize_attention import run_visualisation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        student = self.student(device)
        teacher = self.student(device)
        base = Encoder(student.model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for modality in ("photo", "sketch"):
                folder = root / modality / "cabin"
                folder.mkdir(parents=True)
                for i in range(2):
                    pixels = np.random.default_rng(i).integers(
                        0, 256, (16, 16, 3), dtype=np.uint8
                    )
                    Image.fromarray(pixels).save(folder / f"{i}.png")
            args = argparse.Namespace(
                root=str(root),
                dataset="sketchy_2",
                seed=42,
                output_dir=str(root / "output"),
                max_size=16,
                backbone="test",
                ckpt_path="explicit.ckpt",
                teacher_cache_path="explicit.pt",
                teacher_mode="tuned",
                student_only=False,
                method="pair_grad",
                classes=["cabin"],
                test_batch_size=2,
                sketches_per_class=1,
                top_k=2,
                pairs_per_figure=2,
                alpha=0.55,
            )
            with patch(
                "src.visualize_attention._load_clip_model", return_value=student.model
            ), patch(
                "src.visualize_attention.student_encoders",
                return_value=(base, student, {}),
            ), patch(
                "src.visualize_attention.teacher_encoder",
                return_value=(teacher, {"mode": "tuned"}),
            ), patch(
                "src.visualize_attention.UNSEEN_CLASSES", {"sketchy_2": ["cabin"]}
            ):
                run_visualisation(args)
            output = root / "output"
            report = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(report["pairs"]), 2)
            for pair in report["pairs"]:
                self.assertEqual(len(pair["models"]), 3)
                self.assertIn("Teacher DFN5B (tuned)", pair["models"])
            with np.load(next(output.glob("*.npz"))) as maps:
                self.assertIn("pair0_model2_sketch", maps.files)
                self.assertIn("pair1_model2_photo", maps.files)
            self.assertTrue((root / "output.zip").is_file())


if __name__ == "__main__":
    unittest.main()
