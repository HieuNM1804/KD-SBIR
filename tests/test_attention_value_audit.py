import importlib.util
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from clip.model import CLIP, convert_weights
from src.attention_diagnostics import Encoder
from src.model import IndependentVisualPromptLearner
from src.teacher_prompts import TeacherPromptController

spec = importlib.util.spec_from_file_location(
    "value_audit",
    Path(__file__).resolve().parents[1] / "test/kaggle_attention_value_audit.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def tiny_student(device="cpu"):
    model = (
        CLIP(32, 16, 3, 64, 4, 16, 128, 64, 1, 1)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    if device == "cuda":
        convert_weights(model)
    prompts = {
        m: IndependentVisualPromptLearner(3, 64, 42, 3)
        .to(device)
        .eval()
        .requires_grad_(False)
        for m in ("photo", "sketch")
    }
    return Encoder(model, prompts=prompts)


def tiny_teacher(device="cpu"):
    from open_clip.transformer import VisionTransformer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = VisionTransformer(16, 2, 96, 4, 3, 2, output_dim=48)

        def encode_image(self, images):
            return self.visual(images)

    model = Model().to(device).eval().requires_grad_(False)
    if device == "cuda":
        model.half()
    controller = (
        TeacherPromptController(model.visual, 3, 3, 0.02, 42)
        .eval()
        .requires_grad_(False)
    )
    return Encoder(model, controller=controller, openclip=True)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        torch.use_deterministic_algorithms(True)

    def test_reconstruction_real_student_teacher_raw_tuned(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            student = tiny_student(device)
            teacher = tiny_teacher(device)
            for encoder in [
                student,
                Encoder(student.model),
                teacher,
                Encoder(teacher.model, openclip=True),
            ]:
                for modality in ("photo", "sketch"):
                    image = torch.randn(1, 3, 16, 16)
                    with torch.no_grad():
                        reference = encoder.encode(image, modality).cpu().numpy()[0]
                    emb, data = audit.capture(encoder, image, modality, [1, 2, 3])
                    np.testing.assert_allclose(emb, reference, atol=0.003, rtol=0.003)
                    for d in data.values():
                        self.assertLess(d["relative_error"], 0.004)
                        np.testing.assert_allclose(
                            d["group_mass"].sum(-1), 1, atol=0.001
                        )
                        self.assertTrue((d["cancellation_ratio"] <= 1.0001).all())
                        if encoder.prompts is None and encoder.controller is None:
                            self.assertEqual(np.abs(d["group_vectors"][:, 2]).sum(), 0)
                    self.assertTrue(
                        all(
                            not b.attn._forward_hooks and not b.attn._forward_pre_hooks
                            for b in encoder.visual.transformer.resblocks
                        )
                    )
                    self.assertTrue(
                        all(p.grad is None for p in encoder.model.parameters())
                    )

    def test_nonzero_output_bias_and_token_order(self):
        for batch_first in (False, True):
            m = torch.nn.MultiheadAttention(12, 3, batch_first=batch_first).eval()
            with torch.no_grad():
                m.out_proj.bias.fill_(0.7)
            x = torch.randn((1, 7, 12) if batch_first else (7, 1, 12))
            with torch.no_grad():
                output = m(x, x, x, need_weights=True, average_attn_weights=False)
            d = audit.decompose_cls_attention(m, (x, x, x), {}, output, 4)
            expected = output[0][0, 0].detach().numpy()
            np.testing.assert_allclose(
                d["group_vectors"].sum((0, 1)) + d["out_bias"], expected, atol=1e-6
            )
            a = d["attention"]
            np.testing.assert_allclose(
                d["group_mass"][:, 2], a[:, 5:].sum(-1), atol=1e-7
            )
            m.add_zero_attn = True
            with self.assertRaises(ValueError):
                audit.decompose_cls_attention(m, (x, x, x), {}, output, 4)

    def test_probe_metrics_and_empty_prompt(self):
        q = np.eye(3, dtype=np.float32)
        g = np.repeat(q, 2, axis=0)
        labels = ["cat", "dog", "ant"]
        gl = np.repeat(labels, 2)
        m = audit.metrics(q, g, labels, gl)
        self.assertEqual(m["top1"], 1)
        self.assertAlmostEqual(m["P5"], 0.4)
        self.assertAlmostEqual(m["positive_negative_margin"], 1)
        m = audit.metrics(q * 0, g, labels, gl)
        self.assertEqual(m["valid_queries"], 0)
        self.assertIsNone(m["P5"])

    def test_small_end_to_end(self):
        import json
        from PIL import Image
        from src.dataset import normal_transform

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "student.ckpt"
            ckpt.write_bytes(b"fixture")
            teacher_cache = root / "teacher.pt"
            torch.save(
                {
                    "metadata": {
                        "dataset": "sketchy_2",
                        "classnames": ["cat", "dog"],
                        "teacher_model": "fixture",
                    }
                },
                teacher_cache,
            )
            for cls in ["cat", "dog"]:
                for modality in ["sketch", "photo"]:
                    folder = root / modality / cls
                    folder.mkdir(parents=True)
                    for i in range(2):
                        Image.fromarray(
                            np.random.default_rng(i).integers(
                                0, 256, (16, 16, 3), dtype=np.uint8
                            )
                        ).save(folder / f"{i}.png")

            previous = root / "previous.json"
            previous.write_text(
                json.dumps(
                    {
                        "dataset": "sketchy_2",
                        "queries": [str(root / "sketch/cat/0.png")],
                        "gallery": [
                            str(root / "photo/cat/0.png"),
                            str(root / "photo/dog/0.png"),
                        ],
                        "common_pairs_gallery_indices": {
                            "0": {"same_class": 0, "different_class": 1}
                        },
                        "models": {"KD student": {}, "Teacher tuned": {}},
                    }
                )
            )

            def load_student(model, *args):
                encoder = tiny_student("cuda" if torch.cuda.is_available() else "cpu")
                return Encoder(encoder.model), encoder, {"fixture": True}

            def load_teacher(*args):
                return tiny_teacher("cuda" if torch.cuda.is_available() else "cpu"), {
                    "fixture": True
                }

            # Keep numerical pipeline / archive / one real plot, omit the many
            # per-image panels here (covered separately below).
            with patch.multiple(
                audit,
                PROJECT=Path.cwd(),
                ROOT=root,
                CKPT=ckpt,
                TEACHER_CACHE=teacher_cache,
                PREVIOUS_AUDIT=previous,
                N_CLASSES=2,
                IMAGES_PER_CLASS=2,
                OUT_ROOT=root,
            ), patch(
                "src.model._load_clip_model",
                side_effect=lambda *_: tiny_student().model,
            ), patch(
                "src.attention_diagnostics.student_encoders", side_effect=load_student
            ), patch(
                "src.attention_diagnostics.teacher_encoder", side_effect=load_teacher
            ), patch(
                "src.dataset.normal_transform", return_value=normal_transform(16)
            ), patch.object(
                audit, "plot_image"
            ), patch(
                "IPython.display.display"
            ):
                out = audit.main()
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["models"]), 4)
            self.assertTrue(Path(str(out) + ".zip").is_file())
            self.assertTrue((out / "group_probe.csv").is_file())
            self.assertTrue((out / "fixed_pair_cosines.csv").is_file())
            with np.load(out / "Teacher_tuned_probe_features.npz") as a:
                self.assertEqual(a["L4_group_vectors"].shape[0], 8)
                self.assertEqual(a["L4_group_vectors"].shape[2], 3)

    def test_plot_and_npz(self):
        from PIL import Image

        encoder = tiny_student()
        _, data = audit.capture(encoder, torch.randn(1, 3, 16, 16), "photo", [1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            audit.plot_image(
                out, "test", encoder, Image.new("RGB", (16, 16), "white"), data
            )
            self.assertEqual(len(list(out.glob("*.png"))), 5)
            with np.load(out / "test_raw.npz") as a:
                self.assertIn("L3_out_bias", a.files)


if __name__ == "__main__":
    unittest.main()
