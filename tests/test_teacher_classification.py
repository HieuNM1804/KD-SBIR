"""Metric, split, and end-to-end inference checks without pretrained weights."""

import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import torch
from torch import nn

from src.evaluate_teacher_classification import list_dataset, main, main_transform, parse_args
from src.teacher_classification_metrics import ClassificationMetrics
from src.teacher_classification_specs import TEACHERS, prompt_templates


class ClassificationMetricTests(unittest.TestCase):
    def test_confusion_macro_f1_nll_and_calibration_independently(self):
        probabilities = torch.tensor([[.9, .05, .05], [.2, .7, .1], [.1, .8, .1], [.2, .6, .2]])
        accumulator = ClassificationMetrics(["a", "b", "c"], bins=2)
        accumulator.update(probabilities[:2].log(), torch.tensor([0, 0]))
        accumulator.update(probabilities[2:].log(), torch.tensor([1, 2]))
        result, classes = accumulator.finalize()
        self.assertEqual(accumulator.confusion.tolist(), [[1, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + .5) / 3)
        self.assertAlmostEqual(result["nll"], -sum(math.log(p) for p in [.9, .2, .8, .2]) / 4)
        self.assertAlmostEqual(result["ece"], .25, places=6)
        self.assertEqual(result["top5_accuracy"], 1)
        self.assertEqual(result["topk"], 3)
        self.assertEqual(classes[2]["recall"], 0)
        self.assertEqual(result["confidence_thresholds"][-1]["retained"], 0)
        self.assertIsNone(result["confidence_thresholds"][-1]["accuracy"])

    def test_topk_hits_are_not_top1_hits(self):
        accumulator = ClassificationMetrics(list("abcdef"))
        logits = torch.tensor([[6., 5., 4., 3., 2., 1.], [6., 5., 4., 3., 2., 1.]])
        accumulator.update(logits, torch.tensor([1, 5]))
        result, _ = accumulator.finalize()
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["top5_accuracy"], .5)

    def test_pseudo_label_topk_uses_predicted_class_and_confidence(self):
        accumulator = ClassificationMetrics(["a", "b"])
        confidence = torch.linspace(.99, .6, 40)
        logits = torch.stack([confidence, 1 - confidence], dim=1).log()
        accumulator.update(logits, torch.tensor([0] * 16 + [1] * 24))
        result, _ = accumulator.finalize()
        retained = result["topk_per_predicted_class"][0]
        self.assertEqual(retained["retained"], 16)
        self.assertEqual(retained["coverage"], .4)
        self.assertEqual(retained["accuracy"], 1)
        self.assertEqual(result["predicted_classes"], 1)
        self.assertEqual(result["coverage_targets"][0]["retained"], 4)
        self.assertEqual(result["coverage_targets"][0]["accuracy"], 1)

    def test_invalid_outputs_fail_instead_of_silent_skips(self):
        accumulator = ClassificationMetrics(["a", "b"])
        with self.assertRaises(ValueError):
            accumulator.update(torch.tensor([[float("nan"), 1.]]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            accumulator.update(torch.zeros(1, 2), torch.tensor([2]))
        with self.assertRaises(ValueError):
            accumulator.finalize()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for modality in ("sketch", "photo"):
            for classname in ("cat", "dog", "bat"):
                folder = self.root / modality / classname
                folder.mkdir(parents=True)
                for i in range(4):
                    Image.new("RGB", (5, 5)).save(folder / f"{i}.png")

    def tearDown(self):
        self.temporary.cleanup()

    def test_seen_candidates_exclude_unseen_and_share_manifest(self):
        data = list_dataset(self.root, "quickdraw", max_per_class=2)
        self.assertEqual(data["candidate_classes"], ["cat", "dog"])
        self.assertEqual(len(data["records"]["sketch"]), 4)
        self.assertTrue(all("/bat/" not in p for p, _ in data["records"]["photo"]))
        self.assertEqual(data, list_dataset(self.root, "quickdraw", max_per_class=2))
        all_candidates = list_dataset(self.root, "quickdraw", candidate_scope="all")
        self.assertEqual(all_candidates["candidate_classes"], ["bat", "cat", "dog"])
        self.assertNotEqual(data["manifest_sha256"], all_candidates["manifest_sha256"])

    def test_missing_class_images_and_modalities_fail(self):
        for path in (self.root / "photo" / "cat").glob("*"):
            path.unlink()
        with self.assertRaises(ValueError):
            list_dataset(self.root, "quickdraw")

    def test_arguments_default_to_seen_and_templates_match_main(self):
        args = parse_args(["--models-dir", "models"])
        self.assertEqual(args.split, "seen")
        self.assertEqual(args.candidate_scope, "split")
        self.assertEqual(len(args.teachers), 5)
        self.assertEqual(prompt_templates("sketch", "modality"), ["a sketch of a {}."])
        self.assertEqual(prompt_templates("photo", "modality"), ["a photo of a {}."])


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.register_parameter("class_embedding", nn.Parameter(torch.zeros(2)))
        self.visual.conv1 = nn.Conv2d(3, 2, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.)))

    def encode_text(self, tokens):
        assert not torch.is_grad_enabled()
        assert not self.training and not any(p.requires_grad for p in self.parameters())
        return torch.nn.functional.one_hot(tokens, 2).float()

    def encode_image(self, images):
        assert not torch.is_grad_enabled()
        assert not self.training and not any(p.requires_grad for p in self.parameters())
        assert images.dtype == self.visual.conv1.weight.dtype
        return images[:, [0, 2]].mean(dim=(2, 3))


class InferenceTests(unittest.TestCase):
    def test_full_cli_produces_separate_photo_sketch_and_pseudo_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, weights, output = (directory / name for name in ("data", "models", "out"))
            for modality in ("photo", "sketch"):
                for name, color in (("cat", "red"), ("dog", "blue")):
                    folder = root / modality / name
                    folder.mkdir(parents=True)
                    Image.new("RGB", (9, 7), color).save(folder / "image.png")
            keys = ["dfn5b_h14", "openai_l14"]
            for key in keys:
                folder = weights / key
                folder.mkdir(parents=True)
                (folder / TEACHERS[key]["filename"]).touch()
            calls = []

            def factory(name, pretrained, precision, device, **configuration):
                calls.append((name, pretrained))
                self.assertTrue(Path(pretrained).is_file())
                self.assertEqual(precision, "fp32")
                self.assertTrue(configuration["force_quick_gelu"])
                self.assertEqual(configuration["image_resize_mode"], "squash")
                teacher = FakeTeacher()
                # Match mixed OpenCLIP parameter dtypes: first parameter FP32,
                # input convolution FP16. Input casting must follow conv1.
                teacher.visual.conv1.half()
                return teacher, None, main_transform()

            def tokenizer(prompts):
                return torch.tensor([0 if "cat" in text else 1 for text in prompts])

            fake = SimpleNamespace(__version__="test", create_model_and_transforms=factory,
                                   get_tokenizer=lambda _: tokenizer,
                                   get_pretrained_cfg=lambda *_: dict(quick_gelu=True, resize_mode="squash"))
            with patch.dict("sys.modules", {"open_clip": fake}):
                main(["--models-dir", str(weights), "--datasets", "quickdraw", "--root", str(root),
                      "--teachers", *keys, "--device", "cpu", "--workers", "0", "--batch-size", "2",
                      "--output-dir", str(output)])
            self.assertEqual(len(calls), 2)
            run = next(output.iterdir())
            with zipfile.ZipFile(run / "analysis_report.zip") as archive:
                self.assertIn("metrics_photo.csv", archive.namelist())
                self.assertIn("metrics_sketch.csv", archive.namelist())
                self.assertFalse(any("predictions_" in name for name in archive.namelist()))
            for modality in ("photo", "sketch"):
                summary = json.loads((run / f"metrics_{modality}.json").read_text())
                self.assertEqual(len(summary), 2)
                self.assertTrue(all(row["accuracy"] == 1 for row in summary))
                for key in keys:
                    folder = run / key / "quickdraw"
                    report = json.loads((folder / f"metrics_{modality}.json").read_text())
                    self.assertEqual(report["accuracy"], 1)
                    self.assertEqual(report["images"], 2)
                    self.assertEqual(report["split"], "seen")
                    with (folder / f"predictions_{modality}.csv").open() as stream:
                        predictions = list(csv.DictReader(stream))
                    self.assertEqual(len(predictions), 2)
                    self.assertTrue(all(row["correct"] == "1" for row in predictions))

    def test_wheel_resolver_resolves_each_python_without_cuda_downloads(self):
        from src.kaggle_classification_wheels import download_runtime_wheels
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)

            def run(command):
                calls.append(command)
                self.assertIn("--only-binary=:all:", command)
                if "--report" in command:
                    report = Path(command[command.index("--report") + 1])
                    python_version = command[command.index("--python-version") + 1]
                    numpy_version = "2.0.2" if python_version == "3.12" else "2.3.5"
                    report.write_text(json.dumps(dict(install=[
                        dict(metadata=dict(name="numpy", version=numpy_version)),
                        dict(metadata=dict(name="torch", version="2.5.1")),
                        dict(metadata=dict(name="nvidia-cublas-cu12", version="12.0")),
                    ])))
                else:
                    self.assertIn("--no-deps", command)
                    requirements = Path(command[command.index("-r") + 1]).read_text()
                    self.assertNotIn("torch", requirements)
                    self.assertNotIn("nvidia", requirements)

            inventory = download_runtime_wheels(["numpy"], ["3.12", "3.13"], bundle, bundle / "wheels", run)
            self.assertEqual(inventory["3.12"], dict(numpy="2.0.2"))
            self.assertEqual(inventory["3.13"], dict(numpy="2.3.5"))
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
