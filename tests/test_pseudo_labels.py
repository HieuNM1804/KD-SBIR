"""Label leakage, frozen classification, pairing and cache checks; no model downloads."""

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

from src.dataset import TrainDataset, WorkerInvariantSampler
from src.losses import batch_hard_teacher_triplet_loss
from src.pseudo_labels import (
    classify_seen, default_pseudo_label_path, input_metadata,
    prediction_fingerprint, prepare_pseudo_labels, validate_payload,
)


class FrozenTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.class_embedding = nn.Parameter(torch.zeros(2))
        self.visual.conv1 = nn.Conv2d(3, 2, 1).half()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.prompts_seen = []

    def text_tokenizer(self, texts):
        self.prompts_seen.extend(texts)
        return torch.tensor([0 if "cat" in name else 1 for name in texts])

    def encode_text(self, tokens):
        assert not self.training and not torch.is_grad_enabled()
        assert not any(parameter.requires_grad for parameter in self.parameters())
        return torch.nn.functional.one_hot(tokens, 2).float()

    def encode_image(self, images):
        assert not self.training and not torch.is_grad_enabled()
        assert not any(parameter.requires_grad for parameter in self.parameters())
        assert images.dtype == self.visual.conv1.weight.dtype
        return images[:, [0, 2]].mean((2, 3))


class PseudoLabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        # All predictions intentionally DISAGREE with the true folder labels.
        for modality in ("sketch", "photo"):
            for name, color in (("cat", "blue"), ("dog", "red"), ("bat", "green")):
                directory = self.root / modality / name
                directory.mkdir(parents=True)
                for index in range(3):
                    Image.new("RGB", (11, 9), color).save(directory / f"{index}.png")
        self.args = SimpleNamespace(root=str(self.root), dataset="quickdraw", seed=42,
                                    max_size=224, pseudo_batch_size=2, workers=0, progress=False,
                                    pseudo_label_dir=str(Path(self.temporary.name) / "labels"),
                                    pseudo_label_path="", rebuild_pseudo_labels=False,
                                    batch_size=2, teacher_pretrain_batch_size=2)
        self.dataset = TrainDataset(self.args)

    def test_frozen_classification_ignores_folder_labels_and_unseen_images(self):
        teacher = FrozenTeacher()
        before = {key: value.clone() for key, value in teacher.state_dict().items()}
        output = classify_seen(teacher, self.dataset, 2, 0, False)
        for modality in ("sketch", "photo"):
            self.assertEqual(output[modality + "_labels"].tolist(), [1, 1, 1, 0, 0, 0])
            self.assertTrue((output[modality + "_confidence"] > 0.99).all())
        self.assertEqual(teacher.prompts_seen, ["a sketch of a cat.", "a sketch of a dog.",
                                               "a photo of a cat.", "a photo of a dog."])
        self.assertTrue(all("bat" not in path for path in self.dataset.all_sketches_path))
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_pairs_and_triplet_masks_use_pseudo_labels_not_true_categories(self):
        # Predictions for sketches and photos differ inside the SAME true class.
        sketch_labels = torch.tensor([1, 0, 1, 0, 1, 0])
        photo_labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.dataset.set_pseudo_labels(sketch_labels, photo_labels)
        self.dataset.set_teacher_features(torch.arange(6).reshape(-1, 1).float(),
                                          torch.arange(6).reshape(-1, 1).float())
        # A forbidden access to true-category sampling must not affect pseudo training.
        self.dataset.all_photos_path = None
        self.dataset.category_to_label = None
        for epoch in (0, 1, 3):
            for index in range(6):
                _, _, photo_feature, sketch_feature, label = self.dataset[(epoch, index)]
                photo_index, sketch_index = int(photo_feature.item()), int(sketch_feature.item())
                self.assertEqual(label, int(sketch_labels[sketch_index]))
                self.assertEqual(label, int(photo_labels[photo_index]))
        labels = torch.tensor([self.dataset[(0, index)][-1] for index in range(6)])
        sketch = torch.randn(6, 7, generator=torch.Generator().manual_seed(5), requires_grad=True)
        photo = torch.randn(6, 7, generator=torch.Generator().manual_seed(8), requires_grad=True)
        loss = batch_hard_teacher_triplet_loss(sketch, photo, labels, 0.3)
        loss.backward()
        self.assertTrue(torch.isfinite(sketch.grad).all())
        self.assertGreater(float(photo.grad.norm()), 0)
        # Keep main's epoch-dependent deterministic sampler.
        first = WorkerInvariantSampler(self.dataset, 42)
        second = WorkerInvariantSampler(self.dataset, 42)
        self.assertEqual(list(first), list(second))
        self.assertEqual(list(first), list(second))

    def test_missing_pseudo_positive_preserves_feature_bank_indices(self):
        self.dataset.all_categories.append("third_seen_class")
        report = self.dataset.set_pseudo_labels(torch.tensor([2, 1, 0, 2, 1, 0]),
                                                torch.tensor([0, 1, 0, 1, 0, 1]))
        self.assertEqual(report["excluded_sketches_no_pseudo_positive"], 2)
        self.assertEqual(self.dataset.training_sketch_indices, [1, 2, 4, 5])
        self.assertEqual(len(self.dataset.all_sketches_path), 6)
        self.dataset.set_teacher_features(torch.arange(6).reshape(-1, 1),
                                          torch.arange(6).reshape(-1, 1))
        self.assertEqual(int(self.dataset[(0, 0)][3]), 1)
        self.assertEqual(int(self.dataset[(0, 3)][3]), 5)

    def test_rejects_collapsed_or_invalid_pseudo_labels_without_true_label_fallback(self):
        for sketches, photos in ((torch.zeros(6, dtype=torch.long), torch.zeros(6, dtype=torch.long)),
                                 (torch.zeros(5, dtype=torch.long), torch.zeros(6, dtype=torch.long)),
                                 (torch.ones(6), torch.zeros(6, dtype=torch.long)),
                                 (torch.full((6,), 2), torch.zeros(6, dtype=torch.long))):
            with self.subTest(sketches=sketches.tolist()), self.assertRaises(ValueError):
                self.dataset.set_pseudo_labels(sketches, photos)

    def test_cache_reuse_skips_teacher_and_exports_both_modalities(self):
        teacher = FrozenTeacher()
        returned = prepare_pseudo_labels(self.args, self.dataset, lambda: teacher)
        self.assertIs(returned, teacher)
        fingerprint = self.args.pseudo_label_fingerprint
        path = Path(self.args.pseudo_label_path)
        for modality in ("sketch", "photo"):
            with path.with_name(path.stem + f"_{modality}.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["pseudo_class"], "dog")
            self.assertNotIn("true_label", rows[0])
        def forbidden():
            raise AssertionError("A pseudo-label cache hit must not load DFN5B.")
        reused = prepare_pseudo_labels(self.args, TrainDataset(self.args), forbidden)
        self.assertIsNone(reused)
        self.assertEqual(self.args.pseudo_label_fingerprint, fingerprint)
        self.args.rebuild_pseudo_labels = True
        with self.assertRaises(AssertionError):
            prepare_pseudo_labels(self.args, self.dataset, forbidden)

    def test_pseudo_metadata_and_teacher_cache_names_bind_exact_predictions(self):
        payload = {"metadata": input_metadata(self.args, self.dataset),
                   **classify_seen(FrozenTeacher(), self.dataset, 2, 0, False)}
        validate_payload(payload, input_metadata(self.args, self.dataset))
        first = prediction_fingerprint(payload)
        payload["sketch_labels"][0] = 0
        self.assertNotEqual(first, prediction_fingerprint(payload))
        payload["photo_confidence"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_payload(payload, input_metadata(self.args, self.dataset))
        changed = dict(payload["metadata"], dataset="sketchy_2")
        with self.assertRaises(RuntimeError):
            validate_payload(payload, changed)
        source = Path(__file__).resolve().parents[1] / "src/model.py"
        functions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_teacher_training_config", "default_teacher_cache_path"}]
        namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path, DFN5B_MODEL="ViT-H-14-quickgelu",
                         DFN5B_PRETRAINED="dfn5b", DFN5B_OUTPUT_DIM=1024, TEACHER_CACHE_FORMAT_VERSION=6)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        fields = {node.attr: 1 for node in ast.walk(functions[0]) if isinstance(node, ast.Attribute)
                  and isinstance(node.value, ast.Name) and node.value.id == "args"}
        args = SimpleNamespace(**fields, root=str(self.root), dataset="quickdraw", teacher_cache_dir="cache",
                               pseudo_label_fingerprint=first)
        first_path = namespace["default_teacher_cache_path"](args, self.dataset)
        args.pseudo_label_fingerprint = "other_predictions"
        self.assertNotEqual(first_path, namespace["default_teacher_cache_path"](args, self.dataset))
        config = namespace["_teacher_training_config"](args)
        self.assertEqual(config["checkpoint_selection"], "best_unseen_precision")
        self.assertEqual(config["training_label_source"], "frozen_dfn5b_pseudo_v1")

    def test_class_names_change_invalidate_static_label_cache(self):
        first = default_pseudo_label_path(self.args, self.dataset)
        self.dataset.all_categories.reverse()
        self.assertNotEqual(first, default_pseudo_label_path(self.args, self.dataset))

    def test_kaggle_resolver_separates_abis_without_cuda_wheels(self):
        from src.kaggle_bundle_wheels import download_runtime_wheels
        calls = []
        bundle = Path(self.temporary.name)

        def run(command):
            calls.append(command)
            if "--report" in command:
                version = command[command.index("--python-version") + 1]
                report = Path(command[command.index("--report") + 1])
                report.write_text(json.dumps({"install": [
                    {"metadata": {"name": "numpy", "version": "2.0.2" if version == "3.12" else "2.3.5"}},
                    {"metadata": {"name": "torch", "version": "2.6.0"}},
                    {"metadata": {"name": "nvidia-cublas-cu12", "version": "12.0"}},
                ]}))
            else:
                self.assertIn("--no-deps", command)
                requirements = Path(command[command.index("-r") + 1]).read_text()
                self.assertNotIn("torch", requirements)
                self.assertNotIn("nvidia", requirements)
        inventory = download_runtime_wheels(["numpy"], ["3.12", "3.13"], bundle, bundle / "wheels", run)
        self.assertEqual(inventory["3.12"], {"numpy": "2.0.2"})
        self.assertEqual(inventory["3.13"], {"numpy": "2.3.5"})
        self.assertEqual(len(calls), 4)

    def test_all_three_stages_with_real_prompt_training_and_kd_without_weights(self):
        from clip.model import CLIP
        from open_clip.transformer import VisionTransformer
        from src.losses import loss_fn

        class TinyTeacher(FrozenTeacher):
            def __init__(self):
                super().__init__()
                self.visual = VisionTransformer(image_size=224, patch_size=224, width=64,
                                                layers=2, heads=1, mlp_ratio=2, output_dim=1024)
                self.output_dim = 1024

            def encode_text(self, tokens):
                # Still the original frozen text encoder, now with main's 1024 dimensions.
                return torch.nn.functional.one_hot(tokens, 1024).float()

            def encode_image(self, images):
                return torch.nn.functional.pad(super().encode_image(images), (0, 1022))

        teacher = TinyTeacher()
        original = {key: value.clone() for key, value in teacher.state_dict().items()}
        student = CLIP(embed_dim=32, image_resolution=224, vision_layers=2,
                       vision_width=64, vision_patch_size=224, context_length=77,
                       vocab_size=49408, transformer_width=64, transformer_heads=1,
                       transformer_layers=1).float()
        captured = {}

        class CPUTrainer:
            def __init__(self, **kwargs):
                pass

            def fit(self, model, train_loader, validation_loaders):
                self.assertions(model, train_loader, validation_loaders)

            def assertions(self, model, train_loader, validation_loaders):
                captured["args"] = model.args
                captured["pseudo"] = train_loader.dataset.pseudo_sketch_labels.tolist()
                captured["teacher_released"] = model.model._teacher is None
                captured["prompts_released"] = model.model.teacher_prompts is None
                batch = next(iter(train_loader))
                loss, parts = loss_fn(model.args, model.model(batch))
                loss.backward()
                captured["student_grad"] = sum(float(p.grad.norm()) for p in model.model.parameters()
                                                if p.requires_grad and p.grad is not None)
                captured["student_backbone_frozen"] = all(not p.requires_grad and p.grad is None
                                                          for p in model.model.clip_model.parameters())
                captured["student_loss"] = float(loss.detach())
                captured["parts"] = set(parts)

        command = ["src.train", "--root", str(self.root), "--dataset", "quickdraw",
                   "--epochs", "1", "--workers", "0", "--batch_size", "6",
                   "--test_batch_size", "3", "--n_ctx_visual", "2", "--prompt_depth", "2",
                   "--teacher_pretrain_epochs", "1", "--teacher_pretrain_batch_size", "6",
                   "--teacher_n_ctx_visual", "2", "--teacher_prompt_depth", "2",
                   "--teacher_prompt_lr", "0.003", "--lambda_domain", "2", "--lambda_modality", "2",
                   "--pseudo_batch_size", "2", "--pseudo_label_dir", self.args.pseudo_label_dir,
                   "--teacher_cache_dir", str(Path(self.temporary.name) / "teacher_cache"), "--no_progress"]
        previous = Path.cwd()
        try:
            os.chdir(self.temporary.name)
            with patch.object(sys, "argv", command), patch("pytorch_lightning.Trainer", CPUTrainer), \
                 patch("src.model._load_teacher", return_value=teacher) as factory, \
                 patch("src.model._load_clip_model", return_value=student):
                runpy.run_module("src.train", run_name="__main__")
            factory.assert_called_once()
            self.assertTrue(factory.call_args.kwargs["ignore_cache"])
        finally:
            os.chdir(previous)
        self.assertEqual(captured["pseudo"], [1, 1, 1, 0, 0, 0])
        self.assertTrue(captured["teacher_released"] and captured["prompts_released"])
        self.assertTrue(math.isfinite(captured["student_loss"]))
        self.assertGreater(captured["student_grad"], 0)
        self.assertTrue(captured["student_backbone_frozen"])
        self.assertEqual(captured["parts"], {"domain_kd", "modality_kd"})
        payload = torch.load(captured["args"].teacher_cache_path, weights_only=True)
        self.assertEqual(payload["metadata"]["pseudo_label_fingerprint"], captured["args"].pseudo_label_fingerprint)
        self.assertTrue(payload["teacher_prompt_state_dict"])
        from src.teacher_prompts import build_teacher_prompt_controller
        initial_prompts = build_teacher_prompt_controller(teacher, 2, 2, std=0.02, seed=42).state_dict()
        self.assertTrue(any(not torch.equal(value, initial_prompts[key])
                            for key, value in payload["teacher_prompt_state_dict"].items()))
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, original[key])


if __name__ == "__main__":
    unittest.main()
