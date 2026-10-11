"""Direction, gradient, and cache checks without loading teacher weights."""

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from src.losses import batch_hard_teacher_triplet_loss


def reference_loss(sketch, photo, labels, margin):
    distance = 1 - F.normalize(sketch, dim=-1) @ F.normalize(photo, dim=-1).t()
    losses = []
    for index, label in enumerate(labels):
        positive = labels.eq(label)
        negative = ~positive
        if negative.any():
            losses.append(F.relu(distance[index, positive].max()
                                 - distance[index, negative].min() + margin))
    return torch.stack(losses).mean() if losses else distance.new_zeros(())


class TeacherTripletTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(13)
        self.sketch = torch.randn(6, 5, generator=generator)
        self.photo = torch.randn(6, 5, generator=generator)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])

    def test_only_sketch_anchors_with_same_class_hard_mining(self):
        expected = reference_loss(self.sketch, self.photo, self.labels, 0.2)
        reverse = reference_loss(self.photo, self.sketch, self.labels, 0.2)
        self.assertGreater(abs(float(expected - reverse)), 1e-3)
        actual = batch_hard_teacher_triplet_loss(
            self.sketch, self.photo, self.labels, 0.2)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(abs(float(actual - 0.5 * (expected + reverse))), 1e-3)

    def test_both_encoders_receive_only_forward_triplet_gradients(self):
        sketch = self.sketch.clone().requires_grad_()
        photo = self.photo.clone().requires_grad_()
        actual = batch_hard_teacher_triplet_loss(sketch, photo, self.labels)
        gradients = torch.autograd.grad(actual, (sketch, photo))
        expected = torch.autograd.grad(
            reference_loss(sketch, photo, self.labels, 0.2), (sketch, photo))
        for gradient, target in zip(gradients, expected):
            torch.testing.assert_close(gradient, target)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0)

    def test_single_class_batch_returns_finite_zero(self):
        actual = batch_hard_teacher_triplet_loss(
            self.sketch, self.photo, torch.zeros(6, dtype=torch.long))
        self.assertEqual(float(actual), 0)

    def test_teacher_direction_changes_metadata_and_default_cache_path(self):
        # Load the actual cache functions without importing heavyweight model
        # dependencies or constructing a CLIP model.
        source = Path(__file__).resolve().parents[1] / "src" / "model.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_teacher_training_config", "default_teacher_cache_path"}]
        namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path,
                         DFN5B_MODEL="ViT-H-14-quickgelu", DFN5B_PRETRAINED="dfn5b",
                         DFN5B_OUTPUT_DIM=1024, TEACHER_CACHE_FORMAT_VERSION=6)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        fields = {node.attr: 1 for node in ast.walk(functions[0])
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id == "args"}
        args = SimpleNamespace(**fields, dataset="sketchy_1", root="dataset",
                               teacher_cache_dir="cache")
        dataset = SimpleNamespace(max_size=224, all_categories=["cat"],
                                  all_sketches_path=["dataset/sketch/cat/a.png"],
                                  all_photo_paths=["dataset/photo/cat/a.jpg"])
        config = namespace["_teacher_training_config"](args)
        self.assertEqual(config["teacher_triplet_direction"], "sketch_to_photo")
        path = namespace["default_teacher_cache_path"](args, dataset)
        legacy = dict(config)
        del legacy["teacher_triplet_direction"]
        namespace["_teacher_training_config"] = lambda unused: legacy
        legacy_path = namespace["default_teacher_cache_path"](args, dataset)
        self.assertNotEqual(path, legacy_path)


if __name__ == "__main__":
    unittest.main()
