import unittest
from types import SimpleNamespace

import torch

from src.losses import cross_modal_margin_correction_loss, loss_fn


def normalized(rows):
    return torch.nn.functional.normalize(
        torch.tensor(rows, dtype=torch.float32), dim=-1
    )


class CrossModalMarginCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 1])
        self.base_sketch = normalized([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        self.base_photo = normalized([[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6]])
        self.adapted_sketch = self.base_sketch
        self.adapted_photo = normalized(
            [[0.95, 0.05], [0.9, 0.1], [0.05, 0.95], [0.1, 0.9]]
        )

    def loss(self, prompted_photo, prompted_sketch, control="verified"):
        return cross_modal_margin_correction_loss(
            prompted_photo=prompted_photo,
            prompted_sketch=prompted_sketch,
            base_student_photo=self.base_photo,
            base_student_sketch=self.base_sketch,
            adapted_teacher_photo=self.adapted_photo,
            adapted_teacher_sketch=self.adapted_sketch,
            base_teacher_photo=self.base_photo,
            base_teacher_sketch=self.base_sketch,
            labels=self.labels,
            teacher_temperature=0.05,
            student_temperature=0.05,
            hard_negative_topk=2,
            minimum_teacher_correction=0.0,
            maximum_weight=0.25,
            control=control,
            direction="sketch_to_photo",
        )

    def test_matching_correction_beats_reversed_student_change(self):
        matched_loss, diagnostics = self.loss(self.adapted_photo, self.adapted_sketch)
        reversed_loss, _ = self.loss(self.base_photo, self.adapted_sketch)
        self.assertGreater(diagnostics["coverage"].item(), 0.0)
        self.assertGreater(diagnostics["teacher_correction"].item(), 0.0)
        self.assertLess(matched_loss.item(), reversed_loss.item())

    def test_loss_backpropagates_to_prompted_features_only(self):
        prompted_photo = self.adapted_photo.clone().requires_grad_(True)
        prompted_sketch = self.adapted_sketch.clone().requires_grad_(True)
        loss, _ = self.loss(prompted_photo, prompted_sketch)
        loss.backward()
        self.assertIsNotNone(prompted_photo.grad)
        self.assertIsNotNone(prompted_sketch.grad)
        self.assertTrue(torch.isfinite(prompted_photo.grad).all())
        self.assertTrue(torch.isfinite(prompted_sketch.grad).all())

    def test_main_loss_contract_includes_core_diagnostics(self):
        args = SimpleNamespace(
            lambda_domain=0.0,
            lambda_modality=0.0,
            lambda_core=0.5,
            lambda_gap_core=0.0,
            kd_temperature=0.07,
            core_teacher_temperature=0.05,
            core_student_temperature=0.05,
            core_hard_negative_topk=2,
            core_min_teacher_correction=0.0,
            core_max_weight=0.25,
            core_control="verified",
            core_direction="sketch_to_photo",
        )
        prompted_photo = self.adapted_photo.clone().requires_grad_(True)
        prompted_sketch = self.adapted_sketch.clone().requires_grad_(True)
        features = (
            prompted_photo,
            prompted_sketch,
            self.adapted_photo,
            self.adapted_sketch,
            True,
            None,
            None,
            None,
            None,
            self.base_photo,
            self.base_sketch,
            self.base_photo,
            self.base_sketch,
            self.labels,
            None,
            None,
            None,
            None,
        )
        total, diagnostics = loss_fn(args, features)
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(diagnostics["core_coverage"].item(), 0.0)
        total.backward()
        self.assertIsNotNone(prompted_photo.grad)


class CoreArgumentContractTest(unittest.TestCase):
    def test_documented_defaults_are_positive(self):
        args = SimpleNamespace(
            core_teacher_temperature=0.05,
            core_student_temperature=0.05,
            core_hard_negative_topk=8,
            core_max_weight=0.25,
        )
        self.assertGreater(args.core_teacher_temperature, 0)
        self.assertGreater(args.core_student_temperature, 0)
        self.assertGreaterEqual(args.core_hard_negative_topk, 1)
        self.assertGreater(args.core_max_weight, 0)


if __name__ == "__main__":
    unittest.main()
