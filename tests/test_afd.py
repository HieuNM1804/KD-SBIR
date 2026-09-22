import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from src.afd import (
    AugmentedFeatureFusion,
    controlled_afd_inputs,
    dual_axis_afd_loss,
    image_class_text_contrastive_loss,
    multi_positive_contrastive_loss,
)
from src.losses import loss_fn


class AugmentedFeatureDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_student_identity_initialization_preserves_student_without_teacher(self):
        fusion = AugmentedFeatureFusion(4, 6, "student_identity")
        student = torch.randn(5, 4)
        teacher = torch.zeros(5, 6)
        actual = fusion(student, teacher)
        expected = F.normalize(student.float(), dim=-1)
        torch.testing.assert_close(actual, expected)

    def test_teacher_is_detached_but_student_and_fusion_receive_gradients(self):
        fusion = AugmentedFeatureFusion(4, 6, "xavier")
        student = torch.randn(5, 4, requires_grad=True)
        teacher = torch.randn(5, 6, requires_grad=True)
        output = fusion(student, teacher)
        output.square().sum().backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertGreater(fusion.projection.weight.grad.abs().sum().item(), 0)

    def test_controls_only_change_requested_teacher_axis(self):
        student = torch.randn(4, 3)
        teacher = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        _, verified = controlled_afd_inputs(student, teacher, "verified", "image")
        _, shuffled = controlled_afd_inputs(
            student, teacher, "shuffled_image", "image"
        )
        _, text_untouched = controlled_afd_inputs(
            student, teacher, "shuffled_text", "image"
        )
        torch.testing.assert_close(verified, teacher)
        torch.testing.assert_close(shuffled, teacher.roll(1, dims=0))
        torch.testing.assert_close(text_untouched, teacher)

    def test_multi_positive_loss_rewards_correct_class_relations(self):
        labels = torch.tensor([0, 0, 1, 1])
        anchors = F.normalize(
            torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]]),
            dim=-1,
        )
        aligned = anchors.clone()
        swapped = aligned.flip(0)
        good = multi_positive_contrastive_loss(
            anchors, aligned, labels, labels, temperature=0.1
        )
        bad = multi_positive_contrastive_loss(
            anchors, swapped, labels, labels, temperature=0.1
        )
        self.assertLess(good.item(), bad.item())

    def test_image_text_loss_backpropagates_to_both_fused_branches(self):
        images = torch.randn(6, 4, requires_grad=True)
        text = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss, _ = image_class_text_contrastive_loss(images, text, labels, 0.2)
        loss.backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertGreater(text.grad.abs().sum().item(), 0)

    def test_loss_fn_is_exactly_the_two_afd_axes(self):
        labels = torch.tensor([0, 1, 0, 1])
        photo = torch.randn(4, 5, requires_grad=True)
        sketch = torch.randn(4, 5, requires_grad=True)
        sketch_text = torch.randn(2, 5, requires_grad=True)
        photo_text = torch.randn(2, 5, requires_grad=True)
        args = SimpleNamespace(
            lambda_afd_sp=0.3,
            lambda_afd_it=0.7,
            afd_temperature_sp=0.11,
            afd_temperature_it=0.13,
            # Deliberately nonsensical legacy values: loss_fn must not read them.
            lambda_domain=999.0,
            lambda_modality=999.0,
        )
        actual, metrics = loss_fn(
            args, (photo, sketch, sketch_text, photo_text), labels
        )
        expected, _ = dual_axis_afd_loss(
            sketch,
            photo,
            sketch_text,
            photo_text,
            labels,
            sketch_photo_weight=0.3,
            image_text_weight=0.7,
            sketch_photo_temperature=0.11,
            image_text_temperature=0.13,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual, metrics["afd_sp_weighted"] + metrics["afd_it_weighted"]
        )

    def test_disabled_text_axis_accepts_missing_text_banks(self):
        labels = torch.tensor([0, 1])
        sketch = torch.randn(2, 4)
        photo = torch.randn(2, 4)
        loss, metrics = dual_axis_afd_loss(
            sketch,
            photo,
            None,
            None,
            labels,
            sketch_photo_weight=1.0,
            image_text_weight=0.0,
            sketch_photo_temperature=0.1,
            image_text_temperature=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["afd_it"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
