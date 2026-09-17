import unittest
from types import SimpleNamespace

import torch

from src.losses import loss_fn
from src.photo_sketch_promptkd import (
    build_cross_modal_prototypes,
    prototype_kd_loss,
    select_central_anchor_indices,
)


class PhotoSketchPromptKDTest(unittest.TestCase):
    def test_anchor_selection_is_class_balanced_and_deterministic(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.0, -1.0],
            ]
        )
        labels = [0, 0, 0, 1, 1, 1]

        selected = select_central_anchor_indices(
            features,
            labels,
            class_count=2,
            anchors_per_class=2,
        )

        self.assertEqual(selected.tolist(), [[1, 0], [4, 3]])

    def test_cross_modal_prototypes_are_normalized(self):
        sketch = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
        photo = torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.1, 0.9], [0.0, 1.0]])
        indices = torch.tensor([[0, 1], [2, 3]])

        prototypes = build_cross_modal_prototypes(
            sketch,
            photo,
            indices,
            indices,
        )

        self.assertEqual(tuple(prototypes.shape), (2, 2))
        torch.testing.assert_close(
            prototypes.norm(dim=-1),
            torch.ones(2),
        )
        self.assertGreater(prototypes[0, 0].item(), prototypes[0, 1].item())
        self.assertGreater(prototypes[1, 1].item(), prototypes[1, 0].item())

    def test_kd_matches_prototype_rankings_across_different_dimensions(self):
        teacher_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        teacher_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        student_features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        )
        student_prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        matched = prototype_kd_loss(
            student_features,
            teacher_features,
            student_prototypes,
            teacher_prototypes,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )
        mismatched = prototype_kd_loss(
            student_features.flip(0),
            teacher_features,
            student_prototypes,
            teacher_prototypes,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )

        self.assertLess(matched.item(), 1e-6)
        self.assertGreater(mismatched.item(), 1.0)
        matched.backward()
        self.assertIsNotNone(student_features.grad)

    def test_anchor_selection_rejects_underrepresented_classes(self):
        with self.assertRaisesRegex(ValueError, "2 anchors were requested"):
            select_central_anchor_indices(
                torch.eye(2),
                [0, 1],
                class_count=2,
                anchors_per_class=2,
            )

    def test_text_free_objective_updates_both_student_modalities(self):
        args = SimpleNamespace(
            lambda_domain=0.0,
            lambda_modality=0.0,
            lambda_prototype=0.5,
            kd_temperature=0.07,
            photo_text_kd_temperature=0.1,
            sketch_text_kd_temperature=0.1,
            prototype_student_temperature=0.1,
            prototype_teacher_temperature=0.1,
        )
        student_photo = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
        student_sketch = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
        teacher_photo = torch.tensor([[1.0, 0.0]])
        teacher_sketch = torch.tensor([[1.0, 0.0]])
        student_prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        teacher_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        features = (
            student_photo,
            student_sketch,
            teacher_photo,
            teacher_sketch,
            True,
            None,
            None,
            None,
            None,
            True,
            student_prototypes,
            teacher_prototypes,
        )

        loss, parts = loss_fn(args, features)
        loss.backward()

        self.assertGreater(parts["prototype_kd"].item(), 1.0)
        self.assertIsNotNone(student_photo.grad)
        self.assertIsNotNone(student_sketch.grad)
        self.assertGreater(student_photo.grad.norm().item(), 0.0)
        self.assertGreater(student_sketch.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
