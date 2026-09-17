import unittest
from types import SimpleNamespace

import torch

from src.losses import loss_fn
from src.photo_sketch_promptkd import (
    build_retrieval_vocabulary,
    directional_vocabulary_kd_loss,
    paired_coordinate_consistency,
    select_diverse_candidates,
    vocabulary_coordinates,
)


class PhotoSketchPromptKDTest(unittest.TestCase):
    def test_candidate_selection_is_balanced_and_diverse(self):
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
        selected = select_diverse_candidates(
            features,
            [0, 0, 0, 1, 1, 1],
            class_count=2,
            candidates_per_class=2,
        )
        self.assertEqual(selected.tolist(), [[1, 2], [4, 5]])

    def test_candidate_selection_rejects_underrepresented_classes(self):
        with self.assertRaisesRegex(ValueError, "2 candidates were requested"):
            select_diverse_candidates(
                torch.eye(2),
                [0, 1],
                class_count=2,
                candidates_per_class=2,
            )

    def test_vocabulary_contains_unique_paired_landmarks(self):
        sketch = torch.eye(4)
        photo = torch.eye(4)
        labels = [0, 0, 1, 1]

        vocabulary = build_retrieval_vocabulary(
            sketch,
            photo,
            labels,
            labels,
            vocabulary_size=4,
            mutual_topk=1,
            class_count=2,
        )

        self.assertEqual(len(vocabulary["sketch_indices"].unique()), 4)
        self.assertEqual(len(vocabulary["photo_indices"].unique()), 4)
        self.assertEqual(vocabulary["labels"].bincount().tolist(), [2, 2])
        self.assertTrue(torch.all(vocabulary["margin"] > 0))

    def test_vocabulary_rejects_cross_class_teacher_matches(self):
        sketch = torch.eye(2)
        photo = torch.eye(2).flip(0)
        with self.assertRaisesRegex(RuntimeError, "No label-consistent"):
            build_retrieval_vocabulary(
                sketch,
                photo,
                [0, 1],
                [0, 1],
                vocabulary_size=2,
                mutual_topk=1,
                class_count=2,
            )

    def test_kd_matches_rankings_across_different_dimensions(self):
        teacher_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        teacher_landmarks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        student_features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        )
        student_landmarks = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        matched, statistics = directional_vocabulary_kd_loss(
            student_features,
            teacher_features,
            student_landmarks,
            teacher_landmarks,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )
        mismatched, _ = directional_vocabulary_kd_loss(
            student_features.flip(0),
            teacher_features,
            student_landmarks,
            teacher_landmarks,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )

        self.assertLess(matched.item(), 1e-6)
        self.assertGreater(mismatched.item(), 1.0)
        self.assertGreater(statistics["teacher_confidence"].item(), 0.9)
        matched.backward()
        self.assertIsNotNone(student_features.grad)

    def test_coordinate_consistency_and_descriptor(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        landmarks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        matched = paired_coordinate_consistency(
            features,
            features,
            landmarks,
            landmarks,
            temperature=0.1,
        )
        mismatched = paired_coordinate_consistency(
            features,
            features.flip(0),
            landmarks,
            landmarks,
            temperature=0.1,
        )
        coordinates = vocabulary_coordinates(features, landmarks, 0.1)

        self.assertLess(matched.item(), 1e-6)
        self.assertGreater(mismatched.item(), 0.5)
        self.assertEqual(tuple(coordinates.shape), (2, 2))
        torch.testing.assert_close(coordinates.norm(dim=-1), torch.ones(2))

    def test_text_free_objective_updates_both_student_modalities(self):
        args = SimpleNamespace(
            lambda_domain=0.0,
            lambda_modality=0.0,
            lambda_retrieval_vocab=0.5,
            lambda_retrieval_vocab_pair=0.1,
            kd_temperature=0.07,
            photo_text_kd_temperature=0.1,
            sketch_text_kd_temperature=0.1,
            retrieval_vocab_student_temperature=0.1,
            retrieval_vocab_teacher_temperature=0.1,
        )
        student_photo = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
        student_sketch = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
        teacher_photo = torch.tensor([[1.0, 0.0]])
        teacher_sketch = torch.tensor([[1.0, 0.0]])
        student_sketch_landmarks = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        student_photo_landmarks = student_sketch_landmarks.clone()
        teacher_sketch_landmarks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        teacher_photo_landmarks = teacher_sketch_landmarks.clone()
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
            student_sketch_landmarks,
            student_photo_landmarks,
            teacher_sketch_landmarks,
            teacher_photo_landmarks,
        )

        loss, parts = loss_fn(args, features)
        loss.backward()

        self.assertGreater(parts["retrieval_vocab_kd"].item(), 1.0)
        self.assertIsNotNone(student_photo.grad)
        self.assertIsNotNone(student_sketch.grad)
        self.assertGreater(student_photo.grad.norm().item(), 0.0)
        self.assertGreater(student_sketch.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
