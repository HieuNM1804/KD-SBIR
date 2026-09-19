import unittest

import torch

from src.gap_core_audit import (
    build_cgrd_audit_rows,
    build_gap_audit_rows,
    summarize_cgrd_rows,
    summarize_gap_rows,
)
from src.losses import (
    counterfactual_gap_ranking_loss,
    gap_core_margin_correction_loss,
)
from src.model import _teacher_pretrain_inputs
from src.teacher_prompts import ModalityVisualPrompts


class CommonPromptDecompositionTest(unittest.TestCase):
    def test_common_prompt_is_exact_modality_average(self):
        prompts = ModalityVisualPrompts(
            width=2,
            n_ctx=1,
            depth=1,
            std=0.02,
            seed=7,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            prompts.prompts["photo"][0].copy_(torch.tensor([[2.0, 4.0]]))
            prompts.prompts["sketch"][0].copy_(torch.tensor([[0.0, 2.0]]))
        common_photo = prompts.for_layer(
            "photo", 0, 2, torch.float32, torch.device("cpu"), "common"
        )
        common_sketch = prompts.for_layer(
            "sketch", 0, 2, torch.float32, torch.device("cpu"), "common"
        )
        expected = torch.tensor([[[1.0, 3.0]], [[1.0, 3.0]]])
        self.assertTrue(torch.equal(common_photo, expected))
        self.assertTrue(torch.equal(common_sketch, expected))

    def test_unknown_prompt_mode_is_rejected(self):
        prompts = ModalityVisualPrompts(2, 1, 1, 0.02, 7, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "prompt mode"):
            prompts.for_layer(
                "photo", 0, 1, torch.float32, torch.device("cpu"), "invalid"
            )

    def test_swapped_prompt_uses_the_other_modality(self):
        prompts = ModalityVisualPrompts(
            width=2,
            n_ctx=1,
            depth=1,
            std=0.02,
            seed=7,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            prompts.prompts["photo"][0].copy_(torch.tensor([[2.0, 4.0]]))
            prompts.prompts["sketch"][0].copy_(torch.tensor([[0.0, 2.0]]))
        swapped_photo = prompts.for_layer(
            "photo", 0, 1, torch.float32, torch.device("cpu"), "swapped"
        )
        swapped_sketch = prompts.for_layer(
            "sketch", 0, 1, torch.float32, torch.device("cpu"), "swapped"
        )
        self.assertTrue(torch.equal(swapped_photo, torch.tensor([[[0.0, 2.0]]])))
        self.assertTrue(torch.equal(swapped_sketch, torch.tensor([[[2.0, 4.0]]])))

    def test_teacher_pretrain_labels_do_not_depend_on_trailing_cache_fields(self):
        labels = torch.tensor([3, 7])
        batch = tuple(torch.empty(2, 0) for _ in range(8)) + (
            labels,
            torch.empty(2, 0),
            torch.empty(2, 0),
            torch.empty(2, 0),
            torch.empty(2, 0),
        )
        _, _, extracted = _teacher_pretrain_inputs(batch)
        self.assertIs(extracted, labels)


class GapCorrectionAuditTest(unittest.TestCase):
    def test_audit_detects_positive_fixed_pair_margin_correction(self):
        labels = torch.tensor([0, 1])
        common_sketch = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        common_photo = torch.tensor([[0.7, 0.7], [0.7, 0.7]])
        full_sketch = common_sketch.clone()
        full_photo = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        rows = build_gap_audit_rows(
            full_sketch,
            common_sketch,
            labels,
            full_photo,
            common_photo,
            labels,
            seed=42,
        )
        summary = summarize_gap_rows(rows, bootstrap_samples=100)
        self.assertEqual(summary["queries"], 4)
        self.assertGreater(
            summary["verified_margin_correction"]["mean"],
            0.0,
        )
        self.assertGreater(summary["positive_delta"]["mean"], 0.0)

    def test_tri_state_audit_detects_monotonic_prompt_intervention(self):
        labels = torch.tensor([0, 1])
        full = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        common = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        swapped = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        rows = build_cgrd_audit_rows(
            full,
            common,
            swapped,
            labels,
            full,
            common,
            swapped,
            labels,
            hard_negative_topk=1,
        )
        summary = summarize_cgrd_rows(rows, bootstrap_samples=100)
        self.assertEqual(summary["pairs"], 4)
        self.assertGreater(summary["full_correction"]["mean"], 0.0)
        self.assertGreater(summary["swap_correction"]["mean"], 0.0)
        self.assertEqual(summary["monotonic"]["mean"], 1.0)


class GapCoreStudentLossTest(unittest.TestCase):
    def test_matching_full_minus_common_margin_has_lower_loss(self):
        labels = torch.tensor([0, 0, 1, 1])
        common_sketch = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.2], [0.9, 0.3], [0.2, 1.0], [0.3, 0.9]]),
            dim=-1,
        )
        common_photo = torch.nn.functional.normalize(
            torch.tensor([[0.7, 0.5], [0.6, 0.5], [0.5, 0.7], [0.5, 0.6]]),
            dim=-1,
        )
        full_sketch = common_sketch.clone()
        full_photo = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            dim=-1,
        )

        matched_photo = full_photo.clone().requires_grad_(True)
        matched_sketch = full_sketch.clone().requires_grad_(True)
        matched, diagnostics = gap_core_margin_correction_loss(
            matched_photo,
            matched_sketch,
            common_photo,
            common_sketch,
            full_photo,
            full_sketch,
            common_photo,
            common_sketch,
            labels,
            direction="bidirectional",
        )
        unchanged, _ = gap_core_margin_correction_loss(
            common_photo,
            common_sketch,
            common_photo,
            common_sketch,
            full_photo,
            full_sketch,
            common_photo,
            common_sketch,
            labels,
            direction="bidirectional",
        )
        self.assertGreater(diagnostics["coverage"].item(), 0.0)
        self.assertLess(matched.item(), unchanged.item())
        matched.backward()
        self.assertIsNotNone(matched_photo.grad)
        self.assertIsNotNone(matched_sketch.grad)


class CounterfactualGapRankingLossTest(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 1])
        self.common_sketch = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.2], [0.9, 0.3], [0.2, 1.0], [0.3, 0.9]]),
            dim=-1,
        )
        self.common_photo = torch.nn.functional.normalize(
            torch.tensor([[0.7, 0.5], [0.6, 0.5], [0.5, 0.7], [0.5, 0.6]]),
            dim=-1,
        )
        self.full_sketch = self.common_sketch.clone()
        self.full_photo = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            dim=-1,
        )
        self.swapped_sketch = self.common_sketch.clone()
        self.swapped_photo = torch.nn.functional.normalize(
            torch.tensor([[0.2, 1.0], [0.3, 0.9], [1.0, 0.2], [0.9, 0.3]]),
            dim=-1,
        )

    def _loss(self, student_full_photo, student_swapped_photo):
        return counterfactual_gap_ranking_loss(
            student_full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            student_swapped_photo,
            self.swapped_sketch,
            self.full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            self.swapped_photo,
            self.swapped_sketch,
            self.labels,
            hard_negative_topk=2,
            direction="sketch_to_photo",
        )

    def test_matching_two_sided_counterfactual_ranking_has_lower_loss(self):
        matched_full = self.full_photo.clone().requires_grad_(True)
        matched_swapped = self.swapped_photo.clone().requires_grad_(True)
        matched, diagnostics = self._loss(matched_full, matched_swapped)
        collapsed, _ = self._loss(self.common_photo, self.common_photo)
        self.assertGreater(diagnostics["coverage"].item(), 0.0)
        self.assertGreater(diagnostics["monotonicity"].item(), 0.0)
        self.assertLess(matched.item(), collapsed.item())
        matched.backward()
        self.assertIsNotNone(matched_full.grad)
        self.assertIsNotNone(matched_swapped.grad)

    def test_non_monotonic_swapped_teacher_is_rejected(self):
        loss, diagnostics = counterfactual_gap_ranking_loss(
            self.full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            self.full_photo,
            self.full_sketch,
            self.full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            self.full_photo,
            self.full_sketch,
            self.labels,
            hard_negative_topk=2,
            direction="sketch_to_photo",
        )
        self.assertEqual(diagnostics["coverage"].item(), 0.0)
        self.assertEqual(loss.item(), 0.0)

    def test_shuffled_control_preserves_positive_target_distribution(self):
        _, verified = self._loss(self.full_photo, self.swapped_photo)
        _, shuffled = counterfactual_gap_ranking_loss(
            self.full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            self.swapped_photo,
            self.swapped_sketch,
            self.full_photo,
            self.full_sketch,
            self.common_photo,
            self.common_sketch,
            self.swapped_photo,
            self.swapped_sketch,
            self.labels,
            hard_negative_topk=2,
            control="shuffled",
            direction="sketch_to_photo",
        )
        self.assertAlmostEqual(
            shuffled["coverage"].item(), verified["coverage"].item()
        )
        self.assertAlmostEqual(
            shuffled["teacher_full_correction"].item(),
            verified["teacher_full_correction"].item(),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            shuffled["teacher_swap_correction"].item(),
            verified["teacher_swap_correction"].item(),
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
