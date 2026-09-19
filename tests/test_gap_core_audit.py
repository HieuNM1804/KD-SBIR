import unittest

import torch

from src.gap_core_audit import build_gap_audit_rows, summarize_gap_rows
from src.losses import gap_core_margin_correction_loss
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


if __name__ == "__main__":
    unittest.main()
