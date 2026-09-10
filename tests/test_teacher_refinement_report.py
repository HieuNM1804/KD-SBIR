import tempfile
import unittest
from pathlib import Path

from src.teacher_refinement_report import (
    report_path_for_cache,
    summarize_teacher_refinement_reports,
    write_teacher_refinement_report,
)


class TeacherRefinementReportTests(unittest.TestCase):
    @staticmethod
    def _write(directory, seed, delta, semantic_weight=0.25):
        cache_path = Path(directory) / f"teacher_seed_{seed}.pt"
        metadata = {
            "teacher_objective": (
                "matched_control_staged_visual_text_semantic_visual_refinement"
            ),
            "dataset_fingerprint": "same-dataset",
            "seed": seed,
            "teacher_prompt_seed": seed,
            "teacher_text_prompt_seed": seed + 50_000,
            "lambda_teacher_semantic_refine": semantic_weight,
        }
        result = {
            "best_phase": (
                "semantic_refine" if delta > 0 else "matched_control"
            ),
            "phase_a_acc1": 0.2,
            "phase_a_acc5": 0.4,
            "control_acc1": 0.22,
            "control_acc5": 0.42,
            "semantic_acc1": 0.22 + delta,
            "semantic_acc5": 0.42 + delta,
            "best_acc1": max(0.22, 0.22 + delta),
            "best_acc5": max(0.42, 0.42 + delta),
            "semantic_delta_vs_phase_a_acc1": 0.02 + delta,
            "semantic_delta_vs_phase_a_acc5": 0.02 + delta,
            "text_added_value_acc1": delta,
            "text_added_value_acc5": delta,
            "best_delta_vs_phase_a_acc1": 0.02 + max(0.0, delta),
            "best_delta_vs_phase_a_acc5": 0.02 + max(0.0, delta),
        }
        return write_teacher_refinement_report(
            cache_path,
            metadata,
            result,
            {"visual": [], "text": []},
        )

    def test_report_is_written_next_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, 42, 0.01)
            self.assertEqual(
                path,
                report_path_for_cache(Path(directory) / "teacher_seed_42.pt"),
            )
            self.assertTrue(path.is_file())

    def test_three_compatible_seeds_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(directory, 42, 0.01),
                self._write(directory, 43, 0.02),
                self._write(directory, 44, 0.03),
            ]
            summary = summarize_teacher_refinement_reports(paths)
            self.assertEqual(summary["run_count"], 3)
            self.assertEqual(summary["positive_text_value_acc1_runs"], 3)
            self.assertTrue(summary["all_text_value_acc1_positive"])
            self.assertAlmostEqual(
                summary["mean_text_added_value_acc1"], 0.02
            )
            self.assertEqual([run["seed"] for run in summary["runs"]], [42, 43, 44])

    def test_non_seed_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(directory, 42, 0.01),
                self._write(directory, 43, 0.02),
                self._write(directory, 44, 0.03, semantic_weight=0.5),
            ]
            with self.assertRaisesRegex(ValueError, "incompatible"):
                summarize_teacher_refinement_reports(paths)

    def test_minimum_seed_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [self._write(directory, 42, 0.01)]
            with self.assertRaisesRegex(ValueError, "at least 3"):
                summarize_teacher_refinement_reports(paths)


if __name__ == "__main__":
    unittest.main()
