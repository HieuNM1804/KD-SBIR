import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test"
CACHE = "/kaggle/working/teacher_cache/sketchy1_pcsgcd_pairwise_s42_k4_m10.pt"


class SgcdCommandTest(unittest.TestCase):
    def read(self, name):
        return (TEST / name).read_text(encoding="utf-8")

    @staticmethod
    def arguments(text):
        result = {}
        for line in text.splitlines():
            line = line.strip().removesuffix("\\").strip()
            if not line.startswith("--"):
                continue
            parts = line.split(maxsplit=1)
            result[parts[0]] = parts[1] if len(parts) == 2 else True
        return result

    def test_main_is_exact_baseline(self):
        text = self.read("kaggle_main_baseline_train.ipy")
        self.assertIn("--retrieval_head main", text)
        self.assertIn("--lambda_domain 3.0", text)
        self.assertIn("--lambda_modality 1.0", text)
        self.assertNotIn("--lambda_sgcd ", text)

    def test_audit_precedes_full_preparation(self):
        audit = self.read("kaggle_sgcd_audit.ipy")
        prepare = self.read("kaggle_sgcd_prepare.ipy")
        self.assertIn("--sgcd_audit_only", audit)
        self.assertNotIn("--sgcd_force_prepare", audit)
        self.assertIn("--sgcd_prepare_only", prepare)
        for text in (audit, prepare):
            self.assertIn(CACHE, text)
            self.assertIn("--sgcd_min_effect_ratio 1.15", text)
            self.assertIn("--sgcd_min_win_rate 0.53", text)
            self.assertIn("--sgcd_max_random_map_cosine 0.75", text)
            self.assertIn("--sgcd_effect_mode pairwise", text)
            self.assertIn("--sgcd_negative_topk 3", text)

    def test_primary_and_controls_share_all_non_target_settings(self):
        expected = {
            "kaggle_sgcd_train.ipy": "verified",
            "kaggle_sgcd_local_control.ipy": "local",
            "kaggle_sgcd_random_control.ipy": "random",
            "kaggle_sgcd_shuffle_control.ipy": "shuffled",
        }
        for name, target in expected.items():
            text = self.read(name)
            self.assertIn(CACHE, text)
            self.assertIn(f"--sgcd_target {target}", text)
            self.assertIn("--lambda_domain 3.0", text)
            self.assertIn("--lambda_modality 1.0", text)
            self.assertIn("--lambda_sgcd_where 1.0", text)
            self.assertIn("--lambda_sgcd_rank 0.50", text)
            self.assertIn("--sgcd_rank_margin 0.20", text)
        standalone = self.read("kaggle_sgcd_standalone.ipy")
        self.assertIn("--lambda_domain 0.0", standalone)
        self.assertIn("--lambda_modality 0.0", standalone)

    def test_pairwise_ablation_cells_are_explicit(self):
        no_rank = self.read("kaggle_pcsgcd_no_rank_ablation.ipy")
        self.assertIn(CACHE, no_rank)
        self.assertIn("--sgcd_effect_mode pairwise", no_rank)
        self.assertIn("--lambda_sgcd_rank 0.0", no_rank)
        positive_prepare = self.read("kaggle_sgcd_positive_prepare.ipy")
        positive_train = self.read("kaggle_sgcd_positive_ablation.ipy")
        for text in (positive_prepare, positive_train):
            self.assertIn("--sgcd_effect_mode positive", text)
            self.assertIn("sketchy1_sgcd_positive_s42_k4_m10.pt", text)
        self.assertIn("--sgcd_prepare_only", positive_prepare)
        self.assertIn("--sgcd_min_win_rate 0.60", positive_prepare)
        self.assertIn("--lambda_sgcd_rank 0.50", positive_train)

    def test_native_prompt_component_ablations_are_matched(self):
        expected = {
            "kaggle_sgcd_native_where_ablation.ipy": ("1.0", "0.0", "0.0"),
            "kaggle_sgcd_native_where_what_ablation.ipy": ("1.0", "0.25", "0.0"),
            "kaggle_sgcd_native_where_effect_ablation.ipy": ("1.0", "0.0", "0.25"),
            "kaggle_sgcd_native_prompt_train.ipy": ("1.0", "0.25", "0.25"),
        }
        configurations = {}
        for name, (where, what, effect) in expected.items():
            text = self.read(name)
            configurations[name] = self.arguments(text)
            self.assertIn(CACHE, text)
            self.assertIn("--sgcd_student_mode native_prompt", text)
            self.assertIn("--sgcd_target verified", text)
            self.assertIn("--lambda_domain 3.0", text)
            self.assertIn("--lambda_modality 1.0", text)
            self.assertIn(f"--lambda_sgcd_where {where}", text)
            self.assertIn(f"--lambda_sgcd_what {what}", text)
            self.assertIn(f"--lambda_sgcd_effect {effect}", text)
            self.assertIn("--lambda_sgcd_anchor 0.0", text)
            self.assertIn("--lambda_sgcd_rank 0.0", text)
            self.assertIn("--sgcd_beta 0.0", text)
            self.assertIn("--seed 42", text)
            self.assertIn("--epochs 5", text)
        varying = {
            "--lambda_sgcd_where",
            "--lambda_sgcd_what",
            "--lambda_sgcd_effect",
        }
        reference = configurations["kaggle_sgcd_native_prompt_train.ipy"]
        reference = {
            key: value for key, value in reference.items() if key not in varying
        }
        for name, arguments in configurations.items():
            current = {
                key: value for key, value in arguments.items() if key not in varying
            }
            self.assertEqual(current, reference, name)

    def test_native_target_controls_only_change_the_target(self):
        expected = {
            "kaggle_sgcd_native_where_effect_ablation.ipy": "verified",
            "kaggle_sgcd_native_random_control.ipy": "random",
            "kaggle_sgcd_native_shuffle_control.ipy": "shuffled",
        }
        configurations = {}
        for name, target in expected.items():
            text = self.read(name)
            arguments = self.arguments(text)
            configurations[name] = arguments
            self.assertEqual(arguments["--sgcd_student_mode"], "native_prompt")
            self.assertEqual(arguments["--sgcd_target"], target)
            self.assertEqual(arguments["--lambda_sgcd_where"], "1.0")
            self.assertEqual(arguments["--lambda_sgcd_what"], "0.0")
            self.assertEqual(arguments["--lambda_sgcd_effect"], "0.25")
            self.assertEqual(arguments["--lambda_sgcd_anchor"], "0.0")
            self.assertEqual(arguments["--lambda_sgcd_rank"], "0.0")
            self.assertEqual(arguments["--sgcd_beta"], "0.0")
            self.assertEqual(arguments["--seed"], "42")
        reference = configurations["kaggle_sgcd_native_where_effect_ablation.ipy"]
        for name, arguments in configurations.items():
            self.assertEqual(
                {
                    key: value
                    for key, value in arguments.items()
                    if key != "--sgcd_target"
                },
                {
                    key: value
                    for key, value in reference.items()
                    if key != "--sgcd_target"
                },
                name,
            )

    def test_native_seed_replications_have_matched_baselines(self):
        main_reference = self.arguments(self.read("kaggle_main_baseline_train.ipy"))
        method_reference = self.arguments(
            self.read("kaggle_sgcd_native_where_effect_ablation.ipy")
        )
        for seed in (43, 44):
            main = self.arguments(self.read(f"kaggle_main_baseline_s{seed}.ipy"))
            method = self.arguments(
                self.read(f"kaggle_sgcd_native_where_effect_s{seed}.ipy")
            )
            self.assertEqual(main["--seed"], str(seed))
            self.assertEqual(method["--seed"], str(seed))
            self.assertEqual(
                {key: value for key, value in main.items() if key != "--seed"},
                {
                    key: value
                    for key, value in main_reference.items()
                    if key != "--seed"
                },
            )
            self.assertEqual(
                {key: value for key, value in method.items() if key != "--seed"},
                {
                    key: value
                    for key, value in method_reference.items()
                    if key != "--seed"
                },
            )


if __name__ == "__main__":
    unittest.main()
