import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test"


class RsedCommandTest(unittest.TestCase):
    def read(self, name):
        return (TEST / name).read_text(encoding="utf-8")

    def test_main_baseline_selects_original_head_without_rsed(self):
        text = self.read("kaggle_main_baseline_train.ipy")
        self.assertIn("--retrieval_head main", text)
        self.assertIn("--lambda_domain 3.0", text)
        self.assertIn("--lambda_modality 1.0", text)
        self.assertNotIn("--lambda_rsed ", text)

    def test_prepare_and_all_controls_share_the_same_cache(self):
        names = (
            "kaggle_rsed_prepare.ipy",
            "kaggle_rsed_train.ipy",
            "kaggle_rsed_attention_control.ipy",
            "kaggle_rsed_random_control.ipy",
            "kaggle_rsed_shuffle_control.ipy",
            "kaggle_rsed_standalone.ipy",
        )
        cache = "/kaggle/working/teacher_cache/sketchy1_rsed_s42_m10_g1.pt"
        for name in names:
            text = self.read(name)
            self.assertIn(cache, text, name)
            self.assertIn("--retrieval_head rsed", text, name)

    def test_controls_change_only_target_and_standalone_main_weights(self):
        expected = {
            "kaggle_rsed_train.ipy": "retrieval",
            "kaggle_rsed_attention_control.ipy": "attention",
            "kaggle_rsed_random_control.ipy": "random",
            "kaggle_rsed_shuffle_control.ipy": "shuffled",
        }
        for name, target in expected.items():
            text = self.read(name)
            self.assertIn(f"--rsed_target {target}", text)
            self.assertIn("--lambda_domain 3.0", text)
            self.assertIn("--lambda_modality 1.0", text)
        standalone = self.read("kaggle_rsed_standalone.ipy")
        self.assertIn("--rsed_target retrieval", standalone)
        self.assertIn("--lambda_domain 0.0", standalone)
        self.assertIn("--lambda_modality 0.0", standalone)


if __name__ == "__main__":
    unittest.main()
