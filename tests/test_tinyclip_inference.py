import unittest

import torch

from src.infer_tinyclip_suite import parse_model_keys, retrieval_metrics
from src.tinyclip_inference import MODEL_SPECS


class TinyCLIPInferenceTests(unittest.TestCase):
    def test_registry_contains_requested_suite_in_order(self):
        self.assertEqual(list(MODEL_SPECS), ["8m", "22m", "40m", "45m", "61m"])

    def test_model_parser_rejects_duplicate_or_unknown_models(self):
        self.assertEqual(parse_model_keys("8m,40m"), ["8m", "40m"])
        with self.assertRaises(Exception):
            parse_model_keys("8m,8m")
        with self.assertRaises(Exception):
            parse_model_keys("8m,99m")

    def test_retrieval_metrics_use_class_relevance(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gallery = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        )
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        metrics = retrieval_metrics(
            queries,
            gallery,
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 1, 1]),
            precision_k=2,
            query_batch_size=1,
        )
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
