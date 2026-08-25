import unittest

from src.dataset import (
    canonical_category_name,
    select_cross_dataset_classes,
)


class CrossDatasetProtocolTest(unittest.TestCase):
    def test_category_normalization_handles_dataset_spelling(self):
        self.assertEqual(
            canonical_category_name("hot-air_balloon"),
            canonical_category_name("hot air balloon"),
        )
        self.assertEqual(
            canonical_category_name("car_(sedan)"),
            canonical_category_name("car"),
        )

    def test_only_target_test_classes_unseen_in_source_are_kept(self):
        source_seen = ["airplane", "hot-air_balloon", "car_(sedan)"]
        target_test = ["airplane", "hot air balloon", "car", "windmill"]
        self.assertEqual(
            select_cross_dataset_classes(source_seen, target_test),
            ["windmill"],
        )


if __name__ == "__main__":
    unittest.main()
