import unittest

from src.data_config import CROSS_DATASET_CLASSES
from src.dataset import canonical_category_name


class CrossDatasetProtocolTest(unittest.TestCase):
    def test_paper_subsets_have_expected_sizes_and_no_duplicates(self):
        tuberlin = CROSS_DATASET_CLASSES[("sketchy_1", "tuberlin")]
        quickdraw = CROSS_DATASET_CLASSES[("sketchy_1", "quickdraw")]
        self.assertEqual(len(tuberlin), 21)
        self.assertEqual(len(set(tuberlin)), 21)
        self.assertEqual(len(quickdraw), 11)
        self.assertEqual(len(set(quickdraw)), 11)

    def test_category_normalization_handles_dataset_spelling(self):
        self.assertEqual(
            canonical_category_name("hot-air_balloon"),
            canonical_category_name("hot air balloon"),
        )
        self.assertEqual(
            canonical_category_name("car_(sedan)"),
            canonical_category_name("car"),
        )

if __name__ == "__main__":
    unittest.main()
