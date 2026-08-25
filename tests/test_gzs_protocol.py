import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.data_config import GENERALIZED_CLASSES, UNSEEN_CLASSES
from src.dataset import ValidDataset


class GZSDatasetProtocolTest(unittest.TestCase):
    def test_unseen_queries_and_fixed_seen_subset_gallery_share_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            unseen = "bat"
            seen = "teapot"
            excluded_seen = "airplane"
            for modality, category, filename in (
                ("sketch", unseen, "query.png"),
                ("photo", unseen, "unseen.jpg"),
                ("photo", seen, "seen.jpg"),
                ("photo", excluded_seen, "excluded.jpg"),
            ):
                directory = root / modality / category
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).touch()

            for category in UNSEEN_CLASSES["sketchy_2"]:
                (root / "photo" / category).mkdir(parents=True, exist_ok=True)
                (root / "sketch" / category).mkdir(parents=True, exist_ok=True)
            for category in GENERALIZED_CLASSES["sketchy_2"]:
                (root / "photo" / category).mkdir(parents=True, exist_ok=True)

            args = SimpleNamespace(
                root=str(root),
                dataset="sketchy_2",
                max_size=224,
            )
            queries = ValidDataset(args, mode="sketch", protocol="gzs")
            gallery = ValidDataset(args, mode="photo", protocol="gzs")

            self.assertEqual(len(queries), 1)
            self.assertEqual(len(gallery), 2)
            self.assertEqual(queries.category_to_label, gallery.category_to_label)
            self.assertIn(seen, gallery.label_classes)
            self.assertNotIn(excluded_seen, gallery.label_classes)
            self.assertEqual(
                queries.category_to_label[unseen],
                gallery.category_to_label[unseen],
            )

    def test_generalized_classes_are_seen_only(self):
        self.assertEqual(len(GENERALIZED_CLASSES["sketchy_2"]), 9)
        self.assertEqual(len(GENERALIZED_CLASSES["tuberlin"]), 5)
        for dataset, classes in GENERALIZED_CLASSES.items():
            self.assertEqual(len(classes), len(set(classes)))
            self.assertTrue(set(classes).isdisjoint(UNSEEN_CLASSES[dataset]))


if __name__ == "__main__":
    unittest.main()
