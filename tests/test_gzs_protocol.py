import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.dataset import ValidDataset


class GZSDatasetProtocolTest(unittest.TestCase):
    def test_unseen_queries_and_all_category_gallery_share_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            unseen = "bat"
            seen = "airplane"
            for modality, category, filename in (
                ("sketch", unseen, "query.png"),
                ("photo", unseen, "unseen.jpg"),
                ("photo", seen, "seen.jpg"),
            ):
                directory = root / modality / category
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).touch()

            # Sketchy-2 normally has 21 unseen classes.  Create empty category
            # directories for the remainder so the dataset's fail-fast check
            # exercises the actual global-label construction.
            from src.data_config import UNSEEN_CLASSES

            for category in UNSEEN_CLASSES["sketchy_2"]:
                (root / "photo" / category).mkdir(parents=True, exist_ok=True)
                (root / "sketch" / category).mkdir(parents=True, exist_ok=True)

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
            self.assertEqual(
                queries.category_to_label[unseen],
                gallery.category_to_label[unseen],
            )


if __name__ == "__main__":
    unittest.main()
