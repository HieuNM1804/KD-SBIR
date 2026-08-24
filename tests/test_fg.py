import tempfile
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data_config import UNSEEN_CLASSES
from src.dataset_fg import (
    FineGrainedIndex,
    FineGrainedPKBatchSampler,
    photo_id_from_sketch,
)
from src.losses_fg import fine_grained_teacher_triplet_loss
from src.model_fg import (
    better_acc1_acc5,
    default_teacher_cache_path,
    fine_grained_accuracy,
)


class FineGrainedDataTests(unittest.TestCase):
    def test_photo_id_from_sketch(self):
        self.assertEqual(
            photo_id_from_sketch("sketch/cat/n0123_45-7.png"),
            "n0123_45",
        )
        with self.assertRaises(ValueError):
            photo_id_from_sketch("invalid.png")

    def test_exact_pair_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = list(UNSEEN_CLASSES["sketchy_2"]) + ["airplane"]
            for category in categories:
                photo_dir = root / "photo" / category
                sketch_dir = root / "sketch" / category
                photo_dir.mkdir(parents=True)
                sketch_dir.mkdir(parents=True)
                (photo_dir / f"{category}_001.jpg").touch()
                (sketch_dir / f"{category}_001-1.png").touch()
            index = FineGrainedIndex(
                root, dataset="sketchy_2", validate_counts=False
            )
            sketch = str(root / "sketch" / "airplane" / "airplane_001-1.png")
            photo = str(root / "photo" / "airplane" / "airplane_001.jpg")
            self.assertEqual(Path(index.sketch_to_photo[sketch]), Path(photo))


class _FakeDataset:
    def __init__(self):
        self.category_instance_to_sketch_indices = {}
        sample_index = 0
        for category in range(4):
            instance_map = {}
            for instance_offset in range(10):
                instance = category * 100 + instance_offset
                instance_map[instance] = [sample_index, sample_index + 1]
                sample_index += 2
            self.category_instance_to_sketch_indices[category] = instance_map
        self.length = sample_index

    def __len__(self):
        return self.length


class FineGrainedSamplerTests(unittest.TestCase):
    def test_pk_batches_have_distinct_categories_and_instances(self):
        dataset = _FakeDataset()
        sampler = FineGrainedPKBatchSampler(
            dataset, batch_size=8, samples_per_category=4, seed=42
        )
        reverse = {}
        for category, instance_map in (
            dataset.category_instance_to_sketch_indices.items()
        ):
            for instance, indices in instance_map.items():
                for index in indices:
                    reverse[index] = (category, instance)

        batch = next(iter(sampler))
        pairs = [reverse[sample_index] for _, sample_index in batch]
        categories = sorted({category for category, _ in pairs})
        self.assertEqual(len(categories), 2)
        for category in categories:
            instances = [
                instance
                for current_category, instance in pairs
                if current_category == category
            ]
            self.assertEqual(len(instances), 4)
            self.assertEqual(len(set(instances)), 4)

    def test_sampler_is_reproducible(self):
        dataset = _FakeDataset()
        first = FineGrainedPKBatchSampler(dataset, 8, 4, seed=42)
        second = FineGrainedPKBatchSampler(dataset, 8, 4, seed=42)
        self.assertEqual(next(iter(first)), next(iter(second)))


class FineGrainedCacheTests(unittest.TestCase):
    def test_student_settings_reuse_cache_but_teacher_settings_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_index = SimpleNamespace(
                categories=["cat"],
                sketches_by_category={"cat": [str(root / "sketch.png")]},
                photos_by_category={"cat": [str(root / "photo.jpg")]},
            )
            dataset = SimpleNamespace(max_size=224, index=fake_index)
            args = SimpleNamespace(
                root=str(root),
                dataset="sketchy_2",
                teacher_cache_dir=str(root / "cache"),
                teacher_n_ctx_visual=10,
                teacher_prompt_depth=12,
                teacher_prompt_std=0.02,
                teacher_prompt_seed=42,
                teacher_prompt_gradient_checkpointing=True,
                teacher_prompt_lr=3e-2,
                teacher_momentum=0.9,
                teacher_weight_decay=1e-3,
                teacher_pretrain_epochs=2,
                teacher_pretrain_batch_size=64,
                lambda_teacher_retrieval=1.5,
                teacher_triplet_margin=0.2,
                teacher_scheduler_step_size=5,
                teacher_scheduler_gamma=0.1,
                samples_per_category=8,
                seed=42,
                lr=1e-2,
            )
            original = default_teacher_cache_path(args, dataset)
            student_change = copy(args)
            student_change.lr = 1e-3
            self.assertEqual(
                original,
                default_teacher_cache_path(student_change, dataset),
            )
            teacher_change = copy(args)
            teacher_change.teacher_prompt_lr = 1e-2
            self.assertNotEqual(
                original,
                default_teacher_cache_path(teacher_change, dataset),
            )


class FineGrainedLossAndMetricTests(unittest.TestCase):
    def test_acc1_primary_acc5_tie_break(self):
        self.assertTrue(better_acc1_acc5(0.2, 0.4, 0.1, 0.9))
        self.assertTrue(better_acc1_acc5(0.2, 0.5, 0.2, 0.4))
        self.assertFalse(better_acc1_acc5(0.2, 0.4, 0.2, 0.4))
        self.assertFalse(better_acc1_acc5(0.1, 0.9, 0.2, 0.4))

    def test_teacher_triplet_uses_exact_pair_and_same_category_negatives(self):
        features = torch.eye(4)
        categories = torch.tensor([0, 0, 1, 1])
        instances = torch.tensor([0, 1, 2, 3])
        loss = fine_grained_teacher_triplet_loss(
            features, features, categories, instances, margin=0.2
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_micro_acc_at_1_and_5(self):
        gallery = torch.eye(100)
        query = torch.zeros(3, 100)
        query[0, 0] = 1.0
        query[1, 0] = 3.0
        query[1, 2] = 2.0
        query[1, 1] = 1.0
        query[2, :5] = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0])
        query[2, 5] = 1.0
        acc1, acc5 = fine_grained_accuracy(
            query,
            gallery,
            torch.zeros(3, dtype=torch.long),
            torch.zeros(100, dtype=torch.long),
            torch.tensor([0, 1, 5]),
            torch.arange(100),
        )
        self.assertAlmostEqual(acc1.item(), 1 / 3, places=6)
        self.assertAlmostEqual(acc5.item(), 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
