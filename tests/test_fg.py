import tempfile
import unittest
from copy import copy
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data_config import UNSEEN_CLASSES
from src.dataset_fg import (
    FineGrainedFullGalleryBatchSampler,
    FineGrainedIndex,
    photo_id_from_sketch,
)
from src.losses_fg import (
    fine_grained_teacher_infonce_loss,
    full_gallery_relational_kd_loss,
)
from src.model_fg import (
    FineGrainedCustomCLIP,
    FineGrainedZS_SBIR,
    better_acc1_acc5,
    default_teacher_cache_path,
    fine_grained_accuracy,
    fine_grained_train_metric_ids,
)
from src.model import (
    VisualPromptLearner,
    _reduce_on_plateau_patience,
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
        self.sample_category_ids = [0] * 5 + [1] * 7 + [2] * 3

    def __len__(self):
        return len(self.sample_category_ids)


class FineGrainedSamplerTests(unittest.TestCase):
    def test_full_gallery_batches_have_one_category_and_cover_all_sketches(self):
        dataset = _FakeDataset()
        sampler = FineGrainedFullGalleryBatchSampler(
            dataset, batch_size=4, seed=42
        )
        batches = list(iter(sampler))
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            categories = {dataset.sample_category_ids[index] for index in batch}
            self.assertEqual(len(categories), 1)
            self.assertLessEqual(len(batch), 4)

    def test_sampler_is_reproducible(self):
        dataset = _FakeDataset()
        first = FineGrainedFullGalleryBatchSampler(dataset, 4, seed=42)
        second = FineGrainedFullGalleryBatchSampler(dataset, 4, seed=42)
        self.assertEqual(list(iter(first)), list(iter(second)))

    def test_category_chunks_are_shuffled_globally(self):
        dataset = SimpleNamespace(
            sample_category_ids=[0] * 20 + [1] * 20 + [2] * 20
        )
        sampler = FineGrainedFullGalleryBatchSampler(
            dataset,
            batch_size=4,
            seed=42,
        )
        category_sequence = [
            dataset.sample_category_ids[batch[0]] for batch in sampler
        ]
        category_runs = [
            category for category, _ in groupby(category_sequence)
        ]
        self.assertGreater(len(category_runs), 3)


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
                teacher_instance_temperature=0.07,
                teacher_scheduler_patience=3,
                teacher_scheduler_gamma=0.1,
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
            scheduler_change = copy(args)
            scheduler_change.teacher_scheduler_patience = 4
            self.assertNotEqual(
                original,
                default_teacher_cache_path(scheduler_change, dataset),
            )


class FineGrainedLossAndMetricTests(unittest.TestCase):
    def test_student_reuses_one_visual_prompt_for_both_modalities(self):
        model = FineGrainedCustomCLIP.__new__(FineGrainedCustomCLIP)
        torch.nn.Module.__init__(model)
        model.visual_prompt = VisualPromptLearner(
            n_ctx_visual=3,
            visual_width=8,
            seed=42,
            prompt_depth=3,
        )

        photo_prompt = model.get_visual_prompt("photo")
        sketch_prompt = model.get_visual_prompt("sketch")

        self.assertIs(photo_prompt[0], sketch_prompt[0])
        self.assertTrue(
            all(
                photo is sketch
                for photo, sketch in zip(photo_prompt[1], sketch_prompt[1])
            )
        )

        photo_prompt[0].sum().backward(retain_graph=True)
        sketch_prompt[0].sum().backward()
        self.assertTrue(
            torch.equal(
                photo_prompt[0].grad,
                torch.full_like(photo_prompt[0], 2.0),
            )
        )

    def test_acc1_primary_acc5_tie_break(self):
        self.assertTrue(better_acc1_acc5(0.2, 0.4, 0.1, 0.9))
        self.assertTrue(better_acc1_acc5(0.2, 0.5, 0.2, 0.4))
        self.assertFalse(better_acc1_acc5(0.2, 0.4, 0.2, 0.4))
        self.assertFalse(better_acc1_acc5(0.1, 0.9, 0.2, 0.4))

    def test_teacher_infonce_uses_exact_pair_and_all_gallery_negatives(self):
        gallery = torch.eye(100)
        queries = gallery[[7, 31, 99]]
        loss = fine_grained_teacher_infonce_loss(
            queries,
            gallery,
            torch.tensor([7, 31, 99]),
            temperature=0.07,
        )
        self.assertLess(loss.item(), 1e-3)

    def test_teacher_infonce_backpropagates_through_all_gallery_photos(self):
        generator = torch.Generator().manual_seed(42)
        queries = torch.randn(3, 16, generator=generator)
        gallery = torch.randn(100, 16, generator=generator, requires_grad=True)
        loss = fine_grained_teacher_infonce_loss(
            queries,
            gallery,
            torch.tensor([7, 31, 99]),
            temperature=1.0,
        )
        loss.backward()
        self.assertTrue(gallery.grad.norm(dim=-1).gt(0).all())

    def test_domain_kd_supports_rectangular_sketch_gallery_logits(self):
        generator = torch.Generator().manual_seed(42)
        sketches = torch.randn(7, 16, generator=generator)
        photos = torch.randn(100, 16, generator=generator)
        loss = full_gallery_relational_kd_loss(
            sketches,
            photos,
            sketches.clone(),
            photos.clone(),
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

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

    def test_seen_metric_ids_preserve_category_local_photo_targets(self):
        dataset = SimpleNamespace(
            sample_category_ids=[0, 0, 1],
            sample_local_photo_indices=[7, 31, 4],
            all_photo_paths=[f"photo-{index}" for index in range(200)],
            category_to_photo_indices={
                0: list(range(100)),
                1: list(range(100, 200)),
            },
        )
        (
            sketch_categories,
            photo_categories,
            sketch_instances,
            photo_instances,
        ) = fine_grained_train_metric_ids(dataset)
        self.assertTrue(torch.equal(sketch_categories, torch.tensor([0, 0, 1])))
        self.assertTrue(torch.equal(sketch_instances, torch.tensor([7, 31, 4])))
        self.assertTrue(
            torch.equal(photo_categories[:100], torch.zeros(100, dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(photo_categories[100:], torch.ones(100, dtype=torch.long))
        )
        self.assertTrue(torch.equal(photo_instances[:100], torch.arange(100)))
        self.assertTrue(torch.equal(photo_instances[100:], torch.arange(100)))

    def test_teacher_train_evaluation_uses_supported_modalities(self):
        model = FineGrainedCustomCLIP.__new__(FineGrainedCustomCLIP)
        torch.nn.Module.__init__(model)
        model.cfg = SimpleNamespace(test_batch_size=128, seed=42)
        calls = []
        gallery = torch.eye(100)

        def materialize(
            paths,
            modality,
            _batch_size,
            _workers,
            _show_progress,
            generator_seed=None,
        ):
            calls.append((modality, generator_seed))
            return gallery[[0]] if modality == "sketch" else gallery

        model._materialize_teacher_features = materialize
        dataset = SimpleNamespace(
            all_sketches_path=["sketch-0"],
            all_photo_paths=[f"photo-{index}" for index in range(100)],
            sample_category_ids=[0],
            sample_local_photo_indices=[0],
            category_to_photo_indices={0: list(range(100))},
        )
        acc1, acc5 = model._validate_teacher_train(
            dataset,
            epoch=1,
            workers=0,
            show_progress=False,
        )
        self.assertEqual([modality for modality, _ in calls], ["sketch", "photo"])
        self.assertEqual(acc1, 1.0)
        self.assertEqual(acc5, 1.0)


class PlateauSchedulerTests(unittest.TestCase):
    def test_lr_drops_after_exactly_three_non_improving_epochs(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=_reduce_on_plateau_patience(3),
            threshold=0.0,
            threshold_mode="abs",
        )
        scheduler.step(0.5)
        scheduler.step(0.4)
        scheduler.step(0.4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        scheduler.step(0.4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_fg_student_scheduler_monitors_validation_selection(self):
        model = FineGrainedZS_SBIR.__new__(FineGrainedZS_SBIR)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Linear(2, 2)
        model.args = SimpleNamespace(
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-3,
            scheduler_gamma=0.1,
            scheduler_patience=3,
        )
        config = model.configure_optimizers()
        self.assertEqual(config["lr_scheduler"]["monitor"], "fg_selection")
        self.assertIsInstance(
            config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )


if __name__ == "__main__":
    unittest.main()
