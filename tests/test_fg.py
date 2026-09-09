import tempfile
import unittest
from copy import copy
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import torch

from clip import clip
from clip.model import CLIP, VisionTransformer
from src.data_config import UNSEEN_CLASSES
from src.dataset_fg import (
    FineGrainedFullGalleryBatchSampler,
    FineGrainedIndex,
    photo_id_from_sketch,
)
from src.losses_fg import (
    fine_grained_teacher_infonce_loss,
    full_gallery_relational_kd_loss,
    image_conditioned_text_classification_loss,
)
from src.image_text_prompts import (
    ImageConditionedTextPromptLearner,
    PatchToTextContexts,
)
from src.model_fg import (
    FineGrainedCustomCLIP,
    FineGrainedZS_SBIR,
    better_acc1_acc5,
    default_teacher_cache_path,
    fine_grained_accuracy,
    fine_grained_train_metric_ids,
)
from src.model import _reduce_on_plateau_patience
from src.train_fg import build_parser


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
                student_n_ctx_text=8,
                teacher_n_ctx_text=12,
                text_prompt_gate_init=0.1,
                teacher_text_prompt_seed=50042,
                teacher_text_prompt_lr=1e-3,
                teacher_text_prompt_weight_decay=1e-4,
                teacher_text_cls_temperature=0.07,
                lambda_teacher_text_cls=1.0,
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
            teacher_text_change = copy(args)
            teacher_text_change.teacher_text_prompt_lr = 2e-3
            self.assertNotEqual(
                original,
                default_teacher_cache_path(teacher_text_change, dataset),
            )
            student_text_count_change = copy(args)
            student_text_count_change.student_n_ctx_text = 4
            self.assertEqual(
                original,
                default_teacher_cache_path(student_text_count_change, dataset),
            )
            teacher_text_count_change = copy(args)
            teacher_text_count_change.teacher_n_ctx_text = 4
            self.assertNotEqual(
                original,
                default_teacher_cache_path(teacher_text_count_change, dataset),
            )


class FineGrainedLossAndMetricTests(unittest.TestCase):
    def test_visual_encoder_returns_only_real_spatial_patch_tokens(self):
        visual = VisionTransformer(
            input_resolution=8,
            patch_size=4,
            width=64,
            layers=2,
            heads=1,
            output_dim=16,
        )
        pooled, patches = visual(
            torch.randn(2, 3, 8, 8),
            prompt=torch.randn(2, 64),
            compound_prompts=[torch.randn(2, 64)],
            return_patch_tokens=True,
        )
        self.assertEqual(pooled.shape, (2, 16))
        self.assertEqual(patches.shape, (2, 4, 64))

    def test_patch_projection_creates_requested_image_conditioned_contexts(self):
        module = PatchToTextContexts(
            visual_width=8,
            text_width=6,
            context_tokens=3,
            seed=42,
        )
        patches = torch.randn(2, 12, 8, requires_grad=True)
        base = torch.randn(3, 6, requires_grad=True)
        contexts = module(patches, base)
        self.assertEqual(contexts.shape, (2, 3, 6))
        self.assertFalse(torch.equal(contexts[0], contexts[1]))
        contexts.square().mean().backward()
        self.assertIsNotNone(patches.grad)
        self.assertIsNotNone(base.grad)
        self.assertIsNotNone(module.patch_projection.weight.grad)

    def test_openai_text_prompt_learner_backpropagates_to_image_patches(self):
        text_model = CLIP(
            embed_dim=16,
            image_resolution=8,
            vision_layers=1,
            vision_width=64,
            vision_patch_size=4,
            context_length=77,
            vocab_size=49408,
            transformer_width=32,
            transformer_heads=1,
            transformer_layers=1,
        ).eval().requires_grad_(False)
        learner = ImageConditionedTextPromptLearner(
            text_model=text_model,
            tokenizer=clip.tokenize,
            classnames=("cat", "dog"),
            visual_width=64,
            context_tokens=3,
            seed=42,
            text_backend="openai",
            gradient_checkpointing=True,
        )
        patches = torch.randn(2, 4, 64, requires_grad=True)
        features, contexts = learner(
            text_model,
            patches,
            torch.tensor([0, 1]),
            "sketch",
        )
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(contexts.shape, (2, 3, 32))
        features[:, 0].sum().backward()
        self.assertIsNotNone(patches.grad)
        self.assertIsNotNone(learner.base_context.grad)
        self.assertGreater(patches.grad.abs().sum().item(), 0.0)
        self.assertGreater(learner.base_context.grad.abs().sum().item(), 0.0)

    def test_image_conditioned_text_classification_uses_all_class_negatives(self):
        class_text = torch.eye(4)
        labels = torch.tensor([1, 3])
        images = class_text[labels].clone().requires_grad_(True)
        conditioned = class_text[labels].clone().requires_grad_(True)
        loss = image_conditioned_text_classification_loss(
            images,
            conditioned,
            class_text,
            labels,
            temperature=0.07,
        )
        self.assertLess(loss.item(), 1e-3)
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(conditioned.grad)

    def test_image_conditioned_text_classification_is_autocast_safe(self):
        images = torch.randn(3, 16, requires_grad=True)
        conditioned = torch.randn(3, 16, requires_grad=True)
        class_text = torch.randn(5, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = image_conditioned_text_classification_loss(
                images,
                conditioned,
                class_text,
                torch.tensor([0, 2, 4]),
                temperature=0.07,
            )
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(conditioned.grad)

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
        model.model = torch.nn.Module()
        model.model.visual_prompts = torch.nn.Linear(2, 2)
        model.model.student_text_prompt_learner = torch.nn.Linear(2, 2)
        model.args = SimpleNamespace(
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-3,
            text_prompt_lr=2e-3,
            text_prompt_weight_decay=1e-4,
            scheduler_gamma=0.1,
            scheduler_patience=3,
        )
        config = model.configure_optimizers()
        groups = config["optimizer"].param_groups
        self.assertEqual(
            [group["name"] for group in groups],
            ["visual_prompts", "image_conditioned_text_prompts"],
        )
        self.assertEqual([group["lr"] for group in groups], [1e-2, 2e-3])
        self.assertEqual(config["lr_scheduler"]["monitor"], "fg_selection")
        self.assertIsInstance(
            config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )


class FineGrainedCliTests(unittest.TestCase):
    def test_student_and_teacher_text_prompt_counts_are_independent(self):
        args = build_parser().parse_args(
            [
                "--root",
                "dataset",
                "--student_n_ctx_text",
                "5",
                "--teacher_n_ctx_text",
                "11",
            ]
        )
        self.assertEqual(args.student_n_ctx_text, 5)
        self.assertEqual(args.teacher_n_ctx_text, 11)

    def test_legacy_text_prompt_alias_only_sets_student_count(self):
        args = build_parser().parse_args(
            ["--root", "dataset", "--text_prompt_tokens", "5"]
        )
        self.assertEqual(args.student_n_ctx_text, 5)
        self.assertEqual(args.teacher_n_ctx_text, 8)


if __name__ == "__main__":
    unittest.main()
