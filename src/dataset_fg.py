import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data_config import UNSEEN_CLASSES
from src.dataset import load_image, normal_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_COUNTS = {
    "categories": 125,
    "photos": 12_500,
    "sketches": 70_281,
    "seen_photos": 10_400,
    "seen_sketches": 57_587,
    "unseen_photos": 2_100,
    "unseen_sketches": 12_694,
}


def photo_id_from_sketch(path):
    """Return the paired photo ID from `<photo-id>-<sketch-id>` filenames."""
    stem = Path(path).stem
    if "-" not in stem:
        raise ValueError(f"Invalid fine-grained sketch filename: {path}")
    photo_id, sketch_id = stem.rsplit("-", 1)
    if not photo_id or not sketch_id:
        raise ValueError(f"Invalid fine-grained sketch filename: {path}")
    return photo_id


def _image_files(directory):
    return sorted(
        str(path)
        for path in Path(directory).iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class FineGrainedIndex:
    """Validated exact sketch-photo associations for the Sketchy Basic set."""

    def __init__(self, root, dataset="sketchy_2", validate_counts=False):
        if dataset != "sketchy_2":
            raise ValueError(
                "Fine-grained ZS-SBIR currently supports only --dataset sketchy_2."
            )

        self.root = Path(root).resolve()
        self.sketch_root = self.root / "sketch"
        self.photo_root = self.root / "photo"
        if not self.sketch_root.is_dir() or not self.photo_root.is_dir():
            raise FileNotFoundError(
                f"{self.root} must directly contain sketch/ and photo/."
            )

        sketch_categories = {
            path.name
            for path in self.sketch_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        photo_categories = {
            path.name
            for path in self.photo_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if sketch_categories != photo_categories:
            missing_sketch = sorted(photo_categories - sketch_categories)
            missing_photo = sorted(sketch_categories - photo_categories)
            raise RuntimeError(
                "Sketch/photo category mismatch: "
                f"missing_sketch={missing_sketch}, missing_photo={missing_photo}."
            )

        self.categories = sorted(sketch_categories)
        unseen = set(UNSEEN_CLASSES[dataset])
        if not unseen.issubset(sketch_categories):
            missing = sorted(unseen - sketch_categories)
            raise RuntimeError(f"Missing unseen categories: {missing}")
        self.unseen_categories = sorted(unseen)
        self.seen_categories = sorted(sketch_categories - unseen)

        self.photos_by_category = {}
        self.sketches_by_category = {}
        self.photo_by_key = {}
        self.sketch_to_photo = {}

        for category in self.categories:
            photo_paths = _image_files(self.photo_root / category)
            sketch_paths = _image_files(self.sketch_root / category)
            if any(Path(path).stem.startswith("ext_") for path in photo_paths):
                raise RuntimeError(
                    f"Extended photos are not valid for FG-SBIR: {category}."
                )

            photo_ids = [Path(path).stem for path in photo_paths]
            if len(photo_ids) != len(set(photo_ids)):
                raise RuntimeError(f"Duplicate photo IDs in category {category}.")
            by_id = dict(zip(photo_ids, photo_paths))
            self.photos_by_category[category] = photo_paths
            self.sketches_by_category[category] = sketch_paths
            for photo_id, path in by_id.items():
                self.photo_by_key[(category, photo_id)] = path

            for sketch_path in sketch_paths:
                photo_id = photo_id_from_sketch(sketch_path)
                key = (category, photo_id)
                if key not in self.photo_by_key:
                    raise RuntimeError(
                        f"Sketch has no exact paired photo: {sketch_path}"
                    )
                self.sketch_to_photo[sketch_path] = self.photo_by_key[key]

        if validate_counts:
            self._validate_official_counts()

    def _validate_official_counts(self):
        photo_count = sum(map(len, self.photos_by_category.values()))
        sketch_count = sum(map(len, self.sketches_by_category.values()))
        seen_photo_count = sum(
            len(self.photos_by_category[c]) for c in self.seen_categories
        )
        seen_sketch_count = sum(
            len(self.sketches_by_category[c]) for c in self.seen_categories
        )
        unseen_photo_count = photo_count - seen_photo_count
        unseen_sketch_count = sketch_count - seen_sketch_count
        actual = {
            "categories": len(self.categories),
            "photos": photo_count,
            "sketches": sketch_count,
            "seen_photos": seen_photo_count,
            "seen_sketches": seen_sketch_count,
            "unseen_photos": unseen_photo_count,
            "unseen_sketches": unseen_sketch_count,
        }
        mismatches = {
            key: (EXPECTED_COUNTS[key], value)
            for key, value in actual.items()
            if value != EXPECTED_COUNTS[key]
        }
        invalid_photo_categories = {
            category: len(paths)
            for category, paths in self.photos_by_category.items()
            if len(paths) != 100
        }
        if mismatches or invalid_photo_categories:
            raise RuntimeError(
                "Dataset is not the expected Sketchy Basic fine-grained set. "
                f"count_mismatches={mismatches}, "
                f"categories_without_100_photos={invalid_photo_categories}."
            )


class FineGrainedTrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, index=None):
        self.seed = args.seed
        self.max_size = args.max_size
        self.transform = normal_transform(self.max_size)
        self.index = index or FineGrainedIndex(args.root, args.dataset)
        self.all_categories = self.index.seen_categories
        self.category_to_label = {
            category: label for label, category in enumerate(self.all_categories)
        }

        self.all_sketches_path = []
        self.all_photo_paths = []
        self.photo_path_to_index = {}
        self.photo_key_to_instance = {}
        self.category_instance_to_sketch_indices = defaultdict(
            lambda: defaultdict(list)
        )
        self.sample_category_ids = []
        self.sample_instance_ids = []
        self.sample_photo_indices = []

        instance_index = 0
        for category in self.all_categories:
            category_id = self.category_to_label[category]
            for photo_path in self.index.photos_by_category[category]:
                photo_id = Path(photo_path).stem
                self.photo_path_to_index[photo_path] = len(self.all_photo_paths)
                self.all_photo_paths.append(photo_path)
                self.photo_key_to_instance[(category, photo_id)] = instance_index
                instance_index += 1

            for sketch_path in self.index.sketches_by_category[category]:
                photo_path = self.index.sketch_to_photo[sketch_path]
                photo_id = Path(photo_path).stem
                current_instance = self.photo_key_to_instance[(category, photo_id)]
                sample_index = len(self.all_sketches_path)
                self.all_sketches_path.append(sketch_path)
                self.sample_category_ids.append(category_id)
                self.sample_instance_ids.append(current_instance)
                self.sample_photo_indices.append(
                    self.photo_path_to_index[photo_path]
                )
                self.category_instance_to_sketch_indices[category_id][
                    current_instance
                ].append(sample_index)

        self.teacher_sketch_features = None
        self.teacher_photo_features = None

    def set_teacher_features(self, sketch_features, photo_features):
        if len(sketch_features) != len(self.all_sketches_path):
            raise ValueError("Sketch feature cache has the wrong length.")
        if len(photo_features) != len(self.all_photo_paths):
            raise ValueError("Photo feature cache has the wrong length.")
        self.teacher_sketch_features = sketch_features
        self.teacher_photo_features = photo_features

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, sample_key):
        if isinstance(sample_key, tuple):
            _, index = sample_key
        else:
            index = sample_key

        sketch_path = self.all_sketches_path[index]
        photo_index = self.sample_photo_indices[index]
        photo_path = self.all_photo_paths[photo_index]
        sketch = self.transform(load_image(sketch_path, self.max_size))
        photo = self.transform(load_image(photo_path, self.max_size))

        if self.teacher_sketch_features is None:
            teacher_sketch = torch.empty(0)
            teacher_photo = torch.empty(0)
        else:
            teacher_sketch = self.teacher_sketch_features[index]
            teacher_photo = self.teacher_photo_features[photo_index]

        return (
            photo,
            sketch,
            teacher_photo,
            teacher_sketch,
            self.sample_category_ids[index],
            self.sample_instance_ids[index],
        )


class _CyclingPool:
    def __init__(self, values, rng):
        self.values = np.asarray(list(values), dtype=np.int64)
        if len(self.values) == 0:
            raise ValueError("A cycling pool cannot be empty.")
        self.rng = rng
        self.order = self.rng.permutation(self.values)
        self.offset = 0

    def take_distinct(self, count):
        if count > len(self.values):
            raise ValueError(
                f"Cannot take {count} distinct values from {len(self.values)}."
            )
        selected = []
        selected_set = set()
        while len(selected) < count:
            if self.offset == len(self.order):
                self.order = self.rng.permutation(self.values)
                self.offset = 0
            value = int(self.order[self.offset])
            self.offset += 1
            if value not in selected_set:
                selected.append(value)
                selected_set.add(value)
        return selected

    def take_one(self):
        return self.take_distinct(1)[0]


class FineGrainedPKBatchSampler(torch.utils.data.Sampler):
    """Deterministic P-category/K-instance batches for fine-grained learning."""

    def __init__(self, dataset, batch_size, samples_per_category, seed):
        if samples_per_category < 2:
            raise ValueError("samples_per_category must be at least 2.")
        if batch_size % samples_per_category:
            raise ValueError(
                "batch_size must be divisible by samples_per_category."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_category = samples_per_category
        self.categories_per_batch = batch_size // samples_per_category
        self.seed = seed
        self.epoch = 0

        self.instances = {
            int(category): sorted(instance_map)
            for category, instance_map in (
                dataset.category_instance_to_sketch_indices.items()
            )
        }
        if self.categories_per_batch > len(self.instances):
            raise ValueError("The batch requests more categories than available.")
        for category, instance_ids in self.instances.items():
            if len(instance_ids) < samples_per_category:
                raise ValueError(
                    f"Category {category} has only {len(instance_ids)} instances."
                )

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        seed_bytes = hashlib.blake2b(
            f"{self.seed}:{epoch}:fg-pk".encode("utf-8"), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
        category_pool = _CyclingPool(sorted(self.instances), rng)
        instance_pools = {
            category: _CyclingPool(instance_ids, rng)
            for category, instance_ids in self.instances.items()
        }
        sketch_pools = {
            (category, instance): _CyclingPool(indices, rng)
            for category, instance_map in (
                self.dataset.category_instance_to_sketch_indices.items()
            )
            for instance, indices in instance_map.items()
        }

        for _ in range(len(self)):
            categories = category_pool.take_distinct(self.categories_per_batch)
            batch = []
            for category in categories:
                instance_ids = instance_pools[category].take_distinct(
                    self.samples_per_category
                )
                for instance in instance_ids:
                    sample_index = sketch_pools[(category, instance)].take_one()
                    batch.append((epoch, sample_index))
            yield batch


class FineGrainedValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, index, modality):
        if modality not in {"sketch", "photo"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.max_size = args.max_size
        self.transform = normal_transform(self.max_size)
        self.modality = modality
        self.paths = []
        self.category_ids = []
        self.instance_ids = []

        for category_id, category in enumerate(index.unseen_categories):
            photo_ids = {
                Path(path).stem: instance_id
                for instance_id, path in enumerate(
                    index.photos_by_category[category]
                )
            }
            paths = (
                index.sketches_by_category[category]
                if modality == "sketch"
                else index.photos_by_category[category]
            )
            for path in paths:
                photo_id = (
                    photo_id_from_sketch(path)
                    if modality == "sketch"
                    else Path(path).stem
                )
                self.paths.append(path)
                self.category_ids.append(category_id)
                self.instance_ids.append(photo_ids[photo_id])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.transform(load_image(self.paths[index], self.max_size))
        return image, self.category_ids[index], self.instance_ids[index]
