"""Explicit category/instance positives and held-out seen validation.

FG filenames must be <photo-id>-<sketch-id>.<ext>. Category mode never
interprets its random same-category photo as an exact instance association.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data_config import UNSEEN_CLASSES
from src.dataset import load_image, normal_transform, sample_seed

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def image_files(directory):
    return sorted(str(p) for p in Path(directory).iterdir()
                  if p.is_file() and not p.name.startswith('.')
                  and p.suffix.lower() in EXTENSIONS)


def paired_photo_id(path):
    pieces = Path(path).stem.rsplit('-', 1)
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f'FG sketch must have <photo-id>-<sketch-id> filename: {path}')
    return pieces[0]


def holdout_ids(ids, fraction, rng, min_train=1):
    if len(ids) <= min_train:
        raise ValueError('Not enough samples for a disjoint held-out seen split')
    count = min(len(ids) - min_train, max(1, round(len(ids) * fraction)))
    return set(int(i) for i in rng.permutation(ids)[:count])


class CorrespondenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.seed, self.max_size = args.seed, args.max_size
        self.transform = normal_transform(args.max_size)
        root = Path(args.root)
        if not (root / 'sketch').is_dir() or not (root / 'photo').is_dir():
            raise FileNotFoundError('Dataset root must directly contain sketch/ and photo/')
        categories = sorted(p.name for p in (root / 'sketch').iterdir()
                            if p.is_dir() and not p.name.startswith('.'))
        photo_categories = sorted(p.name for p in (root / 'photo').iterdir()
                                  if p.is_dir() and not p.name.startswith('.'))
        if categories != photo_categories:
            raise ValueError('Sketch/photo category directories differ')
        unseen = set(UNSEEN_CLASSES[args.dataset])
        if unseen - set(categories):
            raise ValueError(f'Missing unseen categories: {sorted(unseen - set(categories))}')
        self.all_categories = sorted(set(categories) - unseen)
        self.category_to_label = {c: i for i, c in enumerate(self.all_categories)}
        self.all_sketches_path, self.all_photo_paths = [], []
        self.sketch_categories, self.photo_categories = [], []
        self.paired_indices = []
        self.train_indices, self.val_sketch_indices, self.val_photo_indices = [], [], []
        self.fit_photos = {}
        self.category_to_sketches = {}
        self.instance_to_sketches = defaultdict(list)
        self.category_to_instances = defaultdict(list)
        self.region_targets = None
        self.teacher_sketch_features = self.teacher_photo_features = None
        for category in self.all_categories:
            cid = self.category_to_label[category]
            photos = image_files(root / 'photo' / category)
            sketches = image_files(root / 'sketch' / category)
            if not photos or not sketches:
                raise ValueError(f'Empty category: {category}')
            if args.retrieval_protocol == 'fg' and any(Path(p).stem.startswith('ext_') for p in photos):
                raise ValueError('FG requires paired basic photos, not Sketchy Extended photos')
            if args.retrieval_protocol == 'fg' and args.fg_photos_per_category and len(photos) != args.fg_photos_per_category:
                raise ValueError(f'FG category {category} has {len(photos)} photos; expected {args.fg_photos_per_category}')
            ids = [Path(p).stem for p in photos]
            if len(set(ids)) != len(ids):
                raise ValueError(f'Duplicate photo IDs: {category}')
            pidx = list(range(len(self.all_photo_paths), len(self.all_photo_paths) + len(photos)))
            sidx = list(range(len(self.all_sketches_path), len(self.all_sketches_path) + len(sketches)))
            self.all_photo_paths.extend(photos)
            self.all_sketches_path.extend(sketches)
            self.photo_categories.extend([cid] * len(photos))
            self.sketch_categories.extend([cid] * len(sketches))
            rng = np.random.default_rng(sample_seed(args.split_seed, 0, cid))
            by_id = dict(zip(ids, pidx))
            if args.retrieval_protocol == 'fg':
                pairs = []
                for path in sketches:
                    key = paired_photo_id(path)
                    if key not in by_id:
                        raise ValueError(f'Sketch has no exact paired photo: {path}')
                    pairs.append(by_id[key])
                eligible = sorted(set(pairs))
                held_photos = holdout_ids(eligible, args.seen_val_fraction, rng, min_train=2)
                held_sketches = {s for s, p in zip(sidx, pairs) if p in held_photos}
            else:
                pairs = [-1] * len(sketches)
                held_photos = holdout_ids(pidx, args.seen_val_fraction, rng)
                held_sketches = holdout_ids(sidx, args.seen_val_fraction, rng)
            self.paired_indices.extend(pairs)
            self.val_sketch_indices.extend(sorted(held_sketches))
            self.val_photo_indices.extend(sorted(held_photos))
            fit_sketches = [s for s in sidx if s not in held_sketches]
            self.train_indices.extend(fit_sketches)
            self.fit_photos[cid] = [p for p in pidx if p not in held_photos]
            self.category_to_sketches[cid] = fit_sketches
            if args.retrieval_protocol == 'fg':
                for s in fit_sketches:
                    self.instance_to_sketches[self.paired_indices[s]].append(s)
                self.category_to_instances[cid] = sorted(set(self.paired_indices[s] for s in fit_sketches))
        if not self.train_indices:
            raise ValueError('No training sketches')
        split = {'protocol': args.retrieval_protocol, 'split_seed': args.split_seed,
                 'train_sketches': self.train_indices, 'val_sketches': self.val_sketch_indices,
                 'val_photos': self.val_photo_indices, 'pairs': self.paired_indices,
                 'fit_photos': self.fit_photos}
        self.split_digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        cid = self.sketch_categories[index]
        if self.args.retrieval_protocol == 'fg':
            p = self.paired_indices[index]
            positive_id = p
        else:
            choices = self.fit_photos[cid]
            rng = np.random.default_rng(sample_seed(self.seed, epoch, index))
            p = choices[int(rng.integers(len(choices)))]
            positive_id = cid
        result = {'photo': self.transform(load_image(self.all_photo_paths[p], self.max_size)),
                  'sketch': self.transform(load_image(self.all_sketches_path[index], self.max_size)),
                  'positive_id': positive_id, 'category': cid, 'photo_index': p, 'sketch_index': index}
        if self.region_targets is not None:
            for modality, i in [('photo', p), ('sketch', index)]:
                targets = self.region_targets[modality]
                for name in ('global', 'crops', 'visibility'):
                    result[modality + '_' + name] = targets[name][i]
        return result

    def seen_validation(self, modality):
        ids = self.val_sketch_indices if modality == 'sketch' else self.val_photo_indices
        paths = self.all_sketches_path if modality == 'sketch' else self.all_photo_paths
        cats = self.sketch_categories if modality == 'sketch' else self.photo_categories
        if self.args.retrieval_protocol == 'fg':
            labels = [self.paired_indices[i] if modality == 'sketch' else i for i in ids]
        else:
            labels = [cats[i] for i in ids]
        return RetrievalView([paths[i] for i in ids], labels, self.max_size,
                             [cats[i] for i in ids] if self.args.retrieval_protocol == 'fg' else None)

    def unseen_validation(self, modality):
        root = Path(self.args.root)
        photos, sketches, labels_photo, labels_sketch = [], [], [], []
        cats_photo, cats_sketch = [], []
        for cid, category in enumerate(sorted(UNSEEN_CLASSES[self.args.dataset])):
            pp = image_files(root / 'photo' / category)
            ss = image_files(root / 'sketch' / category)
            by_id = {Path(p).stem: len(photos) + i for i, p in enumerate(pp)}
            if len(by_id) != len(pp):
                raise ValueError(f'Duplicate unseen photo IDs: {category}')
            if self.args.retrieval_protocol == 'fg':
                if self.args.fg_photos_per_category and len(pp) != self.args.fg_photos_per_category:
                    raise ValueError(f'Unseen FG category {category} has {len(pp)} photos; expected {self.args.fg_photos_per_category}')
                if any(Path(p).stem.startswith('ext_') for p in pp):
                    raise ValueError('FG evaluation does not accept extended photos')
                for s in ss:
                    key = paired_photo_id(s)
                    if key not in by_id:
                        raise ValueError(f'Unseen sketch has no exact paired photo: {s}')
                    labels_sketch.append(by_id[key])
                labels_photo.extend(range(len(photos), len(photos) + len(pp)))
            else:
                labels_sketch.extend([cid] * len(ss))
                labels_photo.extend([cid] * len(pp))
            photos.extend(pp)
            sketches.extend(ss)
            cats_photo.extend([cid] * len(pp)); cats_sketch.extend([cid] * len(ss))
        return RetrievalView(sketches if modality == 'sketch' else photos,
                             labels_sketch if modality == 'sketch' else labels_photo, self.max_size,
                             (cats_sketch if modality == 'sketch' else cats_photo) if self.args.retrieval_protocol == 'fg' else None)


class RetrievalView(Dataset):
    def __init__(self, paths, labels, size, categories=None):
        self.paths, self.labels = paths, labels
        self.categories = categories
        self.transform, self.size = normal_transform(size), size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        label = self.labels[index] if self.categories is None else torch.tensor([self.labels[index], self.categories[index]])
        return self.transform(load_image(self.paths[index], self.size)), label


class BalancedCorrespondenceSampler(Sampler):
    """Same seeded batches for GT/global/correspondence controls; resume-safe."""
    def __init__(self, dataset, batch_size, classes_per_batch, seed, steps=0):
        if batch_size < 2 or not 1 <= classes_per_batch <= batch_size or batch_size % classes_per_batch:
            raise ValueError('batch_size must be divisible by classes_per_batch and >= 2')
        self.dataset, self.batch_size, self.classes_per_batch = dataset, batch_size, classes_per_batch
        self.seed, self.epoch = seed, 0
        self.steps = steps or math.ceil(len(dataset.train_indices) / batch_size)

    def __len__(self):
        return self.steps

    def __iter__(self):
        epoch = self.epoch_source.current_epoch if hasattr(self, 'epoch_source') else self.epoch
        self.epoch = epoch + 1
        rng = np.random.default_rng(sample_seed(self.seed, epoch, -1))
        categories = sorted(self.dataset.category_to_sketches)
        count = self.batch_size // self.classes_per_batch
        for _ in range(self.steps):
            chosen = rng.choice(categories, self.classes_per_batch, replace=len(categories) < self.classes_per_batch)
            batch = []
            for category in chosen:
                category = int(category)
                if self.dataset.args.retrieval_protocol == 'fg':
                    instances = self.dataset.category_to_instances[category]
                    selected = rng.choice(instances, count, replace=len(instances) < count)
                    samples = [int(rng.choice(self.dataset.instance_to_sketches[int(i)])) for i in selected]
                else:
                    indices = self.dataset.category_to_sketches[category]
                    samples = [int(i) for i in rng.choice(indices, count, replace=len(indices) < count)]
                batch.extend((epoch, i) for i in samples)
            rng.shuffle(batch)
            yield batch
