import os
import glob
import hashlib
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def sample_seed(global_seed, epoch, index):
    """Stable seed that does not depend on which DataLoader worker gets a sample."""
    key = f"{global_seed}:{epoch}:{index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class WorkerInvariantSampler(torch.utils.data.Sampler):
    """Shuffle by epoch and pass the epoch to Dataset.__getitem__."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        generator = torch.Generator().manual_seed(sample_seed(self.seed, epoch, -1))
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter((epoch, index) for index in indices)

    def __len__(self):
        return len(self.dataset)


def normal_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def teacher_photo_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def teacher_sketch_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=7,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                fill=255,
            ),
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def apply_transform_with_seed(transform, image, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transform(image)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.seed = args.seed
        self.max_size = args.max_size
        self.normal_transform = normal_transform()
        self.teacher_photo_transform = teacher_photo_transform()
        self.teacher_sketch_transform = teacher_sketch_transform()

        sketch_root = os.path.join(args.root, "sketch")
        excluded = set(UNSEEN_CLASSES[args.dataset]) | {".ipynb_checkpoints"}
        self.all_categories = sorted(set(os.listdir(sketch_root)) - excluded)
        self.category_to_label = {
            category: label for label, category in enumerate(self.all_categories)
        }
        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketch_paths = sorted(
                glob.glob(os.path.join(args.root, "sketch", category, "*"))
            )
            photo_paths = sorted(
                glob.glob(os.path.join(args.root, "photo", category, "*"))
            )
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, sample_key):
        if isinstance(sample_key, tuple):
            epoch, index = sample_key
        else:
            epoch, index = 0, sample_key

        current_seed = sample_seed(self.seed, epoch, index)
        photo_rng = np.random.default_rng(current_seed)
        filepath = self.all_sketches_path[index]
        category = filepath.split(os.path.sep)[-2]

        photo_paths = self.all_photos_path[category]
        img_path = photo_paths[photo_rng.integers(len(photo_paths))]

        sk_data = load_image(filepath, self.max_size)
        img_data = load_image(img_path, self.max_size)
        sk_tensor = self.normal_transform(sk_data)
        img_tensor = self.normal_transform(img_data)
        teacher_photo_tensor = apply_transform_with_seed(
            self.teacher_photo_transform,
            img_data,
            sample_seed(self.seed + 1, epoch, index),
        )
        teacher_sketch_tensor = apply_transform_with_seed(
            self.teacher_sketch_transform,
            sk_data,
            sample_seed(self.seed + 2, epoch, index),
        )

        return (
            img_tensor,
            sk_tensor,
            teacher_photo_tensor,
            teacher_sketch_tensor,
            self.category_to_label[category],
        )


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode="photo"):
        super().__init__()
        self.max_size = args.max_size
        self.transform = normal_transform()
        self.unseen_classes = UNSEEN_CLASSES[args.dataset]

        unseen_paths = []
        for category in self.unseen_classes:
            paths = glob.glob(
                os.path.join(args.root, mode, category, "*")
            )
            unseen_paths.extend(sorted(paths))

        self.paths = unseen_paths

    def __getitem__(self, index):
        filepath = self.paths[index]
        category = filepath.split(os.path.sep)[-2]

        image = load_image(filepath, self.max_size)
        image_tensor = self.transform(image)

        return image_tensor, self.unseen_classes.index(category)
    
    def __len__(self):
        return len(self.paths)


def load_image(path, size):
    with Image.open(path) as image:
        return ImageOps.pad(image.convert("RGB"), size=(size, size))
