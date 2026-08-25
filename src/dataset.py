import os
import glob
import hashlib
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from src.data_config import GENERALIZED_CLASSES, UNSEEN_CLASSES

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


def normal_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.seed = args.seed
        self.max_size = args.max_size
        self.normal_transform = normal_transform(self.max_size)

        sketch_root = os.path.join(args.root, "sketch")
        excluded = set(UNSEEN_CLASSES[args.dataset]) | {".ipynb_checkpoints"}
        self.all_categories = sorted(set(os.listdir(sketch_root)) - excluded)
        self.category_to_label = {
            category: label for label, category in enumerate(self.all_categories)
        }
        self.all_sketches_path = []
        self.all_photos_path = {}
        self.all_photo_paths = []
        self.photo_path_to_index = {}
        self.teacher_sketch_features = None
        self.teacher_photo_features = None

        for category in self.all_categories:
            sketch_paths = sorted(
                glob.glob(os.path.join(args.root, "sketch", category, "*"))
            )
            photo_paths = sorted(
                glob.glob(os.path.join(args.root, "photo", category, "*"))
            )
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths
            for path in photo_paths:
                self.photo_path_to_index[path] = len(self.all_photo_paths)
                self.all_photo_paths.append(path)

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

        if self.teacher_sketch_features is None:
            teacher_sketch_feature = torch.empty(0)
            teacher_photo_feature = torch.empty(0)
        else:
            teacher_sketch_feature = self.teacher_sketch_features[index]
            teacher_photo_feature = self.teacher_photo_features[
                self.photo_path_to_index[img_path]
            ]

        return (
            img_tensor,
            sk_tensor,
            teacher_photo_feature,
            teacher_sketch_feature,
            self.category_to_label[category],
        )


class TeacherFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, paths, max_size):
        self.paths = paths
        self.max_size = max_size
        self.transform = normal_transform(self.max_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_image(self.paths[index], self.max_size)
        return self.transform(image)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode="photo", protocol="zs"):
        super().__init__()
        if protocol not in {"zs", "gzs"}:
            raise ValueError(f"Unknown evaluation protocol: {protocol}")
        if mode not in {"photo", "sketch"}:
            raise ValueError(f"Unknown validation modality: {mode}")

        self.max_size = args.max_size
        self.transform = normal_transform(self.max_size)
        self.unseen_classes = UNSEEN_CLASSES[args.dataset]

        photo_root = os.path.join(args.root, "photo")
        available_photo_classes = sorted(
            category
            for category in os.listdir(photo_root)
            if category != ".ipynb_checkpoints"
            and os.path.isdir(os.path.join(photo_root, category))
        )
        missing_unseen = sorted(
            set(self.unseen_classes) - set(available_photo_classes)
        )
        if missing_unseen:
            raise FileNotFoundError(
                "Unseen photo categories are missing from the dataset: "
                + ", ".join(missing_unseen)
            )

        # Subset-GZS keeps unseen sketches as queries and adds a fixed subset
        # of seen-category photos to the complete unseen photo gallery.
        if protocol == "gzs":
            if args.dataset not in GENERALIZED_CLASSES:
                supported = ", ".join(sorted(GENERALIZED_CLASSES))
                raise ValueError(
                    f"No fixed GZS seen subset for {args.dataset}. "
                    f"Supported datasets: {supported}."
                )
            generalized_classes = list(GENERALIZED_CLASSES[args.dataset])
            overlap = sorted(
                set(self.unseen_classes) & set(generalized_classes)
            )
            if overlap:
                raise ValueError(
                    "GZS seen subset overlaps unseen classes: "
                    + ", ".join(overlap)
                )
            missing_generalized = sorted(
                set(generalized_classes) - set(available_photo_classes)
            )
            if missing_generalized:
                raise FileNotFoundError(
                    "GZS seen photo categories are missing from the dataset: "
                    + ", ".join(missing_generalized)
                )
            self.label_classes = (
                list(self.unseen_classes) + generalized_classes
            )
            selected_classes = (
                self.unseen_classes if mode == "sketch" else self.label_classes
            )
        else:
            self.label_classes = list(self.unseen_classes)
            selected_classes = self.unseen_classes
        self.category_to_label = {
            category: index for index, category in enumerate(self.label_classes)
        }
        self.protocol = protocol
        self.mode = mode

        evaluation_paths = []
        for category in selected_classes:
            paths = glob.glob(
                os.path.join(args.root, mode, category, "*")
            )
            evaluation_paths.extend(sorted(paths))

        self.paths = evaluation_paths

    def __getitem__(self, index):
        filepath = self.paths[index]
        category = filepath.split(os.path.sep)[-2]

        image = load_image(filepath, self.max_size)
        image_tensor = self.transform(image)

        return image_tensor, self.category_to_label[category]
    
    def __len__(self):
        return len(self.paths)


def load_image(path, _size):
    with Image.open(path) as image:
        return image.convert("RGB")
