import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def photo_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random",
        ),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def sketch_train_transform():
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=255,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # White fill removes strokes instead of adding artificial black boxes.
        transforms.RandomErasing(
            p=0.25,
            scale=(0.01, 0.08),
            ratio=(0.3, 3.3),
            value=1.0,
        ),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.photo_transform = photo_train_transform()
        self.sketch_transform = sketch_train_transform()
        self.teacher_transform = evaluation_transform()
        
        unseen_classes = UNSEEN_CLASSES[self.args.dataset]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        
        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketch_paths = glob.glob(os.path.join(self.args.root, 'sketch', category, '*'))
            photo_paths = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
            
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        student_sketch = self.sketch_transform(sk_data)
        student_photo = self.photo_transform(img_data)
        student_negative = self.photo_transform(neg_data)

        # Stable, resize-only views are used to construct DFN5B KD targets.
        teacher_sketch = self.teacher_transform(sk_data)
        teacher_photo = self.teacher_transform(img_data)

        return (
            student_photo,
            student_sketch,
            teacher_photo,
            teacher_sketch,
            student_negative,
            self.all_categories.index(category),
        )


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = evaluation_transform()
        self.unseen_classes = UNSEEN_CLASSES[self.args.dataset]
            
        unseen_paths = []
        for category in self.unseen_classes:
            if self.mode == 'photo':
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
            else:
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))

        self.paths = list(unseen_paths)

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        return image_tensor, self.unseen_classes.index(category)
    
    def __len__(self):
        return len(self.paths)
