"""Pretrain DFN5B modality adapters for fixed epochs before distillation."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.dataset import aumented_transform
from src.losses import batch_hard_teacher_triplet_loss, teacher_semantic_loss
from src.teacher_adapters import ModalityAdapters


DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def list_images(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_seen_classes(root, dataset):
    root = Path(root)
    sketch_root = root / "sketch"
    photo_root = root / "photo"
    if not sketch_root.is_dir() or not photo_root.is_dir():
        raise FileNotFoundError(f"Expected sketch/ and photo/ under {root}.")

    unseen = set(UNSEEN_CLASSES[dataset])
    return sorted(
        path.name
        for path in sketch_root.iterdir()
        if path.is_dir()
        and path.name not in unseen
        and (photo_root / path.name).is_dir()
    )


def collect_seen_samples(args, classnames):
    root = Path(args.root)
    samples = {"sketch": [], "photo": []}

    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if not paths:
                raise RuntimeError(
                    f"No {modality} images found for seen class '{classname}'."
                )
            samples[modality].extend((path, label) for path in paths)

    return samples


class ImagePairDataset(Dataset):
    """Pair every sketch with a random same-class photo and augment both."""

    def __init__(self, sketch_samples, photo_samples, transform):
        self.sketch_samples = sketch_samples
        self.photo_paths = {}
        self.transform = transform
        for path, label in photo_samples:
            self.photo_paths.setdefault(label, []).append(path)

    def __len__(self):
        return len(self.sketch_samples)

    def load_image(self, path):
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert("RGB"), size=(224, 224))
            return self.transform(image)

    def __getitem__(self, index):
        sketch_path, label = self.sketch_samples[index]
        candidates = self.photo_paths[label]
        photo_path = candidates[torch.randint(len(candidates), (1,)).item()]
        return (
            self.load_image(sketch_path),
            self.load_image(photo_path),
            label,
        )


@torch.no_grad()
def encode_text(backbone, tokenizer, classnames, device):
    sketch_prompts = [f"a sketch of a {name.replace('_', ' ')}." for name in classnames]
    photo_prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
    tokens = tokenizer(sketch_prompts + photo_prompts).to(device)
    features = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
    class_count = len(classnames)
    return features[:class_count].cpu(), features[class_count:].cpu()


def save_checkpoint(path, adapters, args, epoch, train_metrics, classnames):
    torch.save(
        {
            "epoch": epoch,
            "adapter_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in adapters.state_dict().items()
            },
            "feature_dim": adapters.sketch.norm.normalized_shape[0],
            "bottleneck_dim": args.bottleneck_dim,
            "adapter_mode": "residual",
            "model": DFN5B_MODEL,
            "pretrained": DFN5B_PRETRAINED,
            "dataset": args.dataset,
            "seen_classes": classnames,
            "train_metrics": train_metrics,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="sketchy_1", choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=1.0)
    parser.add_argument("--fp16_backbone", action="store_true", default=True)
    parser.add_argument("--no_fp16_backbone", action="store_false", dest="fp16_backbone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", default="teacher_adapter_runs/dfn5b_sketchy1_full_seen"
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classnames = get_seen_classes(args.root, args.dataset)
    samples = collect_seen_samples(args, classnames)
    print(
        f"[full seen train] sketches={len(samples['sketch']):,}, "
        f"photos={len(samples['photo']):,}, classes={len(classnames)}"
    )

    print(f"Loading frozen backbone {DFN5B_MODEL} ({DFN5B_PRETRAINED})...")
    backbone, _, _ = open_clip.create_model_and_transforms(
        DFN5B_MODEL, pretrained=DFN5B_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    backbone = backbone.eval().requires_grad_(False).to(device)
    if args.fp16_backbone and device.type == "cuda":
        backbone = backbone.half()

    sketch_text, photo_text = encode_text(backbone, tokenizer, classnames, device)
    feature_dim = int(sketch_text.shape[-1])

    pair_dataset = ImagePairDataset(
        samples["sketch"],
        samples["photo"],
        aumented_transform(),
    )
    train_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    if not len(train_loader):
        raise RuntimeError("Adapter training loader has no complete batch.")

    # Make adapter initialization independent of teacher/text initialization details.
    seed_everything(args.seed)
    adapters = ModalityAdapters(feature_dim, args.bottleneck_dim).to(device)
    sketch_text = sketch_text.to(device)
    photo_text = photo_text.to(device)
    optimizer = torch.optim.SGD(
        adapters.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        adapters.train()
        totals = {"total": 0.0, "retrieval": 0.0, "semantic": 0.0}
        progress = tqdm(train_loader, desc=f"adapter epoch {epoch}/{args.epochs}")
        for sketch_images, photo_images, labels in progress:
            sketch_images = sketch_images.to(device, non_blocking=True)
            photo_images = photo_images.to(device, non_blocking=True)
            if args.fp16_backbone and device.type == "cuda":
                sketch_images = sketch_images.half()
                photo_images = photo_images.half()
            with torch.no_grad():
                sketch_features = F.normalize(
                    backbone.encode_image(sketch_images).float(), dim=-1
                )
                photo_features = F.normalize(
                    backbone.encode_image(photo_images).float(), dim=-1
                )
            labels = labels.to(device, non_blocking=True).long()

            adapted_sketch = adapters.sketch(sketch_features)
            adapted_photo = adapters.photo(photo_features)
            retrieval = batch_hard_teacher_triplet_loss(
                adapted_sketch, adapted_photo, labels, args.triplet_margin
            )
            semantic = teacher_semantic_loss(
                adapted_sketch,
                adapted_photo,
                labels,
                sketch_text,
                photo_text,
                args.temperature,
            )
            loss = (
                args.lambda_retrieval * retrieval
                + args.lambda_semantic * semantic
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["total"] += loss.item()
            totals["retrieval"] += retrieval.item()
            totals["semantic"] += semantic.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        batch_count = len(train_loader)
        train_metrics = {key: value / batch_count for key, value in totals.items()}
        save_checkpoint(
            output_dir / "last.pt",
            adapters,
            args,
            epoch,
            train_metrics,
            classnames,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(
            f"[Adapter] epoch={epoch}/{args.epochs} "
            f"train={train_metrics['total']:.6f}; saved last.pt"
        )
        scheduler.step()

    print(
        f"[Adapter] finished all {args.epochs} epochs; "
        f"checkpoint={output_dir / 'last.pt'}"
    )


if __name__ == "__main__":
    main()
