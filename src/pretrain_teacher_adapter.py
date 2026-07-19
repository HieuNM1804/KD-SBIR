"""Pretrain DFN5B modality adapters to convergence before student distillation."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
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
from src.dataset import aumented_transform, normal_transform
from src.teacher_adapters import ModalityAdapters


DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic


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


def collect_seen_splits(args, classnames):
    root = Path(args.root)
    rng = random.Random(args.seed)
    splits = {
        "train": {"sketch": [], "photo": []},
        "validation": {"sketch": [], "photo": []},
    }

    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if len(paths) < 2:
                raise RuntimeError(
                    f"Need at least two {modality} images for seen class '{classname}'."
                )
            if args.max_images_per_class > 0 and len(paths) > args.max_images_per_class:
                paths = sorted(rng.sample(paths, args.max_images_per_class))

            shuffled = paths.copy()
            rng.shuffle(shuffled)
            validation_count = max(1, int(round(len(shuffled) * args.validation_fraction)))
            validation_count = min(validation_count, len(shuffled) - 1)
            validation_paths = sorted(shuffled[:validation_count])
            train_paths = sorted(shuffled[validation_count:])

            splits["train"][modality].extend(
                (path, label) for path in train_paths
            )
            splits["validation"][modality].extend(
                (path, label) for path in validation_paths
            )

    return splits


class PathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert("RGB"), size=(224, 224))
            image = self.transform(image)
        return image, label


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


@torch.inference_mode()
def encode_images(backbone, samples, transform, args, device, description, seed_offset):
    dataset = PathDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + seed_offset),
    )

    features = []
    labels = []
    for images, batch_labels in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        if args.fp16_backbone and device.type == "cuda":
            images = images.half()
        batch_features = F.normalize(backbone.encode_image(images).float(), dim=-1)
        features.append(batch_features.cpu())
        labels.append(batch_labels.cpu())

    return torch.cat(features), torch.cat(labels).long()


@torch.no_grad()
def encode_text(backbone, tokenizer, classnames, device):
    sketch_prompts = [f"a sketch of a {name.replace('_', ' ')}." for name in classnames]
    photo_prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
    tokens = tokenizer(sketch_prompts + photo_prompts).to(device)
    features = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
    class_count = len(classnames)
    return features[:class_count].cpu(), features[class_count:].cpu()


def supervised_cross_modal_loss(sketch_features, photo_features, labels, temperature):
    logits = sketch_features @ photo_features.t() / temperature
    positive_mask = labels[:, None].eq(labels[None, :]).float()
    sketch_targets = positive_mask / positive_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    photo_targets = positive_mask.t() / positive_mask.t().sum(dim=-1, keepdim=True).clamp_min(1)
    sketch_loss = -(sketch_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    photo_loss = -(photo_targets * F.log_softmax(logits.t(), dim=-1)).sum(dim=-1).mean()
    return 0.5 * (sketch_loss + photo_loss)


def batch_hard_triplet_loss(sketch_features, photo_features, labels, margin):
    distance = 1.0 - sketch_features @ photo_features.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask

    def one_direction(current_distance):
        valid = negative_mask.any(dim=-1)
        hardest_positive = current_distance.masked_fill(
            ~positive_mask, -torch.inf
        ).max(dim=-1).values
        hardest_negative = current_distance.masked_fill(
            ~negative_mask, torch.inf
        ).min(dim=-1).values
        losses = F.relu(hardest_positive - hardest_negative + margin)
        return losses[valid].mean() if valid.any() else current_distance.new_zeros(())

    return 0.5 * (one_direction(distance) + one_direction(distance.t()))


def semantic_loss(
    sketch_features,
    photo_features,
    labels,
    sketch_text,
    photo_text,
    temperature,
):
    return 0.5 * (
        F.cross_entropy(sketch_features @ sketch_text.t() / temperature, labels)
        + F.cross_entropy(photo_features @ photo_text.t() / temperature, labels)
    )


@torch.inference_mode()
def adapt_features(adapter, features, device, batch_size=4096):
    outputs = []
    adapter.eval()
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device, non_blocking=True)
        outputs.append(adapter(batch.float()).cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def retrieval_map(adapters, validation_features, device, chunk_size):
    sketch_features = adapt_features(
        adapters.sketch, validation_features["sketch"][0], device
    ).to(device)
    photo_features = adapt_features(
        adapters.photo, validation_features["photo"][0], device
    ).to(device)
    sketch_labels = validation_features["sketch"][1].to(device)
    photo_labels = validation_features["photo"][1].to(device)
    ranks = torch.arange(1, len(photo_features) + 1, device=device).float()

    average_precisions = []
    for start in range(0, len(sketch_features), chunk_size):
        query = sketch_features[start : start + chunk_size]
        query_labels = sketch_labels[start : start + chunk_size]
        similarity = query @ photo_features.t()
        order = similarity.argsort(dim=-1, descending=True)
        relevant = photo_labels[order].eq(query_labels[:, None])
        relevant_float = relevant.float()
        precision = relevant_float.cumsum(dim=-1) / ranks
        relevant_count = relevant_float.sum(dim=-1)
        average_precision = (precision * relevant_float).sum(dim=-1) / relevant_count.clamp_min(1)
        average_precisions.append(average_precision.cpu())

    return float(torch.cat(average_precisions).mean().item())


def make_scheduler(optimizer, total_steps, warmup_fraction):
    warmup_steps = int(total_steps * warmup_fraction)

    def schedule(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(path, adapters, args, epoch, validation_map, classnames):
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
            "validation_map": validation_map,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="sketchy_1", choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=1.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--max_images_per_class", type=int, default=0)
    parser.add_argument("--retrieval_chunk_size", type=int, default=256)
    parser.add_argument("--fp16_backbone", action="store_true", default=True)
    parser.add_argument("--no_fp16_backbone", action="store_false", dest="fp16_backbone")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", default="teacher_adapter_runs/dfn5b_sketchy1_converged"
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation_fraction must be between 0 and 1.")
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        parser.error("Require 1 <= min_epochs <= max_epochs.")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classnames = get_seen_classes(args.root, args.dataset)
    splits = collect_seen_splits(args, classnames)
    for split_name, modalities in splits.items():
        print(
            f"[{split_name}] sketches={len(modalities['sketch']):,}, "
            f"photos={len(modalities['photo']):,}, classes={len(classnames)}"
        )

    print(f"Loading frozen backbone {DFN5B_MODEL} ({DFN5B_PRETRAINED})...")
    backbone, _, _ = open_clip.create_model_and_transforms(
        DFN5B_MODEL, pretrained=DFN5B_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    backbone = backbone.eval().requires_grad_(False).to(device)
    if args.fp16_backbone and device.type == "cuda":
        backbone = backbone.half()

    validation_features = {}
    validation_transform = normal_transform()
    for seed_offset, modality in enumerate(("sketch", "photo"), start=100):
        validation_features[modality] = encode_images(
            backbone,
            splits["validation"][modality],
            validation_transform,
            args,
            device,
            f"encode validation {modality}",
            seed_offset,
        )

    sketch_text, photo_text = encode_text(backbone, tokenizer, classnames, device)
    feature_dim = validation_features["sketch"][0].shape[-1]

    pair_dataset = ImagePairDataset(
        splits["train"]["sketch"],
        splits["train"]["photo"],
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

    # Reset initialization RNG after feature extraction so adapter weights depend only on seed.
    seed_everything(args.seed, deterministic=args.deterministic)
    adapters = ModalityAdapters(feature_dim, args.bottleneck_dim).to(device)
    sketch_text = sketch_text.to(device)
    photo_text = photo_text.to(device)
    optimizer = torch.optim.AdamW(
        adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=args.max_epochs * len(train_loader),
        warmup_fraction=args.warmup_fraction,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    initial_map = retrieval_map(
        adapters, validation_features, device, args.retrieval_chunk_size
    )
    best_map = -math.inf
    best_epoch = 0
    stale_epochs = 0
    save_checkpoint(
        output_dir / "initial.pt", adapters, args, 0, initial_map, classnames
    )
    print(f"[Adapter] epoch=0 validation_mAP={initial_map:.6f} (identity baseline)")

    for epoch in range(1, args.max_epochs + 1):
        adapters.train()
        totals = {"total": 0.0, "contrastive": 0.0, "retrieval": 0.0, "semantic": 0.0}
        progress = tqdm(train_loader, desc=f"adapter epoch {epoch}/{args.max_epochs}")
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
            contrastive = supervised_cross_modal_loss(
                adapted_sketch, adapted_photo, labels, args.temperature
            )
            retrieval = batch_hard_triplet_loss(
                adapted_sketch, adapted_photo, labels, args.triplet_margin
            )
            semantic = semantic_loss(
                adapted_sketch,
                adapted_photo,
                labels,
                sketch_text,
                photo_text,
                args.temperature,
            )
            loss = (
                args.lambda_contrastive * contrastive
                + args.lambda_retrieval * retrieval
                + args.lambda_semantic * semantic
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            totals["total"] += loss.item()
            totals["contrastive"] += contrastive.item()
            totals["retrieval"] += retrieval.item()
            totals["semantic"] += semantic.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        batch_count = len(train_loader)
        train_metrics = {key: value / batch_count for key, value in totals.items()}
        validation_map = retrieval_map(
            adapters, validation_features, device, args.retrieval_chunk_size
        )
        improved = validation_map > best_map + args.min_delta
        if improved:
            best_map = validation_map
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                adapters,
                args,
                epoch,
                validation_map,
                classnames,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            output_dir / "last.pt",
            adapters,
            args,
            epoch,
            validation_map,
            classnames,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_map": validation_map,
            "best_map": best_map,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(
            f"[Adapter] epoch={epoch} train={train_metrics['total']:.6f} "
            f"validation_mAP={validation_map:.6f} best={best_map:.6f}@{best_epoch} "
            f"stale={stale_epochs}/{args.patience}{' *' if improved else ''}"
        )

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(
                f"[Adapter] converged: no validation improvement greater than "
                f"{args.min_delta} for {args.patience} epochs."
            )
            break

    print(
        f"[Adapter] finished; best checkpoint={output_dir / 'best.pt'}, "
        f"mAP={best_map:.6f}, epoch={best_epoch}"
    )


if __name__ == "__main__":
    main()
