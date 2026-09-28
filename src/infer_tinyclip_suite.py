"""Raw, training-free TinyCLIP retrieval benchmark on SBIR datasets."""

import argparse
import csv
import gc
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.dataset import normal_transform
from src.tinyclip_inference import MODEL_SPECS, load_frozen_image_encoder


class ImageDataset(Dataset):
    def __init__(self, paths, labels, image_size=224):
        self.paths = paths
        self.labels = labels
        self.transform = normal_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def collect_paths(root, dataset, scope, max_per_class=0):
    root = Path(root)
    for mode in ("sketch", "photo"):
        if not (root / mode).is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {root / mode}")

    if scope == "unseen":
        classes = list(UNSEEN_CLASSES[dataset])
    else:
        sketch_classes = {
            path.name for path in (root / "sketch").iterdir() if path.is_dir()
        }
        photo_classes = {
            path.name for path in (root / "photo").iterdir() if path.is_dir()
        }
        classes = sorted(sketch_classes & photo_classes)
    if not classes:
        raise RuntimeError("No common sketch/photo classes were found.")

    result = {}
    for mode in ("sketch", "photo"):
        paths = []
        labels = []
        for label, category in enumerate(classes):
            category_dir = root / mode / category
            if not category_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {category_dir}")
            category_paths = sorted(
                path for path in category_dir.iterdir() if path.is_file()
            )
            if max_per_class:
                category_paths = category_paths[:max_per_class]
            if not category_paths:
                raise RuntimeError(f"No {mode} images found for class {category!r}.")
            paths.extend(category_paths)
            labels.extend([label] * len(category_paths))
        result[mode] = (paths, torch.tensor(labels, dtype=torch.long))
    return classes, result


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def encode_dataset(encoder, loader, device, description, warmup_steps):
    features = []
    labels = []
    forward_seconds = 0.0
    encoded_images = 0
    warmed_up = False
    for images, batch_labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device=device, non_blocking=True)
        if not warmed_up:
            for _ in range(warmup_steps):
                encoder(images)
            _sync(device)
            warmed_up = True
        _sync(device)
        start = time.perf_counter()
        output = encoder(images)
        _sync(device)
        forward_seconds += time.perf_counter() - start
        encoded_images += len(images)
        output = F.normalize(output.float(), dim=-1)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{description} produced NaN or infinite features.")
        features.append(output.cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels), forward_seconds, encoded_images


def retrieval_metrics(
    query_features,
    gallery_features,
    query_labels,
    gallery_labels,
    map_k=0,
    precision_k=100,
    query_batch_size=256,
):
    """Class-relevance cosine mAP and P@K with bounded similarity memory."""

    if len(gallery_features) < precision_k:
        raise ValueError(
            f"Gallery has {len(gallery_features)} items, fewer than P@{precision_k}."
        )
    ranks = torch.arange(1, len(gallery_features) + 1, dtype=torch.float32)
    ap_values = []
    precision_values = []
    for start in range(0, len(query_features), query_batch_size):
        stop = min(start + query_batch_size, len(query_features))
        similarity = query_features[start:stop] @ gallery_features.T
        try:
            order = torch.argsort(similarity, dim=1, descending=True, stable=True)
        except TypeError:
            order = torch.argsort(similarity, dim=1, descending=True)
        hits = gallery_labels[order].eq(query_labels[start:stop, None])
        relevant = hits.sum(dim=1).clamp_min(1)
        if map_k:
            cutoff_hits = hits[:, :map_k]
            cutoff_ranks = ranks[:map_k]
            denominator = relevant.clamp_max(map_k)
        else:
            cutoff_hits = hits
            cutoff_ranks = ranks
            denominator = relevant
        precision_at_relevant = cutoff_hits.cumsum(dim=1) / cutoff_ranks
        ap_values.append(
            (precision_at_relevant * cutoff_hits).sum(dim=1) / denominator
        )
        precision_values.append(hits[:, :precision_k].float().mean(dim=1))
    return {
        "mAP": torch.cat(ap_values).mean().item(),
        "precision": torch.cat(precision_values).mean().item(),
        "map_k": map_k,
        "precision_k": precision_k,
    }


def parse_model_keys(value):
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown model(s): {unknown}; choose from {list(MODEL_SPECS)}"
        )
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("Model list contains duplicates.")
    return keys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run raw TinyCLIP 8M/22M/40M/45M/61M SBIR inference."
    )
    parser.add_argument("--root", required=True, help="Dataset root with sketch/ and photo/.")
    parser.add_argument(
        "--dataset", default="sketchy_2", choices=tuple(UNSEEN_CLASSES),
        help="Class split and metric convention.",
    )
    parser.add_argument("--scope", choices=("unseen", "all"), default="unseen")
    parser.add_argument(
        "--models", type=parse_model_keys, default=list(MODEL_SPECS),
        help="Comma-separated subset; default: 8m,22m,40m,45m,61m.",
    )
    parser.add_argument(
        "--models-root", type=Path, default=None,
        help="Directory containing tinyclip{key}_student checkpoint folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--metric-query-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Optional deterministic smoke-test cap; zero uses every image.",
    )
    return parser


def validate_args(parser, args):
    for name in ("batch_size", "metric_query_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0 or args.warmup_steps < 0 or args.max_per_class < 0:
        parser.error("workers, warmup-steps, and max-per-class cannot be negative.")


def write_results(output_dir, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tinyclip_inference_results.json"
    csv_path = output_dir / "tinyclip_inference_results.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = payload["results"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, paths = collect_paths(
        args.root, args.dataset, args.scope, args.max_per_class
    )
    loaders = {
        mode: DataLoader(
            ImageDataset(*paths[mode]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for mode in ("sketch", "photo")
    }
    map_k = 200 if args.dataset == "sketchy_2" else 0
    precision_k = 200 if args.dataset in ("sketchy_2", "quickdraw") else 100
    print(
        f"Dataset={args.dataset} scope={args.scope} classes={len(classes)} "
        f"sketches={len(paths['sketch'][0])} photos={len(paths['photo'][0])}"
    )
    print("Training is disabled: all checkpoints are frozen and evaluated in inference_mode.")
    results = []
    for key in args.models:
        spec = MODEL_SPECS[key]
        print(f"\n[{key}] Loading {spec.name}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        encoder = load_frozen_image_encoder(key, args.models_root, device, args.precision)
        total_parameters = sum(parameter.numel() for parameter in encoder.parameters())
        vision_parameters = encoder.vision_parameter_count()
        sketch_features, sketch_labels, sketch_seconds, sketch_count = encode_dataset(
            encoder, loaders["sketch"], device, f"{key} sketches", args.warmup_steps
        )
        photo_features, photo_labels, photo_seconds, photo_count = encode_dataset(
            encoder, loaders["photo"], device, f"{key} photos", args.warmup_steps
        )
        metrics = retrieval_metrics(
            sketch_features, photo_features, sketch_labels, photo_labels,
            map_k=map_k, precision_k=precision_k,
            query_batch_size=args.metric_query_batch_size,
        )
        total_seconds = sketch_seconds + photo_seconds
        image_count = sketch_count + photo_count
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        )
        row = {
            "key": key,
            "model": spec.name,
            "checkpoint_kind": spec.kind,
            "patch_size": spec.patch_size,
            "vision_width": spec.vision_width,
            "vision_layers": spec.vision_layers,
            "total_parameters": total_parameters,
            "vision_parameters": vision_parameters,
            "queries": len(sketch_features),
            "gallery": len(photo_features),
            "mAP": metrics["mAP"],
            "map_k": metrics["map_k"] or "all",
            "precision": metrics["precision"],
            "precision_k": metrics["precision_k"],
            "forward_ms_per_image": 1000.0 * total_seconds / image_count,
            "forward_images_per_second": image_count / total_seconds,
            "peak_gpu_memory_mib": peak_memory,
        }
        results.append(row)
        map_name = f"mAP@{map_k}" if map_k else "mAP@all"
        print(
            f"[{key}] {map_name}={row['mAP']:.4f} "
            f"P@{precision_k}={row['precision']:.4f} "
            f"latency={row['forward_ms_per_image']:.3f} ms/image",
            flush=True,
        )
        del encoder, sketch_features, photo_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "protocol": {
            "dataset": args.dataset,
            "root": str(Path(args.root).resolve()),
            "scope": args.scope,
            "classes": classes,
            "image_size": 224,
            "normalization": "OpenAI CLIP mean/std",
            "training": False,
            "prompts": False,
            "precision": args.precision,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_per_class": args.max_per_class,
        },
        "results": results,
    }
    json_path, csv_path = write_results(args.output_dir, payload)
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
