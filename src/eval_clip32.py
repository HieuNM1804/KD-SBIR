import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
)
from tqdm import tqdm

from src.data_config import UNSEEN_CLASSES
from src.dataset import ValidDataset
from src.utils import load_clip_to_cpu


DATASETS = ("sketchy_1", "sketchy_2", "tuberlin", "quickdraw")


def metric_setting(dataset):
    if dataset == "sketchy_2":
        return 200, 200
    if dataset == "quickdraw":
        return None, 200
    return None, 100


def build_valid_loader(root, dataset, mode, args):
    dataset_args = SimpleNamespace(
        root=root,
        dataset=dataset,
        max_size=args.max_size,
    )
    valid_dataset = ValidDataset(dataset_args, mode=mode)
    return DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def extract_features(model, loader, device, desc):
    features = []
    labels = []
    for image, label in tqdm(loader, desc=desc, leave=False):
        image = image.to(device, non_blocking=True)
        feature = model.encode_image(image)
        feature = F.normalize(feature.float(), dim=-1)
        features.append(feature.cpu())
        labels.append(label.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def evaluate_retrieval(query_features, query_labels, gallery_features, gallery_labels, dataset, device, chunk_size):
    map_k, p_k = metric_setting(dataset)
    gallery_features = gallery_features.to(device)
    gallery_labels = gallery_labels.to(device)

    ap_values = []
    precision_values = []

    for start in tqdm(range(0, len(query_features), chunk_size), desc="metrics", leave=False):
        end = min(start + chunk_size, len(query_features))
        query_chunk = query_features[start:end].to(device)
        query_label_chunk = query_labels[start:end].to(device)
        similarities = query_chunk @ gallery_features.t()

        for row, query_label in zip(similarities, query_label_chunk):
            target = gallery_labels == query_label
            if map_k is None:
                ap = retrieval_average_precision(row.cpu(), target.cpu())
            else:
                ap = retrieval_average_precision(
                    row.cpu(),
                    target.cpu(),
                    top_k=min(map_k, row.numel()),
                )
            precision = retrieval_precision(row.cpu(), target.cpu(), top_k=min(p_k, row.numel()))
            ap_values.append(ap)
            precision_values.append(precision)

    return {
        "mAP": float(torch.stack(ap_values).mean().item()),
        "precision": float(torch.stack(precision_values).mean().item()),
        "mAP_name": f"mAP@{map_k}" if map_k is not None else "mAP@all",
        "precision_name": f"P@{p_k}",
        "map_k": map_k,
        "precision_k": p_k,
    }


def resolve_runs(args):
    if args.dataset != "all":
        if not args.root:
            raise ValueError("--root is required when --dataset is not all.")
        return [(args.dataset, args.root)]

    roots = {
        "sketchy_1": args.sketchy_root or args.root,
        "sketchy_2": args.sketchy_root or args.root,
        "tuberlin": args.tuberlin_root,
        "quickdraw": args.quickdraw_root,
    }
    missing = [dataset for dataset, root in roots.items() if not root]
    if missing:
        raise ValueError(
            "--dataset all requires roots for all datasets. Missing: "
            + ", ".join(missing)
            + ". Use --sketchy_root, --tuberlin_root, --quickdraw_root."
        )
    return [(dataset, roots[dataset]) for dataset in DATASETS]


def evaluate_dataset(model, dataset, root, args):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    sketch_loader = build_valid_loader(root, dataset, "sketch", args)
    photo_loader = build_valid_loader(root, dataset, "photo", args)

    print(f"\n[CLIP ViT-B/32] dataset={dataset}, root={root}")
    print(f"unseen_classes={len(UNSEEN_CLASSES[dataset])}")

    sketch_features, sketch_labels = extract_features(
        model, sketch_loader, args.device, f"{dataset} sketch"
    )
    photo_features, photo_labels = extract_features(
        model, photo_loader, args.device, f"{dataset} photo"
    )

    metrics = evaluate_retrieval(
        sketch_features,
        sketch_labels,
        photo_features,
        photo_labels,
        dataset,
        args.device,
        args.metric_chunk_size,
    )
    metrics.update(
        {
            "dataset": dataset,
            "root": root,
            "num_sketches": int(len(sketch_labels)),
            "num_photos": int(len(photo_labels)),
            "num_unseen_classes": int(len(UNSEEN_CLASSES[dataset])),
            "backbone": "ViT-B/32",
        }
    )
    print(
        f"{dataset}: {metrics['mAP_name']}={metrics['mAP']:.4f}, "
        f"{metrics['precision_name']}={metrics['precision']:.4f} "
        f"(sketches={metrics['num_sketches']}, photos={metrics['num_photos']})"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot SBIR retrieval baseline with frozen CLIP ViT-B/32 image encoder."
    )
    parser.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    parser.add_argument("--root", default="", help="Root for one dataset, or Sketchy root when --dataset all.")
    parser.add_argument("--sketchy_root", default="", help="Root containing Sketchy sketch/ and photo/.")
    parser.add_argument("--tuberlin_root", default="", help="Root containing TU-Berlin sketch/ and photo/.")
    parser.add_argument("--quickdraw_root", default="", help="Root containing QuickDraw sketch/ and photo/.")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--metric_chunk_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="", help="Optional JSON path to save metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = resolve_runs(args)

    print(f"Loading frozen CLIP ViT-B/32 on {args.device}...")
    model_cfg = SimpleNamespace(backbone="ViT-B/32", n_ctx=0)
    model = load_clip_to_cpu(
        model_cfg,
        design_details={
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        },
    ).to(args.device)
    model.eval()

    results = [evaluate_dataset(model, dataset, root, args) for dataset, root in runs]

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
