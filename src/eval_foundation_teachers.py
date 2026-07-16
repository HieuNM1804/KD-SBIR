import argparse
import gc
import json
import os
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
)
from tqdm import tqdm

from src.data_config import UNSEEN_CLASSES


DATASETS = ("sketchy_1", "sketchy_2", "tuberlin", "quickdraw")


@dataclass(frozen=True)
class TeacherSpec:
    alias: str
    model: str
    pretrained: str


TEACHERS = (
    TeacherSpec("clip_l14_openai", "ViT-L-14", "openai"),
    TeacherSpec("clip_l14_336_openai", "ViT-L-14-336", "openai"),
    TeacherSpec("laion_h14", "ViT-H-14", "laion2b_s32b_b79k"),
    TeacherSpec("laion_g14", "ViT-g-14", "laion2b_s34b_b88k"),
    TeacherSpec("laion_bigg14", "ViT-bigG-14", "laion2b_s39b_b160k"),
    TeacherSpec("metaclip_l14", "ViT-L-14", "metaclip_fullcc"),
    TeacherSpec("metaclip_h14", "ViT-H-14", "metaclip_fullcc"),
    TeacherSpec("metaclip_bigg14", "ViT-bigG-14", "metaclip_fullcc"),
    TeacherSpec("metaclip2_h14", "ViT-H-14-worldwide", "metaclip2_worldwide"),
    TeacherSpec("metaclip2_h14_378", "ViT-H-14-worldwide-378", "metaclip2_worldwide"),
    TeacherSpec("dfn2b_l14", "ViT-L-14", "dfn2b"),
    TeacherSpec("dfn2b_l14_s39b", "ViT-L-14", "dfn2b_s39b"),
    TeacherSpec("dfn5b_h14", "ViT-H-14", "dfn5b"),
    TeacherSpec("dfn5b_h14_378", "ViT-H-14-378", "dfn5b"),
    TeacherSpec("eva01_g14", "EVA01-g-14", "laion400m_s11b_b41k"),
    TeacherSpec("eva02_l14", "EVA02-L-14", "merged2b_s4b_b131k"),
    TeacherSpec("eva02_l14_336", "EVA02-L-14-336", "merged2b_s6b_b61k"),
    TeacherSpec("eva02_e14", "EVA02-E-14", "laion2b_s4b_b115k"),
    TeacherSpec("eva02_e14_plus", "EVA02-E-14-plus", "laion2b_s9b_b144k"),
    TeacherSpec("siglip_so400m_384", "ViT-SO400M-14-SigLIP-384", "webli"),
    TeacherSpec("siglip2_so400m_384", "ViT-SO400M-16-SigLIP2-384", "webli"),
    TeacherSpec("siglip2_gopt_384", "ViT-gopt-16-SigLIP2-384", "webli"),
    TeacherSpec("clipa_h14_336", "ViT-H-14-CLIPA-336", "datacomp1b"),
    TeacherSpec("clipa_bigg14_336", "ViT-bigG-14-CLIPA-336", "datacomp1b"),
)

RECOMMENDED_ALIASES = (
    "dfn5b_h14",
    "dfn5b_h14_378",
    "dfn2b_l14_s39b",
    "metaclip_h14",
    "metaclip2_h14_378",
    "siglip_so400m_384",
    "siglip2_so400m_384",
    "eva02_l14_336",
    "eva02_e14",
    "laion_g14",
)


class ValidImageDataset(Dataset):
    def __init__(self, root, dataset, mode, preprocess):
        self.root = root
        self.dataset = dataset
        self.mode = mode
        self.preprocess = preprocess
        self.unseen_classes = UNSEEN_CLASSES[dataset]

        paths = []
        for category in self.unseen_classes:
            class_dir = os.path.join(root, mode, category)
            if not os.path.isdir(class_dir):
                continue
            paths.extend(
                os.path.join(class_dir, filename)
                for filename in sorted(os.listdir(class_dir))
                if not filename.startswith(".")
            )
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        category = path.split(os.path.sep)[-2]
        image = Image.open(path).convert("RGB")
        image = ImageOps.pad(image, size=image.size if image.width == image.height else (max(image.size), max(image.size)))
        return self.preprocess(image), self.unseen_classes.index(category)


def metric_setting(dataset):
    if dataset == "sketchy_2":
        return 200, 200
    if dataset == "quickdraw":
        return None, 200
    return None, 100


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


def teacher_registry():
    return {spec.alias: spec for spec in TEACHERS}


def config_image_size(model_name):
    cfg = open_clip.get_model_config(model_name) or {}
    vision_cfg = cfg.get("vision_cfg", {})
    image_size = vision_cfg.get("image_size", "unknown")
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 1:
            return int(image_size[0])
        return "x".join(str(int(size)) for size in image_size)
    if isinstance(image_size, int):
        return image_size
    return image_size


def parse_teacher_token(token):
    registry = teacher_registry()
    if token in registry:
        return registry[token]
    if "/" in token:
        model, pretrained = token.split("/", 1)
        alias = f"{model}_{pretrained}".replace("/", "_").replace("-", "_")
        return TeacherSpec(alias, model, pretrained)
    raise ValueError(
        f"Unknown teacher '{token}'. Use --list_teachers to see aliases, "
        "or pass a custom open_clip pair as model/pretrained."
    )


def resolve_teachers(args):
    tokens = args.teacher
    if "all" in tokens:
        return list(TEACHERS)
    if "recommended" in tokens:
        registry = teacher_registry()
        return [registry[alias] for alias in RECOMMENDED_ALIASES]

    seen = set()
    specs = []
    for token in tokens:
        spec = parse_teacher_token(token)
        key = (spec.model, spec.pretrained)
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    return specs


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    visual = sum(p.numel() for p in model.visual.parameters())
    return total, visual


def model_dtype(model):
    return next(model.parameters()).dtype


def set_precision(model, args):
    if args.device == "cpu" or args.precision == "fp32":
        return model.float()
    if args.precision == "bf16":
        return model.bfloat16()
    return model.half()


def build_loader(root, dataset, mode, preprocess, args):
    valid_dataset = ValidImageDataset(root, dataset, mode, preprocess)
    return DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def extract_features(model, loader, args, desc):
    features = []
    labels = []
    dtype = model_dtype(model)
    for image, label in tqdm(loader, desc=desc, leave=False):
        image = image.to(args.device, dtype=dtype, non_blocking=True)
        feature = model.encode_image(image)
        feature = F.normalize(feature.float(), dim=-1)
        features.append(feature.cpu())
        labels.append(label.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def evaluate_retrieval(query_features, query_labels, gallery_features, gallery_labels, dataset, args):
    map_k, p_k = metric_setting(dataset)
    gallery_features = gallery_features.to(args.device)
    gallery_labels = gallery_labels.to(args.device)

    ap_values = []
    precision_values = []
    for start in tqdm(range(0, len(query_features), args.metric_chunk_size), desc="metrics", leave=False):
        end = min(start + args.metric_chunk_size, len(query_features))
        query_chunk = query_features[start:end].to(args.device)
        query_label_chunk = query_labels[start:end].to(args.device)
        similarities = query_chunk @ gallery_features.t()

        for scores, query_label in zip(similarities, query_label_chunk):
            target = gallery_labels == query_label
            if map_k is None:
                ap = retrieval_average_precision(scores.cpu(), target.cpu())
            else:
                ap = retrieval_average_precision(scores.cpu(), target.cpu(), top_k=min(map_k, scores.numel()))
            precision = retrieval_precision(scores.cpu(), target.cpu(), top_k=min(p_k, scores.numel()))
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


def evaluate_dataset(model, preprocess, spec, root, dataset, params, args):
    sketch_loader = build_loader(root, dataset, "sketch", preprocess, args)
    photo_loader = build_loader(root, dataset, "photo", preprocess, args)

    print(f"\n[{spec.alias}] dataset={dataset}, root={root}")
    print(f"model={spec.model}, pretrained={spec.pretrained}")
    print(
        f"params={params['total_params_b']:.3f}B "
        f"(visual={params['visual_params_b']:.3f}B), "
        f"input_size={params['input_size']}, "
        f"unseen_classes={len(UNSEEN_CLASSES[dataset])}"
    )

    sketch_features, sketch_labels = extract_features(
        model, sketch_loader, args, f"{spec.alias} {dataset} sketch"
    )
    photo_features, photo_labels = extract_features(
        model, photo_loader, args, f"{spec.alias} {dataset} photo"
    )

    metrics = evaluate_retrieval(
        sketch_features, sketch_labels, photo_features, photo_labels, dataset, args
    )
    metrics.update(
        {
            "teacher": spec.alias,
            "model": spec.model,
            "pretrained": spec.pretrained,
            "dataset": dataset,
            "root": root,
            "num_sketches": int(len(sketch_labels)),
            "num_photos": int(len(photo_labels)),
            "num_unseen_classes": int(len(UNSEEN_CLASSES[dataset])),
            **params,
        }
    )
    print(
        f"{spec.alias}/{dataset}: {metrics['mAP_name']}={metrics['mAP']:.4f}, "
        f"{metrics['precision_name']}={metrics['precision']:.4f} "
        f"(sketches={metrics['num_sketches']}, photos={metrics['num_photos']})"
    )
    return metrics


def print_summary(results, failures):
    if results:
        print("\n=== Foundation CLIP teacher zero-shot retrieval summary ===")
        header = (
            f"{'teacher':<24} {'params':>8} {'input':>6} {'dataset':<10} "
            f"{'mAP':>12} {'precision':>12} {'sketches':>9} {'photos':>9}"
        )
        print(header)
        print("-" * len(header))
        for result in results:
            map_text = f"{result['mAP_name']}={result['mAP']:.4f}"
            precision_text = f"{result['precision_name']}={result['precision']:.4f}"
            print(
                f"{result['teacher']:<24} {result['total_params_b']:>7.3f}B "
                f"{str(result['input_size']):>6} {result['dataset']:<10} "
                f"{map_text:>12} {precision_text:>12} "
                f"{result['num_sketches']:>9} {result['num_photos']:>9}"
            )

    if failures:
        print("\n=== Skipped / failed teachers ===")
        for failure in failures:
            print(f"{failure['teacher']}: {failure['reason']}")


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_teacher(spec, runs, args):
    input_size = config_image_size(spec.model)
    print(f"\nLoading teacher: {spec.alias} ({spec.model}, {spec.pretrained})")
    print(f"Configured input_size={input_size}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.model,
        pretrained=spec.pretrained,
        device=args.device,
    )
    model = set_precision(model.eval(), args)
    total_params, visual_params = count_params(model)
    params = {
        "total_params": int(total_params),
        "visual_params": int(visual_params),
        "total_params_b": total_params / 1e9,
        "visual_params_b": visual_params / 1e9,
        "input_size": input_size,
    }

    if params["total_params_b"] >= args.max_params_b:
        unload_model(model)
        raise RuntimeError(
            f"skip because params={params['total_params_b']:.3f}B >= "
            f"--max_params_b {args.max_params_b}"
        )

    print(
        f"Loaded {spec.alias}: params={params['total_params_b']:.3f}B, "
        f"visual={params['visual_params_b']:.3f}B, input_size={input_size}, "
        f"precision={args.precision}"
    )
    results = [
        evaluate_dataset(model, preprocess, spec, root, dataset, params, args)
        for dataset, root in runs
    ]
    unload_model(model)
    return results


def list_teachers():
    print("Available built-in teacher aliases:")
    for spec in TEACHERS:
        mark = "*" if spec.alias in RECOMMENDED_ALIASES else " "
        print(
            f"{mark} {spec.alias:<24} {spec.model:<32} "
            f"{spec.pretrained:<24} input={config_image_size(spec.model)}"
        )
    print("\n* = included by --teacher recommended")
    print("Custom format: --teacher ModelName/pretrained_name")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen foundation CLIP/open_clip teachers for zero-shot SBIR. "
            "Models with total params >= --max_params_b are skipped after loading."
        )
    )
    parser.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    parser.add_argument("--root", default="", help="Root for one dataset, or Sketchy root when --dataset all.")
    parser.add_argument("--sketchy_root", default="", help="Root containing Sketchy sketch/ and photo/.")
    parser.add_argument("--tuberlin_root", default="", help="Root containing TU-Berlin sketch/ and photo/.")
    parser.add_argument("--quickdraw_root", default="", help="Root containing QuickDraw sketch/ and photo/.")
    parser.add_argument(
        "--teacher",
        nargs="+",
        default=["recommended"],
        help=(
            "Teacher aliases, 'recommended', 'all', or custom open_clip pairs as model/pretrained. "
            "Use --list_teachers to see aliases."
        ),
    )
    parser.add_argument("--max_params_b", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--metric_chunk_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--output", default="", help="Optional JSON path to save metrics and failures.")
    parser.add_argument("--list_teachers", action="store_true")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one teacher fails to download/load/evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_teachers:
        list_teachers()
        return

    runs = resolve_runs(args)
    specs = resolve_teachers(args)
    results = []
    failures = []

    print(f"Running {len(specs)} teacher(s) on {args.device}, max_params_b={args.max_params_b}")
    for spec in specs:
        try:
            results.extend(evaluate_teacher(spec, runs, args))
        except Exception as exc:
            reason = str(exc)
            failures.append({"teacher": spec.alias, "model": spec.model, "pretrained": spec.pretrained, "reason": reason})
            print(f"[Skip/Fail] {spec.alias}: {reason}")
            if args.stop_on_error:
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_summary(results, failures)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": results, "failures": failures}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
