"""Frozen zero-shot teacher classification, separately for photo and sketch.

Run with python -m src.evaluate_teacher_classification --help. No training,
teacher prompts, retrieval pairs, or student model are constructed.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.teacher_classification_specs import (
    DATASET_ROOTS, TEACHERS, class_text, prompt_templates,
)
from src.teacher_classification_metrics import ClassificationMetrics

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def list_dataset(root, dataset, split="seen", candidate_scope="split", max_per_class=0, seed=42):
    """Folder labels are evaluation truth, never an input to the predictor."""
    root = Path(root)
    categories = {}
    for modality in ("photo", "sketch"):
        folder = root / modality
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        categories[modality] = {p.name for p in folder.iterdir()
                                if p.is_dir() and not p.name.startswith(".")}
    if categories["photo"] != categories["sketch"]:
        raise ValueError(f"Photo/sketch class directories differ in {root}.")
    all_classes = sorted(categories["photo"])
    if not all_classes:
        raise ValueError(f"No class directories: {root}")
    unseen = set(UNSEEN_CLASSES[dataset])
    if split == "unseen" and unseen - set(all_classes):
        raise ValueError(f"Missing unseen classes: {sorted(unseen - set(all_classes))}")
    selected = [c for c in all_classes if split == "all" or ((c in unseen) == (split == "unseen"))]
    if not selected:
        raise ValueError(f"No {split} classes in {root}.")
    candidates = all_classes if candidate_scope == "all" else selected
    label_ids = {c: i for i, c in enumerate(candidates)}
    records = {}
    for modality in ("photo", "sketch"):
        current = []
        for name in selected:
            folder = root / modality / name
            with os.scandir(folder) as entries:
                paths = sorted(f"{modality}/{name}/{entry.name}" for entry in entries
                               if not entry.name.startswith(".")
                               and Path(entry.name).suffix.lower() in EXTENSIONS
                               and entry.is_file())
            if not paths:
                raise ValueError(f"Empty image class: {folder}")
            if max_per_class:
                # Reproducible subset, identical for every teacher. Full evaluation
                # is the default; the optional subset is only a screening run.
                paths = sorted(sorted(paths, key=lambda p: hashlib.sha256(
                    f"{seed}:{p}".encode()).digest())[:max_per_class])
            current.extend((p, label_ids[name]) for p in paths)
        records[modality] = current
    fingerprint = hashlib.sha256(json.dumps(
        dict(candidates=candidates, records=records), sort_keys=True).encode()).hexdigest()
    return dict(root=str(root), dataset=dataset, split=split, candidate_scope=candidate_scope,
                candidate_classes=candidates, evaluated_classes=selected, records=records,
                manifest_sha256=fingerprint, max_per_class=max_per_class, seed=seed)


class ImageList(Dataset):
    def __init__(self, root, records, transform):
        self.root, self.records, self.transform = Path(root), records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative, _label = self.records[index]
        path = self.root / relative
        try:
            with Image.open(path) as image:
                value = self.transform(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"Cannot read image: {path}") from error
        # Labels are intentionally not returned to model inference.
        return value, index


def main_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])


@torch.inference_mode()
def text_bank(model, tokenizer, classnames, modality, mode, device):
    banks = []
    for template in prompt_templates(modality, mode):
        prompts = [template.format(class_text(c)) for c in classnames]
        encoded = []
        for offset in range(0, len(prompts), 128):
            tokens = tokenizer(prompts[offset:offset + 128]).to(device)
            features = model.encode_text(tokens).float()
            if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
                raise ValueError("Invalid text features.")
            encoded.append(F.normalize(features, dim=-1))
        banks.append(torch.cat(encoded))
    return F.normalize(torch.stack(banks).mean(0), dim=-1)


@torch.inference_mode()
def evaluate_modality(model, tokenizer, transform, manifest, modality, args, destination, device):
    names = manifest["candidate_classes"]
    records = manifest["records"][modality]
    texts = text_bank(model, tokenizer, names, modality, args.template_mode, device)
    options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=device.type == "cuda")
    if args.workers:
        options.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
    loader = DataLoader(ImageList(manifest["root"], records, transform), **options)
    # CLIP keeps class/position embeddings in FP32 even with FP16 convolutions.
    # Input must match conv1, not simply the first visual parameter.
    dtype = model.visual.conv1.weight.dtype
    scale = 1.0 / args.temperature if args.temperature else float(model.logit_scale.float().exp())
    if not 0 < scale < float("inf"):
        raise ValueError(f"Invalid logit scale: {scale}")
    metrics = ClassificationMetrics(names)
    prediction_path = destination / f"predictions_{modality}.csv"
    start = time.perf_counter()
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_class", "predicted_class", "confidence",
                         "correct", "top5_classes"])
        for images, indices in tqdm(loader, desc=f"{manifest['dataset']} {modality}"):
            images = images.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
            features = model.encode_image(images).float()
            if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
                raise ValueError(f"Invalid image features at batch starting {int(indices[0])}")
            logits = scale * (F.normalize(features, dim=-1) @ texts.t())
            # Ground truth enters only after model outputs have been computed.
            batch_records = [records[i] for i in indices.tolist()]
            labels = torch.tensor([label for _, label in batch_records])
            pred, confidence, top5 = metrics.update(logits, labels)
            for (relative, label), prediction, probability, ranking in zip(
                    batch_records, pred.tolist(), confidence.tolist(), top5.tolist()):
                writer.writerow([relative, names[label], names[prediction], probability,
                                 int(label == prediction), json.dumps([names[i] for i in ranking])])
            stream.flush()
    result, per_class = metrics.finalize()
    result.update(dataset=manifest["dataset"], modality=modality, split=manifest["split"],
                  candidate_scope=manifest["candidate_scope"], template_mode=args.template_mode,
                  templates=prompt_templates(modality, args.template_mode),
                  preprocess=args.preprocess, logit_scale=scale,
                  manifest_sha256=manifest["manifest_sha256"],
                  screening_subset=bool(args.max_per_class), elapsed_seconds=time.perf_counter() - start)
    save_json(destination / f"metrics_{modality}.json", result)
    save_csv(destination / f"per_class_{modality}.csv", per_class)
    save_json(destination / f"confusion_{modality}.json",
              dict(classes=names, rows="true_class", columns="predicted_class",
                   matrix=metrics.confusion.tolist()))
    print(f"{modality}: accuracy={result['accuracy']:.4f}, macro_accuracy={result['macro_accuracy']:.4f}, "
          f"macro_F1={result['macro_f1']:.4f}, top5={result['top5_accuracy']:.4f}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ROOTS,
                        default=["sketchy_1", "tuberlin", "quickdraw"])
    parser.add_argument("--teachers", nargs="+", choices=TEACHERS, default=list(TEACHERS))
    parser.add_argument("--root", help="Override root when evaluating exactly one dataset.")
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="Directory with <teacher_key>/<original checkpoint filename>.")
    parser.add_argument("--split", choices=["seen", "unseen", "all"], default="seen")
    parser.add_argument("--candidate-scope", choices=["split", "all"], default="split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="0: full dataset. Positive: deterministic screening subset per modality/class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-mode", choices=["modality", "photo", "ensemble"], default="modality")
    parser.add_argument("--preprocess", choices=["main", "official"], default="main")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Optional positive softmax temperature; default uses each teacher's native scale.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/teacher_classification"))
    args = parser.parse_args(argv)
    if args.root and len(args.datasets) != 1:
        parser.error("--root requires exactly one --datasets value.")
    if (args.batch_size < 1 or args.workers < 0 or args.prefetch_factor < 1
            or args.max_per_class < 0 or (args.temperature is not None and args.temperature <= 0)):
        parser.error("Invalid batch/worker/prefetch/subset/temperature settings.")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Enable Kaggle GPU or pass --device cpu.")
    # Fail early on missing weights, before scanning data or constructing models.
    for key in args.teachers:
        path = args.models_dir / key / TEACHERS[key]["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manifests = []
    for dataset in args.datasets:
        root = args.root or DATASET_ROOTS[dataset]
        print(f"Listing {dataset} {args.split}: {root}", flush=True)
        manifest = list_dataset(root, dataset, args.split, args.candidate_scope, args.max_per_class, args.seed)
        manifests.append(manifest)
        save_json(output / f"manifest_{dataset}.json", manifest)
        print(f"  candidates={len(manifest['candidate_classes'])}, "
              f"photo={len(manifest['records']['photo']):,}, sketch={len(manifest['records']['sketch']):,}", flush=True)
    import open_clip
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    save_json(output / "run_config.json", dict(
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        teacher_specs={key: TEACHERS[key] for key in args.teachers}, source_commit=commit,
        python=platform.python_version(), torch=torch.__version__, open_clip=open_clip.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        inference_only=True, teacher_prompts=False,
        confidence_note="Native model scales differ. Compare measured accuracy at retained coverage, not raw confidence alone.",
    ))
    summaries = {"photo": [], "sketch": []}
    for key in args.teachers:
        spec = TEACHERS[key]
        checkpoint = args.models_dir / key / spec["filename"]
        print(f"Loading frozen teacher {key}: {spec['model']} / {spec['pretrained']}", flush=True)
        precision = args.precision if device.type == "cuda" else "fp32"
        # Loading a local file bypasses OpenCLIP's pretrained-tag metadata.
        # Restore its activation and preprocessing explicitly: OpenAI/MetaCLIP
        # use QuickGELU even when the built-in architecture default uses GELU.
        pretrained_cfg = open_clip.get_pretrained_cfg(spec["model"], spec["pretrained"])
        if not pretrained_cfg:
            raise ValueError(f"OpenCLIP does not recognize teacher identity: {key}")
        model, _, official_transform = open_clip.create_model_and_transforms(
            spec["model"], pretrained=str(checkpoint), precision=precision, device=device,
            # OpenAI's checksum-pinned .pt is a TorchScript archive. PyTorch 2.6+
            # cannot open it with weights_only=True; OpenCLIP extracts its state.
            weights_only=spec.get("checkpoint_format") != "torchscript",
            force_quick_gelu=bool(pretrained_cfg.get("quick_gelu", False)),
            image_mean=pretrained_cfg.get("mean"), image_std=pretrained_cfg.get("std"),
            image_interpolation=pretrained_cfg.get("interpolation"),
            image_resize_mode=pretrained_cfg.get("resize_mode"))
        model.eval().requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(spec["model"])
        transform = main_transform() if args.preprocess == "main" else official_transform
        for manifest in manifests:
            destination = output / key / manifest["dataset"]
            destination.mkdir(parents=True)
            for modality in ("photo", "sketch"):
                result = evaluate_modality(model, tokenizer, transform, manifest, modality, args, destination, device)
                row = dict(teacher=key, dataset=manifest["dataset"], split=args.split,
                           template_mode=args.template_mode, preprocess=args.preprocess,
                           **{field: result[field] for field in (
                               "images", "classes", "accuracy", "macro_accuracy", "macro_f1",
                               "top5_accuracy", "nll", "ece", "mean_confidence", "predicted_classes")})
                summaries[modality].append(row)
                # Preserve finished teachers/modalities if a later run is interrupted.
                save_csv(output / f"metrics_{modality}.csv", summaries[modality])
                save_json(output / f"metrics_{modality}.json", summaries[modality])
        del model, tokenizer, transform, official_transform
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # A compact upload for analysis; large per-image predictions and manifests
    # remain available separately and are deliberately excluded from this zip.
    report = output / "analysis_report.zip"
    with zipfile.ZipFile(report, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [output / "run_config.json", *output.glob("metrics_*.*")]
        for key in args.teachers:
            for dataset in args.datasets:
                directory = output / key / dataset
                for pattern in ("metrics_*.json", "per_class_*.csv", "confusion_*.json"):
                    paths.extend(directory.glob(pattern))
        for path in paths:
            archive.write(path, path.relative_to(output))
    print("Results:", output, flush=True)
    print("Share for analysis:", report, flush=True)


if __name__ == "__main__":
    main()
