"""Stage 1: static labels from the original frozen DFN5B-224, before prompts.

Only the seen vocabulary and image paths are supplied. The folder category of
an individual image is never supplied to classification or used to correct it.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import TeacherFeatureDataset


FORMAT_VERSION = 1
TEACHER_REPOSITORY = "apple/DFN5B-CLIP-ViT-H-14"
TEACHER_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
TEACHER_SHA256 = "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae"
TEMPLATES = {"sketch": "a sketch of a {}.", "photo": "a photo of a {}."}


def relative_paths(dataset, root):
    return {modality: [os.path.relpath(path, root).replace("\\", "/") for path in paths]
            for modality, paths in (("sketch", dataset.all_sketches_path),
                                    ("photo", dataset.all_photo_paths))}


def input_metadata(args, dataset):
    digest = hashlib.sha256()
    for modality, paths in relative_paths(dataset, args.root).items():
        digest.update(modality.encode("utf-8") + b"\0")
        for path in paths:
            digest.update(path.encode("utf-8") + b"\0")
    return {"format_version": FORMAT_VERSION, "dataset": args.dataset,
            "classnames": list(dataset.all_categories), "paths_sha256": digest.hexdigest(),
            "sketch_count": len(dataset.all_sketches_path),
            "photo_count": len(dataset.all_photo_paths),
            "teacher_repository": TEACHER_REPOSITORY, "teacher_revision": TEACHER_REVISION,
            "teacher_sha256": TEACHER_SHA256, "resolution": 224,
            "preprocess": "main_rgb_resize_224_clip_normalization",
            "templates": TEMPLATES, "candidate_scope": "all_seen_classes",
            "assignment": "independent_top1_no_confidence_filter",
            "seed": args.seed, "inference_batch_size": args.pseudo_batch_size}


def default_pseudo_label_path(args, dataset):
    encoded = json.dumps(input_metadata(args, dataset), sort_keys=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:16]
    directory = Path(args.pseudo_label_dir)
    return directory / f"{args.dataset}_{key}.pt"


def prediction_fingerprint(payload):
    """Bind teacher feature caches to the exact path order and hard predictions."""
    digest = hashlib.sha256(json.dumps(payload["metadata"], sort_keys=True).encode("utf-8"))
    for modality in ("sketch", "photo"):
        labels = torch.as_tensor(payload[modality + "_labels"]).cpu().long()
        digest.update(labels.numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def validate_payload(payload, expected):
    if payload.get("metadata") != expected:
        raise RuntimeError("Pseudo-label cache metadata differ from the current dataset/configuration. Use a new --pseudo_label_path or --rebuild_pseudo_labels.")
    class_count = len(expected["classnames"])
    for modality in ("sketch", "photo"):
        count = expected[modality + "_count"]
        labels = payload.get(modality + "_labels")
        confidence = payload.get(modality + "_confidence")
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
                or labels.shape != (count,) or (labels < 0).any() or (labels >= class_count).any()):
            raise ValueError(f"Invalid {modality} pseudo-label vector.")
        if (not isinstance(confidence, torch.Tensor) or confidence.shape != (count,)
                or not torch.isfinite(confidence).all()
                or (confidence < 0).any() or (confidence > 1).any()):
            raise ValueError(f"Invalid {modality} confidence vector.")


@torch.no_grad()
def classify_seen(teacher, dataset, batch_size, workers, show_progress=True):
    """No prompt controller or true labels are used; both modalities independent."""
    teacher.eval().requires_grad_(False)
    parameter = teacher.visual.conv1.weight
    device, dtype = parameter.device, parameter.dtype
    output = {}
    scale = teacher.logit_scale.detach().float().exp()
    if not torch.isfinite(scale).all():
        raise ValueError("Teacher logit scale is not finite.")
    for modality, paths in (("sketch", dataset.all_sketches_path),
                            ("photo", dataset.all_photo_paths)):
        if not paths:
            raise ValueError(f"No seen {modality} images to classify.")
        names = [TEMPLATES[modality].format(name.replace("_", " "))
                 for name in dataset.all_categories]
        text_parts = []
        for start in range(0, len(names), 128):
            tokens = teacher.text_tokenizer(names[start:start + 128]).to(device)
            text_parts.append(teacher.encode_text(tokens).float())
        text = F.normalize(torch.cat(text_parts), dim=-1)
        if not torch.isfinite(text).all() or (text.norm(dim=-1) == 0).any():
            raise ValueError(f"Invalid teacher {modality} text features.")
        options = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=device.type == "cuda", persistent_workers=False)
        if workers:
            options["prefetch_factor"] = 2
        loader = DataLoader(TeacherFeatureDataset(paths, 224), **options)
        labels = torch.empty(len(paths), dtype=torch.long)
        confidence = torch.empty(len(paths), dtype=torch.float32)
        offset = 0
        for images in tqdm(loader, desc=f"Frozen DFN5B pseudo-labels: {modality}",
                           disable=not show_progress):
            features = teacher.encode_image(images.to(device=device, dtype=dtype,
                                                       non_blocking=device.type == "cuda")).float()
            if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
                raise ValueError(f"Invalid teacher {modality} image features.")
            logits = scale * (F.normalize(features, dim=-1) @ text.t())
            if not torch.isfinite(logits).all():
                raise ValueError(f"Invalid teacher {modality} classification logits.")
            probabilities, indices = logits.softmax(dim=-1).max(dim=-1)
            end = offset + len(images)
            labels[offset:end] = indices.cpu()
            confidence[offset:end] = probabilities.cpu()
            offset = end
        if offset != len(paths):
            raise RuntimeError("Pseudo-label classification did not cover every seen image.")
        output[modality + "_labels"] = labels
        output[modality + "_confidence"] = confidence
        print(f"[Pseudo Labels] {modality}: {len(paths):,} images, "
              f"{labels.unique().numel()}/{len(names)} predicted classes", flush=True)
    return output


def write_reports(path, payload, dataset, root, sampling):
    """Export predictions and class balance; no ground-truth accuracy is used."""
    names = dataset.all_categories
    relative = relative_paths(dataset, root)
    summary = {"metadata": payload["metadata"], "prediction_fingerprint": prediction_fingerprint(payload),
               "sampling": sampling, "selection_uses_true_labels": False,
               "checkpoint_selection": "main_best_unseen_precision_uses_evaluation_labels"}
    for modality in ("sketch", "photo"):
        labels = payload[modality + "_labels"].tolist()
        confidence = payload[modality + "_confidence"].tolist()
        counts = torch.bincount(payload[modality + "_labels"], minlength=len(names)).tolist()
        summary[modality] = {"count": len(labels), "predicted_counts": dict(zip(names, counts)),
                             "mean_confidence": sum(confidence) / len(confidence)}
        target = path.with_name(path.stem + f"_{modality}.csv")
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("path", "pseudo_label", "pseudo_class", "confidence"))
            writer.writerows((relative_path, label, names[label], score)
                             for relative_path, label, score in zip(relative[modality], labels, confidence))
        os.replace(temporary, target)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def prepare_pseudo_labels(args, dataset, teacher_factory):
    """Restore/generate static labels, apply them, and return a reusable teacher.

    On a cache hit no DFN5B is loaded for this stage. On a miss the exact frozen
    model is handed to the normal main prompt-training path without reloading.
    """
    expected = input_metadata(args, dataset)
    path = Path(args.pseudo_label_path) if args.pseudo_label_path else default_pseudo_label_path(args, dataset)
    args.pseudo_label_path = str(path)
    teacher = None
    if path.is_file() and not args.rebuild_pseudo_labels:
        print(f"[Pseudo Labels] loading {path}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_payload(payload, expected)
    else:
        print("[Stage 1] Original frozen DFN5B-224; independent classification over ALL seen class names.", flush=True)
        teacher = teacher_factory()
        if teacher is None:
            raise RuntimeError("Pseudo-label generation requires the original DFN5B teacher.")
        payload = {"metadata": expected, **classify_seen(teacher, dataset, args.pseudo_batch_size,
                                                       args.workers, args.progress)}
        validate_payload(payload, expected)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        print(f"[Pseudo Labels] saved {path}", flush=True)
    sampling = dataset.set_pseudo_labels(payload["sketch_labels"], payload["photo_labels"])
    if len(dataset) < max(args.batch_size, args.teacher_pretrain_batch_size):
        raise ValueError("Too few pseudo-paired sketches for a complete teacher/student batch.")
    args.pseudo_label_fingerprint = prediction_fingerprint(payload)
    report_path = Path(args.pseudo_label_dir) / path.name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_reports(report_path, payload, dataset, args.root, sampling)
    print(f"[Pseudo Labels] CSV/JSON reports: {report_path.parent / path.stem}", flush=True)
    print(f"[Pseudo Labels] {sampling}; fingerprint={args.pseudo_label_fingerprint[:16]}", flush=True)
    return teacher
