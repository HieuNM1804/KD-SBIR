"""Audit common/gap teacher prompts on Sketchy2 and export one diagnostics ZIP.

Paste this whole file into one offline Kaggle GPU notebook cell after running
`kaggle_gap_core_offline.py`. The script never trains a student.
"""

import csv
import glob
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/sketchy2_core_teacher2_v7.pt"
)
STAMP = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
RUN_NAME = "gap_core_teacher_audit_sketchy2_s42_" + STAMP
OUT = Path("/kaggle/working") / ("gap_core_teacher_audit_" + STAMP)
OUT.mkdir(parents=True, exist_ok=False)
TEACHER_CACHE.parent.mkdir(parents=True, exist_ok=True)


def run_logged(label, command):
    path = OUT / f"{label}.log"
    print("=" * 78, flush=True)
    print(label.upper(), flush=True)
    print(" ".join(command), flush=True)
    print("=" * 78, flush=True)
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            log.write(line)
        code = process.wait()
    print(f"[{label}] exit code: {code}", flush=True)
    return code


# Build the exact prompt-tuned teacher cache only when it is not attached.
cache_command = [
    sys.executable,
    "-u",
    "-m",
    "src.train",
    "--root",
    str(ROOT),
    "--dataset",
    "sketchy_2",
    "--epochs",
    "5",
    "--workers",
    "8",
    "--batch_size",
    "64",
    "--test_batch_size",
    "1024",
    "--n_ctx_visual",
    "3",
    "--prompt_depth",
    "12",
    "--teacher_pretrain_epochs",
    "2",
    "--teacher_cache_path",
    str(TEACHER_CACHE),
    "--teacher_pretrain_batch_size",
    "64",
    "--teacher_n_ctx_visual",
    "10",
    "--teacher_prompt_depth",
    "12",
    "--teacher_prompt_std",
    "0.02",
    "--teacher_prompt_lr",
    "3e-2",
    "--teacher_prompt_seed",
    "42",
    "--teacher_prompt_gradient_checkpointing",
    "--teacher_momentum",
    "0.9",
    "--teacher_weight_decay",
    "1e-3",
    "--lambda_teacher_retrieval",
    "1.5",
    "--teacher_triplet_margin",
    "0.2",
    "--photo_text_kd_temperature",
    "0.15",
    "--sketch_text_kd_temperature",
    "0.02",
    "--lr",
    "1e-2",
    "--momentum",
    "0.9",
    "--weight_decay",
    "5e-4",
    "--seed",
    "42",
    "--lambda_domain",
    "3.0",
    "--lambda_modality",
    "1.0",
    "--lambda_core",
    "0.0",
    "--teacher_cache_only",
    "--no_progress",
    "--exp_name",
    RUN_NAME,
]
cache_code = run_logged("teacher_cache", cache_command)
if cache_code != 0 or not TEACHER_CACHE.is_file():
    archive = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in OUT.rglob("*"):
            if path.is_file():
                bundle.write(path, Path("report") / path.relative_to(OUT))
    print("Failure diagnostics ZIP:", archive)
    raise RuntimeError("Teacher cache preparation failed; inspect teacher_cache.log.")

# The cache command uses PROJECT as its subprocess cwd, which does not change
# the notebook kernel cwd. Make the repository package importable explicitly.
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import open_clip
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.data_config import UNSEEN_CLASSES
from src.dataset import load_image, normal_transform
from src.gap_core_audit import build_gap_audit_rows, summarize_gap_rows
from src.model import DFN5B_MODEL, DFN5B_PRETRAINED, _retrieval_metrics
from src.teacher_prompts import build_teacher_prompt_controller


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


payload = load_cache(TEACHER_CACHE)
prompt_state = payload.get("teacher_prompt_state_dict")
if not prompt_state:
    raise RuntimeError("Teacher cache does not contain the trained prompt state.")
metadata = payload["metadata"]

device = torch.device("cuda")
teacher = open_clip.create_model(
    DFN5B_MODEL,
    pretrained=DFN5B_PRETRAINED,
    precision="fp16",
    device=device,
)
teacher.eval().requires_grad_(False)
controller = build_teacher_prompt_controller(
    teacher,
    n_ctx=metadata["teacher_n_ctx_visual"],
    depth=metadata["teacher_prompt_depth"],
    std=metadata["teacher_prompt_std"],
    seed=metadata["teacher_prompt_seed"],
).to(device)
controller.load_state_dict(prompt_state, strict=True)
controller.eval().requires_grad_(False)
teacher_dtype = teacher.visual.conv1.weight.dtype


class LabeledPaths(Dataset):
    def __init__(self, paths, labels, size=224):
        self.paths = paths
        self.labels = labels
        self.transform = normal_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index], 224)), self.labels[index]


@torch.no_grad()
def encode_states(paths, labels, modality, batch_size=64, workers=8):
    loader = DataLoader(
        LabeledPaths(paths, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    full, common, output_labels = [], [], []
    completed = 0
    for images, current_labels in loader:
        images = images.to(device, dtype=teacher_dtype, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            full_batch = controller(images, modality, prompt_mode="full")
            common_batch = controller(images, modality, prompt_mode="common")
        full.append(F.normalize(full_batch.float(), dim=-1).cpu())
        common.append(F.normalize(common_batch.float(), dim=-1).cpu())
        output_labels.append(current_labels.cpu())
        completed += len(images)
        print(
            f"[Gap Audit] {modality}: {100 * completed / len(paths):.1f}%",
            flush=True,
        )
    return torch.cat(full), torch.cat(common), torch.cat(output_labels)


def paths_and_labels(split_classes, modality, per_class=None):
    paths, labels = [], []
    for label, category in enumerate(split_classes):
        current = sorted(glob.glob(str(ROOT / modality / category / "*")))
        if per_class is not None:
            current = current[:per_class]
        paths.extend(current)
        labels.extend([label] * len(current))
    return paths, labels


unseen_classes = UNSEEN_CLASSES["sketchy_2"]
all_classes = sorted(
    path.name
    for path in (ROOT / "sketch").iterdir()
    if path.is_dir() and path.name != ".ipynb_checkpoints"
)
seen_classes = [name for name in all_classes if name not in set(unseen_classes)]

# Seen audit uses equal class budgets so large categories cannot dominate it.
seen_sketch_paths, seen_sketch_labels = paths_and_labels(
    seen_classes, "sketch", per_class=4
)
seen_photo_paths, seen_photo_labels = paths_and_labels(
    seen_classes, "photo", per_class=4
)
seen_full_sketch, seen_common_sketch, seen_sketch_labels = encode_states(
    seen_sketch_paths, seen_sketch_labels, "sketch"
)
seen_full_photo, seen_common_photo, seen_photo_labels = encode_states(
    seen_photo_paths, seen_photo_labels, "photo"
)
rows = build_gap_audit_rows(
    seen_full_sketch,
    seen_common_sketch,
    seen_sketch_labels,
    seen_full_photo,
    seen_common_photo,
    seen_photo_labels,
    seed=42,
)
summary = summarize_gap_rows(rows, seed=42, bootstrap_samples=2000)

with (OUT / "seen_query_audit.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Full unseen retrieval checks whether the real modality-gap prompt generalizes.
unseen_sketch_paths, unseen_sketch_labels = paths_and_labels(
    unseen_classes, "sketch"
)
unseen_photo_paths, unseen_photo_labels = paths_and_labels(unseen_classes, "photo")
unseen_full_sketch, unseen_common_sketch, unseen_sketch_labels = encode_states(
    unseen_sketch_paths, unseen_sketch_labels, "sketch"
)
unseen_full_photo, unseen_common_photo, unseen_photo_labels = encode_states(
    unseen_photo_paths, unseen_photo_labels, "photo"
)


def retrieval(features_sketch, features_photo):
    mean_ap, precision, map_k, p_k = _retrieval_metrics(
        features_sketch,
        features_photo,
        unseen_sketch_labels,
        unseen_photo_labels,
        "sketchy_2",
    )
    return {
        "mAP_k": map_k,
        "precision_k": p_k,
        "mAP": float(mean_ap),
        "precision": float(precision),
    }


retrieval_common = retrieval(unseen_common_sketch, unseen_common_photo)
retrieval_full = retrieval(unseen_full_sketch, unseen_full_photo)
retrieval_delta = {
    "mAP": retrieval_full["mAP"] - retrieval_common["mAP"],
    "precision": retrieval_full["precision"] - retrieval_common["precision"],
}

gate_checks = {
    "full_improves_unseen_mAP": retrieval_delta["mAP"] > 0,
    "verified_beats_shuffled_ci95": summary["verified_minus_shuffled"][
        "ci95_low"
    ]
    > 0,
    "positive_evidence_ci95": summary["positive_delta"]["ci95_low"] > 0,
}
gate_checks["passed"] = all(gate_checks.values())

report = {
    "created": datetime.now(UTC).isoformat(),
    "dataset": "sketchy_2",
    "seed": 42,
    "teacher_cache": str(TEACHER_CACHE),
    "teacher_cache_format": metadata.get("format_version"),
    "prompt_decomposition": {
        "full_photo": "P_common + P_gap",
        "full_sketch": "P_common - P_gap",
        "common_both_modalities": "(P_photo + P_sketch) / 2",
        "trainable_parameters_changed": False,
    },
    "seen_sample": {
        "classes": len(seen_classes),
        "sketches": len(seen_sketch_paths),
        "photos": len(seen_photo_paths),
        "per_class_per_modality": 4,
    },
    "seen_gap_correction": summary,
    "unseen_retrieval": {
        "common": retrieval_common,
        "full": retrieval_full,
        "full_minus_common": retrieval_delta,
    },
    "gate": gate_checks,
    "decision": (
        "implement_student_gap_distillation"
        if gate_checks["passed"]
        else "redesign_or_stop_gap_distillation"
    ),
    "notes": [
        "Positive and negative identities are selected once under the common state.",
        "Full and shuffled corrections are evaluated on those same fixed pairs.",
        "The shuffled control reassigns full-minus-common feature residuals across image identities.",
        "No student is trained and no checkpoint or feature tensor is included in the ZIP.",
    ],
}
(OUT / "gap_core_teacher_audit.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(
    ["common", "full"],
    [100 * retrieval_common["mAP"], 100 * retrieval_full["mAP"]],
)
axes[0].set_ylabel("Unseen mAP@200 (%)")
axes[0].set_title("Teacher retrieval")
axes[1].hist(
    [row["verified_margin_correction"] for row in rows],
    bins=35,
    alpha=0.7,
    label="verified gap",
)
axes[1].hist(
    [row["shuffled_margin_correction"] for row in rows],
    bins=35,
    alpha=0.6,
    label="shuffled residual",
)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Fixed-pair margin correction")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "gap_core_teacher_audit.png", dpi=170)
plt.close(fig)

source_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "source_commit": source_commit,
            "cache_stage_return_code": cache_code,
            "teacher_only": True,
            "student_training_started": False,
            "gate_passed": gate_checks["passed"],
            "decision": report["decision"],
        },
        indent=2,
    ),
    encoding="utf-8",
)

archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("report") / path.relative_to(OUT))

from IPython.display import FileLink, Image, display

display(Image(filename=str(OUT / "gap_core_teacher_audit.png")))
print("Gap-CoRe teacher gate:", "PASS" if gate_checks["passed"] else "FAIL")
print("Decision:", report["decision"])
print("Send this ZIP:", archive)
display(FileLink(str(archive)))
