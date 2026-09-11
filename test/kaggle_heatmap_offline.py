"""Restore bundle, train, and visualize attention heatmaps on offline Kaggle.

This script:
  1. Finds and validates the offline bundle from the online notebook.
  2. Installs Python wheels, restores model caches.
  3. Trains teacher prompts (1 epoch) then student (5 epochs).
  4. Runs attention heatmap visualization on all unseen classes.
  5. Zips the output for download.

Attach the online bundle output and the ``sketchy`` dataset to an
Internet-disabled GPU notebook, then run this script.
"""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/attention-heatmap-visualization"
EXPECTED_COMMIT = "5f30aa162a043428b9041c45d972274471e9c131"
EXPECTED_TASK = "attention_heatmap_visualization"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR"
SKETCHY_ROOT = Path(
    "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Phase 1: Locate and validate the offline bundle ──────────────────

WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_ROOT)

manifest_paths = []
for pattern in (
    "/kaggle/input/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/*/offline_bundle/bundle_manifest.json",
):
    manifest_paths.extend(Path(path) for path in glob.glob(pattern))
manifest_paths = sorted(set(manifest_paths))
print("Manifest files found:", len(manifest_paths))

matching_bundles = []
manifest_reports = []
for manifest_path in manifest_paths:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        manifest_reports.append(
            f"- {manifest_path}\n"
            f"  unreadable: {type(error).__name__}: {error}"
        )
        continue

    manifest_reports.append(
        f"- {manifest_path}\n"
        f"  branch: {manifest.get('branch')}\n"
        f"  commit: {manifest.get('commit')}\n"
        f"  task: {manifest.get('task')}\n"
        f"  entrypoint: {manifest.get('entrypoint')}\n"
        f"  dataset: {manifest.get('dataset')}"
    )
    commit_ok = (
        EXPECTED_COMMIT is None
        or manifest.get("commit") == EXPECTED_COMMIT
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and commit_ok
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("dataset") == EXPECTED_DATASET
    ):
        matching_bundles.append((manifest_path, manifest))

if not matching_bundles:
    raise FileNotFoundError(
        "Cannot find the required attention heatmap visualization bundle.\n\n"
        f"Expected branch: {EXPECTED_BRANCH}\n"
        f"Expected task: {EXPECTED_TASK}\n\n"
        "Manifest files inspected:\n"
        + ("\n".join(manifest_reports) if manifest_reports else "(none)")
    )
if len(matching_bundles) > 1:
    print("Warning: matching bundle attached more than once; using the first.")

manifest_path, manifest = matching_bundles[0]
bundle = manifest_path.parent
wheels = bundle / "wheels"
source_project = bundle / "source" / "KD-SBIR"
clip_cache = bundle / "clip_cache"
dfn_source = bundle / "dfn5b_openclip" / manifest["teacher_filename"]
student_source = clip_cache / manifest["student_filename"]
print("Selected bundle:", bundle)
print("Branch:", manifest["branch"])
print("Commit:", manifest["commit"])


# ── Phase 2: Validate bundle contents ────────────────────────────────

required_bundle_paths = (
    bundle / "requirements.txt",
    wheels,
    source_project / ".git",
    source_project / "clip" / "model.py",
    source_project / "src" / "dataset.py",
    source_project / "src" / "losses.py",
    source_project / "src" / "model.py",
    source_project / "src" / "teacher_prompts.py",
    source_project / "src" / "train.py",
    source_project / "src" / "visualize_attention.py",
    dfn_source,
    student_source,
)
missing_paths = [
    str(path) for path in required_bundle_paths if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )
wheel_files = list(wheels.glob("*.whl"))
if not wheel_files:
    raise FileNotFoundError(f"No wheel files found inside {wheels}")
print("Offline wheels:", len(wheel_files))


# ── Phase 3: Install packages offline ────────────────────────────────

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels),
        "open-clip-torch",
        "pytorch-lightning",
        "torchmetrics",
        "lightning-utilities",
        "huggingface-hub",
        "ftfy",
        "regex",
        "tensorboard",
        "packaging",
        "tqdm",
        "numpy",
        "pillow",
        "matplotlib",
    ],
    cwd=WORKING_ROOT,
    check=True,
)
print("Offline Python packages installed")


# ── Phase 4: Validate checksums ──────────────────────────────────────

if dfn_source.stat().st_size != manifest["teacher_size"]:
    raise RuntimeError("DFN5B checkpoint size mismatch.")
actual_dfn_sha = file_sha256(dfn_source)
if actual_dfn_sha != manifest["teacher_sha256"]:
    raise RuntimeError(
        "DFN5B checksum mismatch:\n"
        f"Expected: {manifest['teacher_sha256']}\n"
        f"Actual:   {actual_dfn_sha}"
    )

if student_source.stat().st_size != manifest["student_size"]:
    raise RuntimeError("ViT-B/32 checkpoint size mismatch.")
actual_student_sha = file_sha256(student_source)
if actual_student_sha != manifest["student_sha256"]:
    raise RuntimeError(
        "ViT-B/32 checksum mismatch:\n"
        f"Expected: {manifest['student_sha256']}\n"
        f"Actual:   {actual_student_sha}"
    )


# ── Phase 5: Restore model caches ───────────────────────────────────

# OpenAI CLIP cache.
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
shutil.copy2(student_source, student_target)
print("Student checkpoint:", student_target)

# Hugging Face snapshot for offline open_clip.
hf_home = WORKING_ROOT / "huggingface"
hf_hub = hf_home / "hub"
repo_cache_name = "models--" + manifest["teacher_repo"].replace("/", "--")
repo_cache = hf_hub / repo_cache_name
snapshot = repo_cache / "snapshots" / manifest["teacher_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
dfn_target = snapshot / manifest["teacher_filename"]
shutil.copy2(dfn_source, dfn_target)
refs = repo_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(manifest["teacher_revision"], encoding="utf-8")

os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_hub)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
print("DFN5B checkpoint:", dfn_target)


# ── Phase 6: Copy source to working directory ───────────────────────

os.chdir(WORKING_ROOT)
if WORKING_PROJECT.exists():
    shutil.rmtree(WORKING_PROJECT)
shutil.copytree(source_project, WORKING_PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
print("Repository copied:", WORKING_PROJECT)
print("Commit:", actual_commit)


# ── Phase 7: Validate dataset ───────────────────────────────────────

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
print("Dataset:", SKETCHY_ROOT)


# ── Phase 8: Smoke test ─────────────────────────────────────────────

smoke_test = """
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch, open_clip, pytorch_lightning, matplotlib
print('PyTorch:', torch.__version__)
print('OpenCLIP:', getattr(open_clip, '__version__', 'unknown'))
print('Lightning:', pytorch_lightning.__version__)
print('Matplotlib:', matplotlib.__version__)
# Quick import check for visualize_attention.
from src.visualize_attention import AttentionCapture, overlay_heatmap
print('Visualization module: OK')
"""
subprocess.run(
    [sys.executable, "-c", smoke_test],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print()
print("=" * 70)
print("OFFLINE SETUP COMPLETE — READY TO TRAIN")
print("=" * 70)


# ── Phase 9: Train ──────────────────────────────────────────────────

EXP_NAME = "heatmap_viz_sketchy2"

train_command = [
    sys.executable,
    "-m",
    "src.train",
    "--root",
    str(SKETCHY_ROOT),
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
    "1",
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
    "--lambda_domain",
    "3.0",
    "--lambda_modality",
    "1.0",
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
    "--exp_name",
    EXP_NAME,
    "--progress",
]

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("Command:", " ".join(train_command))
print()

subprocess.run(
    train_command,
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


# ── Phase 10: Find best checkpoint ──────────────────────────────────

saved_dir = WORKING_PROJECT / "saved_models" / EXP_NAME
last_ckpt = saved_dir / "last.ckpt"

# Prefer best checkpoint, fall back to last.
best_candidates = sorted(saved_dir.glob("epoch=*.ckpt"))
if best_candidates:
    ckpt_path = str(best_candidates[-1])
    print("Using best checkpoint:", ckpt_path)
elif last_ckpt.exists():
    ckpt_path = str(last_ckpt)
    print("Using last checkpoint:", ckpt_path)
else:
    raise FileNotFoundError(
        f"No checkpoints found in {saved_dir}. Training may have failed."
    )


# ── Phase 11: Visualize attention heatmaps ───────────────────────────

HEATMAP_DIR = WORKING_ROOT / "heatmaps"

viz_command = [
    sys.executable,
    "-m",
    "src.visualize_attention",
    "--root",
    str(SKETCHY_ROOT),
    "--dataset",
    "sketchy_2",
    "--ckpt_path",
    ckpt_path,
    "--n_ctx_visual",
    "3",
    "--prompt_depth",
    "12",
    "--seed",
    "42",
    "--method",
    "rollout",
    "--sketches_per_class",
    "5",
    "--top_k",
    "10",
    "--output_dir",
    str(HEATMAP_DIR),
]

print()
print("=" * 70)
print("STARTING ATTENTION HEATMAP VISUALIZATION")
print("=" * 70)
print("Command:", " ".join(viz_command))
print()

subprocess.run(
    viz_command,
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print()
print("=" * 70)
print("ATTENTION HEATMAP VISUALIZATION COMPLETE")
print("=" * 70)
print("Output directory:", HEATMAP_DIR)
print("ZIP file:", HEATMAP_DIR.with_suffix(".zip"))
print()
print("Download heatmaps.zip from the notebook output.")
