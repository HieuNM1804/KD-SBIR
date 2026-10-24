"""ONE ONLINE Kaggle cell: original weights for five frozen teachers + source + wheels."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/teacher-zero-shot-classification"
SOURCE_COMMIT = "c62ea560cfac2724ba01b21ce4497422d58994f0"
TASK = "teacher_zero_shot_classification"
WORKING = Path("/kaggle/working")
BUNDLE = WORKING / ("teacher_classification_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")) / "offline_bundle"
PROJECT = BUNDLE / "KD-SBIR"
MODELS = BUNDLE / "models"
WHEELS = BUNDLE / "wheels"
TEACHER_KEYS = ["dfn5b_h14", "dfn2b_l14", "laion_h14", "metaclip_l14", "openai_l14"]
TARGET_PYTHONS = ("3.12", "3.13")
# Optional: point to existing attached weight files to avoid downloading them again.
# Example: {"dfn5b_h14": "/kaggle/input/.../dfn5b_openclip/open_clip_pytorch_model.bin"}
REUSE_WEIGHTS = {}
ROOTS = ["open-clip-torch==3.2.0", "huggingface-hub<1.0", "safetensors",
         "ftfy", "regex", "numpy", "pillow", "tqdm", "packaging"]


def run(command, cwd=WORKING):
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True)
WHEELS.mkdir()
print("[1/5] New bundle:", BUNDLE, flush=True)
run(["git", "clone", "--single-branch", "--branch", BRANCH, REPOSITORY, str(PROJECT)])
run(["git", "checkout", "--detach", SOURCE_COMMIT], PROJECT)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
if actual != SOURCE_COMMIT:
    raise RuntimeError(f"Source commit mismatch: {actual}")
sys.path.insert(0, str(PROJECT))
from src.teacher_classification_specs import TEACHERS
from src.kaggle_classification_wheels import download_runtime_wheels

# Resolve separately for each ABI: do not force Python-3.12 NumPy pins on 3.13.
# The existing Kaggle torch/torchvision stack is provided, never downloaded.
run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub<1.0", "packaging"])
inventories = download_runtime_wheels(ROOTS, TARGET_PYTHONS, BUNDLE, WHEELS, run)
print("[2/5] Non-CUDA wheels:", len(list(WHEELS.glob("*.whl"))), flush=True)

from huggingface_hub import hf_hub_download

files = []
for index, key in enumerate(TEACHER_KEYS, 1):
    spec = TEACHERS[key]
    directory = MODELS / key
    directory.mkdir()
    path = directory / spec["filename"]
    if key in REUSE_WEIGHTS:
        shutil.copy2(Path(REUSE_WEIGHTS[key]), path)
    else:
        hf_hub_download(repo_id=spec["repository"], filename=spec["filename"],
                        revision=spec["revision"], local_dir=directory)
    print(f"[3/5] Checking {index}/{len(TEACHER_KEYS)}: {key}", flush=True)
    actual_sha = sha256(path)
    if actual_sha != spec["sha256"]:
        raise RuntimeError(f"Checkpoint checksum mismatch for {key}: {actual_sha}")
    files.append(dict(path=path.relative_to(BUNDLE).as_posix(), bytes=path.stat().st_size,
                      sha256=actual_sha))
    print(f"  ready: {path.stat().st_size / 1024**3:.2f} GiB", flush=True)

# Source files and wheels are small; full integrity inventory excludes .git/HF metadata.
for directory in (PROJECT, WHEELS):
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(BUNDLE)
        if not path.is_file() or any(part in {".git", "__pycache__", ".cache"} for part in relative.parts):
            continue
        files.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
for path in sorted(BUNDLE.glob("requirements*.txt")):
    files.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha256(path)))
manifest = dict(task=TASK, repository=REPOSITORY, branch=BRANCH, commit=SOURCE_COMMIT,
                python=list(sys.version_info[:2]), target_pythons=list(TARGET_PYTHONS),
                runtime_requirements=ROOTS, teacher_keys=TEACHER_KEYS,
                teacher_specs={key: TEACHERS[key] for key in TEACHER_KEYS},
                wheel_inventories=inventories, files=files)
(BUNDLE / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("[4/5] Five original teacher checkpoints bundled; no student checkpoint or dataset copied.", flush=True)
print("[5/5] COMPLETE:", BUNDLE, flush=True)
print("Save Version -> Save & Run All -> Always save output.")
print("Attach that output to the offline GPU notebook with the datasets; run the offline cell then the run cell.")
