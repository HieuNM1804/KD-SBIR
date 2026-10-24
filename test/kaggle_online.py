"""ONE ONLINE Kaggle cell: bundle code, DFN5B-224, ViT-B/32 and non-CUDA wheels."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/dfn5b-pseudo-label-retrieval"
SOURCE_COMMIT = "2ed12a595b429dec2c0069488261784df3945c37"
TASK = "dfn5b224_pseudo_label_retrieval"
WORKING = Path("/kaggle/working")
BUNDLE = WORKING / ("dfn5b224_pseudo_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")) / "offline_bundle"
PROJECT = BUNDLE / "KD-SBIR"
MODELS = BUNDLE / "models"
WHEELS = BUNDLE / "wheels"
TARGET_PYTHONS = ("3.12", "3.13")
# Optional existing attached checkpoint paths. None downloads that weight online.
REUSE_TEACHER = None  # "/kaggle/input/.../dfn5b_openclip/open_clip_pytorch_model.bin"
REUSE_STUDENT = None  # "/kaggle/input/.../clip_cache/ViT-B-32.pt"
ROOTS = ["open-clip-torch==3.2.0", "huggingface-hub<1.0", "safetensors",
         "pytorch-lightning>=2.5,<3", "torchmetrics>=1.6,<2", "lightning-utilities",
         "ftfy", "regex", "numpy", "pillow", "tqdm", "packaging", "tensorboard", "matplotlib"]


def run(command, cwd=WORKING):
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True)
WHEELS.mkdir()
print("[1/5] New pseudo-label training bundle:", BUNDLE, flush=True)
run(["git", "clone", "--single-branch", "--branch", BRANCH, REPOSITORY, str(PROJECT)])
run(["git", "checkout", "--detach", SOURCE_COMMIT], PROJECT)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
if actual != SOURCE_COMMIT:
    raise RuntimeError(f"Source commit mismatch: {actual}")
sys.path.insert(0, str(PROJECT))
from src.kaggle_pseudo_specs import TEACHER, STUDENT
from src.kaggle_bundle_wheels import download_runtime_wheels

run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub<1.0", "packaging"])
# Resolve for each interpreter independently. Never reuse 3.12 NumPy pins for 3.13.
inventories = download_runtime_wheels(ROOTS, TARGET_PYTHONS, BUNDLE, WHEELS, run)
print("[2/5] Non-CUDA dependency wheels:", len(list(WHEELS.glob("*.whl"))), flush=True)

from huggingface_hub import hf_hub_download

teacher_directory = MODELS / "dfn5b224"
teacher_directory.mkdir()
teacher_path = teacher_directory / TEACHER["filename"]
if REUSE_TEACHER:
    shutil.copy2(Path(REUSE_TEACHER), teacher_path)
else:
    hf_hub_download(repo_id=TEACHER["repository"], filename=TEACHER["filename"],
                    revision=TEACHER["revision"], local_dir=teacher_directory)
print("[3/5] Checking original DFN5B-224 checkpoint...", flush=True)
teacher_sha = sha256(teacher_path)
if teacher_sha != TEACHER["sha256"]:
    raise RuntimeError(f"DFN5B checksum mismatch: {teacher_sha}")

student_path = MODELS / STUDENT["filename"]
if REUSE_STUDENT:
    shutil.copy2(Path(REUSE_STUDENT), student_path)
else:
    urllib.request.urlretrieve(STUDENT["url"], student_path)
print("[4/5] Checking original ViT-B/32 student...", flush=True)
student_sha = sha256(student_path)
if student_sha != STUDENT["sha256"]:
    raise RuntimeError(f"Student checksum mismatch: {student_sha}")

files = [dict(path=path.relative_to(BUNDLE).as_posix(), bytes=path.stat().st_size, sha256=value)
         for path, value in ((teacher_path, teacher_sha), (student_path, student_sha))]
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
                runtime_requirements=ROOTS, teacher=TEACHER, student=STUDENT,
                wheel_inventories=inventories, files=files)
(BUNDLE / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("[5/5] COMPLETE:", BUNDLE, flush=True)
print("No image dataset downloaded and no training performed by this setup cell.")
print("Save Version -> Save & Run All -> Always save output.")
print("Attach output + dataset to offline GPU notebook; run offline setup then kaggle_train.ipy.")
