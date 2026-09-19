"""Build the pinned offline Kaggle bundle for the Gap-CoRe teacher audit.

Run this file as one Kaggle cell with Internet enabled. Save the notebook
output, then attach that output to the offline GPU notebook.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKING = Path("/kaggle/working")
BUNDLE = (
    WORKING
    / ("gap_core_bundle_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"))
    / "offline_bundle"
)
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
DFN_DIR = BUNDLE / "dfn5b_openclip"

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/gap-core-teacher-audit"
COMMIT = "e48af9e4732630eaee12fb98ca94ca11eaa956c2"
TASK = "gap_core_teacher_audit"
ENTRYPOINT = "src.train"
DATASET = "b20dccn616nguynhutun/sketchy"

DFN_REPOSITORY = "apple/DFN5B-CLIP-ViT-H-14"
DFN_FILENAME = "open_clip_pytorch_model.bin"
DFN_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
DFN_SHA256 = "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae"
STUDENT_FILENAME = "ViT-B-32.pt"
STUDENT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
for directory in (WHEELS, SOURCE, CLIP_CACHE, DFN_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/6] Empty bundle created:", BUNDLE)

requirements = """
open-clip-torch
pytorch-lightning
torchmetrics
lightning-utilities
huggingface-hub
ftfy
regex
tensorboard
packaging
tqdm
numpy
pillow
matplotlib
"""
requirements_path = BUNDLE / "requirements.txt"
requirements_path.write_text(requirements.strip() + "\n", encoding="utf-8")
run(
    [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(WHEELS),
        "--only-binary=:all:",
        "-r",
        str(requirements_path),
    ],
    WORKING,
)
for pattern in (
    "torch-*.whl",
    "torchvision-*.whl",
    "torchaudio-*.whl",
    "triton-*.whl",
    "nvidia_*.whl",
    "cuda_*.whl",
):
    for wheel in WHEELS.glob(pattern):
        print("Removing Kaggle-provided wheel:", wheel.name)
        wheel.unlink()

required_wheels = (
    "open_clip_torch-*.whl",
    "pytorch_lightning-*.whl",
    "torchmetrics-*.whl",
    "lightning_utilities-*.whl",
    "huggingface_hub-*.whl",
    "ftfy-*.whl",
    "regex-*.whl",
    "matplotlib-*.whl",
)
missing_wheels = [
    pattern for pattern in required_wheels if not list(WHEELS.glob(pattern))
]
if missing_wheels:
    raise FileNotFoundError("Missing required wheels:\n" + "\n".join(missing_wheels))
print("[2/6] Wheels downloaded:", len(list(WHEELS.glob("*.whl"))))

run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "huggingface-hub",
        "ftfy",
        "regex",
        "packaging",
    ],
    WORKING,
)

project = SOURCE / "KD-SBIR"
run(
    [
        "git",
        "clone",
        "--branch",
        BRANCH,
        "--single-branch",
        REPOSITORY,
        str(project),
    ],
    WORKING,
)
run(["git", "checkout", "--detach", COMMIT], project)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=project, text=True
).strip()
if actual_commit != COMMIT:
    raise RuntimeError(f"Repository commit mismatch: {actual_commit} != {COMMIT}")
print("[3/6] Source commit:", actual_commit)

from huggingface_hub import hf_hub_download

downloaded_dfn = Path(
    hf_hub_download(
        repo_id=DFN_REPOSITORY,
        filename=DFN_FILENAME,
        revision=DFN_REVISION,
    )
)
dfn_target = DFN_DIR / DFN_FILENAME
shutil.copy2(downloaded_dfn, dfn_target)
actual_dfn_sha = file_sha256(dfn_target)
if actual_dfn_sha != DFN_SHA256:
    raise RuntimeError(f"DFN5B checksum mismatch: {actual_dfn_sha} != {DFN_SHA256}")
print("[4/6] DFN5B downloaded:", dfn_target)

for module_name in list(sys.modules):
    if module_name == "clip" or module_name.startswith("clip."):
        del sys.modules[module_name]
sys.path.insert(0, str(project))
from clip import clip as project_clip

downloaded_student = Path(project_clip.download_model("ViT-B/32"))
student_target = CLIP_CACHE / STUDENT_FILENAME
shutil.copy2(downloaded_student, student_target)
actual_student_sha = file_sha256(student_target)
if actual_student_sha != STUDENT_SHA256:
    raise RuntimeError(
        f"ViT-B/32 checksum mismatch: {actual_student_sha} != {STUDENT_SHA256}"
    )
print("[5/6] ViT-B/32 downloaded:", student_target)

required_paths = (
    BUNDLE / "requirements.txt",
    WHEELS,
    project / ".git",
    project / "clip" / "model.py",
    project / "src" / "dataset.py",
    project / "src" / "losses.py",
    project / "src" / "model.py",
    project / "src" / "teacher_prompts.py",
    project / "src" / "gap_core_audit.py",
    project / "src" / "train.py",
    project / "tests" / "test_gap_core_audit.py",
    project / "test" / "kaggle_gap_core_teacher_audit.py",
    project / "docs" / "gap_core_teacher_audit.md",
    dfn_target,
    student_target,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )

manifest = {
    "repository": REPOSITORY,
    "branch": BRANCH,
    "commit": actual_commit,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "dataset": DATASET,
    "teacher_repository": DFN_REPOSITORY,
    "teacher_revision": DFN_REVISION,
    "teacher_filename": DFN_FILENAME,
    "teacher_sha256": DFN_SHA256,
    "teacher_size": dfn_target.stat().st_size,
    "student_filename": STUDENT_FILENAME,
    "student_sha256": STUDENT_SHA256,
    "student_size": student_target.stat().st_size,
    "python_version": sys.version,
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

bundle_size = sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
print("[6/6] Bundle validated")
print("=" * 70)
print("ONLINE GAP-CORE BUNDLE COMPLETE")
print("=" * 70)
print("Bundle:", BUNDLE)
print("Branch:", BRANCH)
print("Commit:", actual_commit)
print("Bundle size:", f"{bundle_size / 1024**3:.3f} GiB")
print("Use Save Version -> Save & Run All -> Always save output.")
