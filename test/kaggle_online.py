"""One online Kaggle cell: build the bidirectional teacher triplet and symmetric student KD bundle."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime


WORKING = Path("/kaggle/working")
BUNDLE = WORKING / (
    "teacher_triplet_bidirectional_symmetric_kd_bundle_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
) / "offline_bundle"
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
DFN_DIR = BUNDLE / "dfn5b_openclip"

REPO_URL = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/teacher-triplet-bidirectional-symmetric-kd"
BASE_COMMIT = "a8d22f4f95b6f853ff164a744ca9c4b111f54aa1"
# Pin the source commit. The later bundle-script commit intentionally differs.
COMMIT = "908e482f7a99cb16f023dd3527fdfe24cfef8ca4"
TASK = "teacher_triplet_bidirectional_symmetric_kd"
ENTRYPOINT = "src.train"
DATASET = "b20dccn616nguynhutun/sketchy"

DFN_REPO = "apple/DFN5B-CLIP-ViT-H-14"
DFN_FILENAME = "open_clip_pytorch_model.bin"
DFN_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
DFN_SHA256 = (
    "d67de50faa7f3ddce52fbab4f4656b046"
    "86a0bb15c26ebd0144d375cfa08b8ae"
)
STUDENT_FILENAME = "ViT-B-32.pt"
STUDENT_SHA256 = (
    "40d365715913c9da98579312b702a82c1"
    "8be219cc2a73407c4526f58eba950af"
)


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
print("[1/6] New symmetric-KD-based bidirectional teacher triplet and symmetric student KD bundle:", BUNDLE)

# Resolve using Kaggle's installed PyTorch stack, then download only the
# non-CUDA dependency closure at exact versions for this Python environment.
# requirements.txt records that inventory; offline installation requests the
# runtime roots below, allowing compatible packages already on Kaggle to remain.
roots = [
    "open-clip-torch==3.2.0", "pytorch-lightning", "torchmetrics",
    "lightning-utilities", "huggingface-hub", "ftfy", "regex", "tensorboard",
    "packaging", "tqdm", "numpy", "pillow", "matplotlib",
]
run([sys.executable, "-m", "pip", "install", "-q", *roots], WORKING)
from importlib.metadata import distribution
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pending = [Requirement(spec).name for spec in roots]
resolved = {}
while pending:
    package = canonicalize_name(pending.pop())
    if package in resolved or package in {"torch", "torchvision", "torchaudio", "triton"} or package.startswith(("nvidia-", "cuda-")):
        continue
    installed = distribution(package)
    resolved[package] = installed.version
    for specification in installed.requires or []:
        dependency = Requirement(specification)
        if dependency.marker is None or dependency.marker.evaluate({"extra": ""}):
            pending.append(dependency.name)
requirements_path = BUNDLE / "requirements.txt"
requirements_path.write_text(
    "".join(f"{name}=={version}\n" for name, version in sorted(resolved.items())),
    encoding="utf-8",
)
run([
    sys.executable, "-m", "pip", "download", "--no-deps",
    "--only-binary=:all:", "--dest", str(WHEELS), "-r", str(requirements_path),
], WORKING)
print("[2/6] Pinned non-CUDA dependency wheels:", len(list(WHEELS.glob("*.whl"))))

project = SOURCE / "KD-SBIR"
run(
    [
        "git",
        "clone",
        "--branch",
        BRANCH,
        "--single-branch",
        REPO_URL,
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
if subprocess.check_output(
    ["git", "merge-base", "HEAD", BASE_COMMIT], cwd=project, text=True
).strip() != BASE_COMMIT:
    raise RuntimeError("Source is not based on the expected symmetric-KD base commit.")
print("[3/6] Source commit:", actual_commit)
run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
     "-p", "test_*.py", "-v"], project)
run([sys.executable, "-m", ENTRYPOINT, "--help"], project)


from huggingface_hub import hf_hub_download


downloaded_dfn = Path(
    hf_hub_download(
        repo_id=DFN_REPO,
        filename=DFN_FILENAME,
        revision=DFN_REVISION,
    )
)
dfn_target = DFN_DIR / DFN_FILENAME
shutil.copy2(downloaded_dfn, dfn_target)
actual_dfn_sha = file_sha256(dfn_target)
if actual_dfn_sha != DFN_SHA256:
    raise RuntimeError(
        f"DFN5B checksum mismatch: {actual_dfn_sha} != {DFN_SHA256}"
    )
print(
    "[4/6] DFN5B downloaded:",
    dfn_target,
    f"({dfn_target.stat().st_size / 1024**3:.3f} GiB)",
)

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
        "ViT-B/32 checksum mismatch: "
        f"{actual_student_sha} != {STUDENT_SHA256}"
    )
print(
    "[5/6] ViT-B/32 downloaded:",
    student_target,
    f"({student_target.stat().st_size / 1024**2:.1f} MiB)",
)

required_paths = (
    BUNDLE / "requirements.txt",
    WHEELS,
    project / ".git",
    project / "clip" / "model.py",
    project / "src" / "dataset.py",
    project / "src" / "losses.py",
    project / "src" / "model.py",
    project / "src" / "teacher_prompts.py",
    project / "src" / "train.py",
    project / "tests" / "test_teacher_triplet.py",
    project / "tests" / "test_symmetric_kd.py",
    dfn_target,
    student_target,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )

manifest = {
    "repository": REPO_URL,
    "branch": BRANCH,
    "commit": actual_commit,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "dataset": DATASET,
    "teacher_repo": DFN_REPO,
    "teacher_revision": DFN_REVISION,
    "teacher_filename": DFN_FILENAME,
    "teacher_sha256": DFN_SHA256,
    "teacher_size": dfn_target.stat().st_size,
    "student_filename": STUDENT_FILENAME,
    "student_sha256": STUDENT_SHA256,
    "student_size": student_target.stat().st_size,
    "python_version": sys.version,
    "python_major_minor": list(sys.version_info[:2]),
    "base_commit": BASE_COMMIT,
    "requirements_sha256": file_sha256(requirements_path),
    "runtime_requirements": roots,
    "wheels": [
        {"filename": wheel.name, "size": wheel.stat().st_size, "sha256": file_sha256(wheel)}
        for wheel in sorted(WHEELS.glob("*.whl"))
    ],
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

bundle_size = sum(
    path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file()
)
print("[6/6] Bundle validated")
print("=" * 70)
print("ONLINE BIDIRECTIONAL TEACHER TRIPLET + SYMMETRIC STUDENT KD BUNDLE COMPLETE")
print("=" * 70)
print("Bundle:", BUNDLE)
print("Branch:", BRANCH)
print("Commit:", actual_commit)
print("Entry point:", ENTRYPOINT)
print("DFN5B SHA256:", actual_dfn_sha)
print("Student SHA256:", actual_student_sha)
print("Bundle size:", f"{bundle_size / 1024**3:.3f} GiB")
print()
print("Save Version -> Save & Run All -> Always save output.")
print("After offline setup: run test/kaggle_train.ipy as the second cell.")
