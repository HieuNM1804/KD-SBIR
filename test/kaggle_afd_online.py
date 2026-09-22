"""Build the pinned dual-axis AFD bundle in an Internet-enabled Kaggle notebook."""

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


WORKING = Path("/kaggle/working")
BUNDLE = WORKING / (
    "afd_bundle_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
) / "offline_bundle"
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
DFN_DIR = BUNDLE / "dfn5b_openclip"

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/clip-kd-dual-axis-afd"
SOURCE_COMMIT = "__PINNED_AFD_SOURCE_COMMIT__"
BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
TASK = "clip_kd_dual_axis_afd"
ENTRYPOINT = "test/kaggle_afd_sweep.py"
DATASET = "b20dccn616nguynhutun/sketchy"

DFN_REPOSITORY = "apple/DFN5B-CLIP-ViT-H-14"
DFN_FILENAME = "open_clip_pytorch_model.bin"
DFN_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
DFN_SHA256 = "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae"
STUDENT_FILENAME = "ViT-B-32.pt"
STUDENT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for directory in (WHEELS, SOURCE, CLIP_CACHE, DFN_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/6] Empty AFD bundle:", BUNDLE)

requirements = """open-clip-torch
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
lock = BUNDLE / "requirements.lock.txt"
lock.write_text(requirements, encoding="utf-8")
run(
    [
        sys.executable, "-m", "pip", "download", "--dest", str(WHEELS),
        "--only-binary=:all:", "-r", str(lock),
    ],
    WORKING,
)
for pattern in (
    "torch-*.whl", "torchvision-*.whl", "torchaudio-*.whl", "triton-*.whl",
    "nvidia_*.whl", "cuda_*.whl",
):
    for wheel in WHEELS.glob(pattern):
        wheel.unlink()
print("[2/6] Dependency wheels:", len(list(WHEELS.glob("*.whl"))))

project = SOURCE / "KD-SBIR"
run(
    [
        "git", "clone", "--branch", BRANCH, "--single-branch", REPOSITORY,
        str(project),
    ],
    WORKING,
)
run(["git", "checkout", "--detach", SOURCE_COMMIT], project)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=project, text=True
).strip()
merge_base = subprocess.check_output(
    ["git", "merge-base", "HEAD", BASE_COMMIT], cwd=project, text=True
).strip()
if actual_commit != SOURCE_COMMIT or merge_base != BASE_COMMIT:
    raise RuntimeError("The AFD source is not the pinned main-based commit.")
required_source = (
    project / "src" / "afd.py",
    project / "src" / "losses.py",
    project / "src" / "model.py",
    project / "src" / "train.py",
    project / "test" / "kaggle_afd_sweep.py",
    project / "tests" / "test_afd.py",
)
missing = [str(path) for path in required_source if not path.exists()]
if missing:
    raise FileNotFoundError("Pinned AFD source is incomplete:\n" + "\n".join(missing))
print("[3/6] Pinned main-based source:", actual_commit)

run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface-hub", "ftfy", "regex"],
    WORKING,
)
from huggingface_hub import hf_hub_download

downloaded_teacher = Path(hf_hub_download(
    repo_id=DFN_REPOSITORY,
    filename=DFN_FILENAME,
    revision=DFN_REVISION,
))
teacher_target = DFN_DIR / DFN_FILENAME
shutil.copy2(downloaded_teacher, teacher_target)
if sha256(teacher_target) != DFN_SHA256:
    raise RuntimeError("DFN5B checksum mismatch.")
print("[4/6] DFN5B checkpoint:", teacher_target)

for module_name in list(sys.modules):
    if module_name == "clip" or module_name.startswith("clip."):
        del sys.modules[module_name]
sys.path.insert(0, str(project))
from clip import clip as project_clip

downloaded_student = Path(project_clip.download_model("ViT-B/32"))
student_target = CLIP_CACHE / STUDENT_FILENAME
shutil.copy2(downloaded_student, student_target)
if sha256(student_target) != STUDENT_SHA256:
    raise RuntimeError("ViT-B/32 checksum mismatch.")
print("[5/6] Student checkpoint:", student_target)

wheels = []
for wheel in sorted(WHEELS.glob("*.whl")):
    wheels.append(
        {"filename": wheel.name, "size": wheel.stat().st_size, "sha256": sha256(wheel)}
    )
manifest = {
    "repository": REPOSITORY,
    "branch": BRANCH,
    "commit": actual_commit,
    "base_commit": BASE_COMMIT,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "dataset": DATASET,
    "python_major_minor": list(sys.version_info[:2]),
    "teacher_repository": DFN_REPOSITORY,
    "teacher_revision": DFN_REVISION,
    "teacher_filename": DFN_FILENAME,
    "teacher_sha256": DFN_SHA256,
    "teacher_size": teacher_target.stat().st_size,
    "student_filename": STUDENT_FILENAME,
    "student_sha256": STUDENT_SHA256,
    "student_size": student_target.stat().st_size,
    "wheels": wheels,
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
size = sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
print("[6/6] OFFLINE BUNDLE READY")
print("Bundle:", BUNDLE)
print("Source commit:", actual_commit)
print("Bundle size:", f"{size / 1024**3:.3f} GiB")
print("Use Save Version -> Save & Run All -> Always save output.")
