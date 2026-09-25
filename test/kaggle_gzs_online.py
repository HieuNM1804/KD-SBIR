"""Build an offline Kaggle bundle for the compact fixed-subset GZS branch.

Run this cell in an online Kaggle notebook, save the output, and attach it to
the offline notebook.  The manifest pins the exact GZS source commit so that
the compact seen-class lists cannot silently drift.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys


WORKING = Path("/kaggle/working")
BUNDLE = WORKING / "offline_bundle"
REPO_URL = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/gzs-sbir"
COMMIT = "de820fc9a774353f350a8b08fced73390e8dd31b"

WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
DFN5B_DIR = BUNDLE / "dfn5b_openclip"

DFN5B_REPO = "apple/DFN5B-CLIP-ViT-H-14"
DFN5B_FILENAME = "open_clip_pytorch_model.bin"
DFN5B_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
DFN5B_SHA256 = (
    "d67de50faa7f3ddce52fbab4f4656b046"
    "86a0bb15c26ebd0144d375cfa08b8ae"
)


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
for directory in (WHEELS, SOURCE, CLIP_CACHE, DFN5B_DIR):
    directory.mkdir(parents=True, exist_ok=True)

requirements = BUNDLE / "requirements.txt"
requirements.write_text(
    "\n".join(
        (
            "open-clip-torch",
            "pytorch-lightning",
            "torchmetrics",
            "ftfy",
            "regex",
            "tensorboard",
            "huggingface-hub",
        )
    )
    + "\n",
    encoding="utf-8",
)
run([
    sys.executable,
    "-m",
    "pip",
    "download",
    "--dest",
    str(WHEELS),
    "--only-binary=:all:",
    "-r",
    str(requirements),
])

for pattern in (
    "torch-*.whl",
    "torchvision-*.whl",
    "torchaudio-*.whl",
    "triton-*.whl",
    "nvidia_*.whl",
    "cuda_*.whl",
):
    for wheel in WHEELS.glob(pattern):
        wheel.unlink()

required_wheels = (
    "open_clip_torch-*.whl",
    "pytorch_lightning-*.whl",
    "torchmetrics-*.whl",
    "ftfy-*.whl",
    "regex-*.whl",
    "tensorboard-*.whl",
    "huggingface_hub-*.whl",
)
missing = [pattern for pattern in required_wheels if not list(WHEELS.glob(pattern))]
if missing:
    raise FileNotFoundError("Missing wheels:\n" + "\n".join(missing))

run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "regex", "huggingface-hub"])

project = SOURCE / "KD-SBIR"
run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(project)])
run(["git", "checkout", "--detach", COMMIT], cwd=project)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=project, text=True
).strip()
if actual_commit != COMMIT:
    raise RuntimeError(f"Commit mismatch: expected {COMMIT}, got {actual_commit}")

for module_name in list(sys.modules):
    if module_name == "clip" or module_name.startswith("clip."):
        del sys.modules[module_name]
sys.path.insert(0, str(project))
from clip import clip as project_clip

student_weight = Path(
    project_clip._download(project_clip._MODELS["ViT-B/32"], root=str(CLIP_CACHE))
)
if not student_weight.is_file() or student_weight.stat().st_size < 300_000_000:
    raise RuntimeError("ViT-B/32 checkpoint download is incomplete")

download_environment = os.environ.copy()
for variable in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_ENABLE_HF_TRANSFER",
):
    download_environment.pop(variable, None)
download_code = (
    "from huggingface_hub import hf_hub_download; "
    f"hf_hub_download(repo_id={DFN5B_REPO!r}, filename={DFN5B_FILENAME!r}, "
    f"revision={DFN5B_REVISION!r}, local_dir={str(DFN5B_DIR)!r})"
)
run([sys.executable, "-c", download_code], env=download_environment)
dfn5b_weight = DFN5B_DIR / DFN5B_FILENAME
if not dfn5b_weight.is_file() or dfn5b_weight.stat().st_size < 3_000_000_000:
    raise RuntimeError("DFN5B checkpoint download is incomplete")
if dfn5b_weight.is_symlink():
    copied = DFN5B_DIR / "open_clip_weight_copy.tmp"
    shutil.copyfile(dfn5b_weight, copied)
    dfn5b_weight.unlink()
    copied.replace(dfn5b_weight)
if (DFN5B_DIR / ".cache").exists():
    shutil.rmtree(DFN5B_DIR / ".cache")

sha256 = hashlib.sha256()
with dfn5b_weight.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        sha256.update(block)
if sha256.hexdigest() != DFN5B_SHA256:
    raise RuntimeError("DFN5B SHA256 mismatch")

config = {}
exec((project / "src" / "data_config.py").read_text(encoding="utf-8"), config)
required_paths = (
    project / "src" / "train.py",
    project / "src" / "model.py",
    project / "src" / "dataset.py",
    project / "src" / "data_config.py",
    project / "src" / "losses.py",
    project / "src" / "teacher_prompts.py",
    CLIP_CACHE / "ViT-B-32.pt",
    dfn5b_weight,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError("Incomplete bundle:\n" + "\n".join(missing_paths))

manifest = {
    "repository": REPO_URL,
    "branch": BRANCH,
    "commit": actual_commit,
    "evaluation_protocol": "gzs",
    "generalized_classes": config["GENERALIZED_CLASSES"],
    "student_filename": "ViT-B-32.pt",
    "student_size": student_weight.stat().st_size,
    "dfn5b_repo": DFN5B_REPO,
    "dfn5b_filename": DFN5B_FILENAME,
    "dfn5b_revision": DFN5B_REVISION,
    "dfn5b_sha256": sha256.hexdigest(),
    "dfn5b_size": dfn5b_weight.stat().st_size,
    "python_version": sys.version,
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
bundle_size = sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
print("GZS offline bundle complete")
print("Branch:", BRANCH)
print("Commit:", actual_commit)
print("Generalized classes:", {key: len(value) for key, value in config["GENERALIZED_CLASSES"].items()})
print("Bundle:", BUNDLE)
print("Bundle size: %.3f GiB" % (bundle_size / 1024**3))
