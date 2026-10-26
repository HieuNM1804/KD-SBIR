"""ONE ONLINE Kaggle cell: inference + training cost source, weights and wheels."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/prompted-encoder-cost"
SOURCE_COMMIT = "9cb3d934edd794e4985aebc118d8807b7fde57e7"
TASK = "prompted_encoder_cost"
WORKING = Path("/kaggle/working")
BUNDLE = WORKING / ("encoder_cost_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")) / "offline_bundle"
PROJECT = BUNDLE / "KD-SBIR"
MODELS = BUNDLE / "models"
WHEELS = BUNDLE / "wheels"
TARGET_PYTHONS = ("3.12", "3.13")
ROOTS = ["open-clip-torch==3.2.0", "transformers==4.57.6", "huggingface-hub<1.0",
         "safetensors", "ftfy", "regex", "numpy", "pillow", "tqdm", "packaging",
         "pytorch-lightning>=2.5,<3", "torchmetrics>=1.6,<2", "lightning-utilities", "tensorboard"]
# Optional reuse; only these three files/dirs are copied, never a whole dataset.
# dfn5b_h14 and clip_b32: exact checkpoint file. tinyclip_61m: directory with
# config.json and model.safetensors from the original TinyCLIP-61M snapshot.
REUSE_WEIGHTS = {
    # "dfn5b_h14": "/kaggle/input/.../dfn5b_openclip/open_clip_pytorch_model.bin",
    # "clip_b32": "/kaggle/input/.../clip_cache/ViT-B-32.pt",
    # "tinyclip_61m": "/kaggle/input/.../models/tinyclip61m_student",
}
AUTO_REUSE_WORKING_WEIGHTS = True  # Reuse verified files from previous/failed encoder_cost_* bundles.


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
print("[1/5] New bundle:", BUNDLE, flush=True)
run(["git", "clone", "--single-branch", "--branch", BRANCH, REPOSITORY, str(PROJECT)])
run(["git", "checkout", "--detach", SOURCE_COMMIT], PROJECT)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
if actual != SOURCE_COMMIT:
    raise RuntimeError(f"Source commit mismatch: {actual}")
sys.path.insert(0, str(PROJECT))
# A previous failed cell may have imported src from an older cloned source.
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]
from src.encoder_cost_models import SPECS
from src.kaggle_cost_wheels import download_runtime_wheels
from src.kaggle_cost_downloads import download_verified, matches, reuse_verified

# Resolve separately for each ABI; do not force the online NumPy version on 3.13.
# PyTorch/CUDA are provided by Kaggle and are not downloaded into this bundle.
run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub<1.0", "packaging"])
inventories = download_runtime_wheels(ROOTS, TARGET_PYTHONS, BUNDLE, WHEELS, run)
print("[2/5] Non-CUDA wheels ready:", len(list(WHEELS.glob("*.whl"))), flush=True)

from huggingface_hub import HfApi, hf_hub_download

for key, spec in SPECS.items():
    directory = MODELS / spec["directory"]
    directory.mkdir(exist_ok=True)
    names = ("config.json", "model.safetensors") if key == "tinyclip_61m" else (spec["filename"],)
    # Pinned HF metadata also validates reused TinyCLIP weights against the LFS hash.
    hf_files = {}
    if key == "tinyclip_61m":
        info = HfApi().model_info(spec["repository"], revision=spec["revision"], files_metadata=True)
        hf_files = {item.rfilename: item for item in info.siblings}
    for name in names:
        target = directory / name
        expected = spec.get("sha256")
        blob_id = None
        if key == "tinyclip_61m":
            item = hf_files[name]
            if item.lfs:
                expected = item.lfs.sha256
            else:
                blob_id = item.blob_id
        candidates = []
        if key in REUSE_WEIGHTS:
            source = Path(REUSE_WEIGHTS[key])
            if key == "tinyclip_61m":
                source = source / name
            if not matches(source, expected, blob_id):
                raise RuntimeError(f"REUSE_WEIGHTS is missing or has the wrong checksum: {source}")
            candidates.append(source)
        if AUTO_REUSE_WORKING_WEIGHTS:
            candidates.extend(sorted(WORKING.glob(
                f"encoder_cost_*/offline_bundle/models/{spec['directory']}/{name}"),
                key=lambda path: path.stat().st_mtime, reverse=True))
            if key == "clip_b32":
                candidates.append(Path.home() / ".cache/clip/ViT-B-32.pt")
        if reuse_verified(target, candidates, expected, blob_id):
            continue
        if spec.get("url"):
            download_verified([spec["url"], *spec.get("fallback_urls", [])], target, expected)
        else:
            hf_hub_download(repo_id=spec["repository"], filename=name,
                            revision=spec["revision"], local_dir=directory)
        if not matches(target, expected, blob_id):
            raise RuntimeError(f"Checkpoint checksum mismatch: {key}/{name}")
    print(f"[3/5] Ready: {spec['name']}", flush=True)

files = []
for directory in (PROJECT, MODELS, WHEELS):
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(BUNDLE)
        if not path.is_file() or any(part in {".git", "__pycache__", ".cache"} for part in relative.parts):
            continue
        files.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
for path in sorted(BUNDLE.glob("requirements*.txt")):
    files.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha256(path)))
manifest = dict(task=TASK, repository=REPOSITORY, branch=BRANCH, commit=SOURCE_COMMIT,
                python=list(sys.version_info[:2]), target_pythons=list(TARGET_PYTHONS),
                runtime_requirements=ROOTS, wheel_inventories=inventories,
                model_specs=SPECS, files=files)
(BUNDLE / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("[4/5] Three original backbones bundled; inference + main training runtime ready.", flush=True)
print("[5/5] COMPLETE:", BUNDLE, flush=True)
print("Save Version -> Save & Run All -> Always save output.")
print("Attach output to an offline GPU notebook; paste offline setup then choose inference or training cell.")
