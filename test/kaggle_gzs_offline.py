"""Install and validate the offline bundle for experiment/gzs-sbir."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_COMMIT = "3a2a35785de40186064bcec28287622b7d72d4b8"
WORKING_PROJECT = Path("/kaggle/working/KD-SBIR")
SKETCHY_ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


manifest_paths = []
for pattern in (
    "/kaggle/input/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/offline_bundle/bundle_manifest.json",
):
    manifest_paths.extend(Path(path) for path in glob.glob(pattern))

matching = []
for manifest_path in sorted(set(manifest_paths)):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if manifest.get("commit") == EXPECTED_COMMIT:
        matching.append((manifest_path, manifest))
if not matching:
    raise FileNotFoundError(
        "No offline bundle with the GZS commit was found. Attach the output "
        "of kaggle_gzs_online.py to this notebook."
    )

manifest_path, manifest = matching[0]
bundle = manifest_path.parent
wheels = bundle / "wheels"
source = bundle / "source"
clip_cache = bundle / "clip_cache"
dfn5b_dir = bundle / "dfn5b_openclip"
print("Offline bundle:", bundle)
print("Branch:", manifest.get("branch"))
print("Commit:", manifest.get("commit"))
print("Generalized classes:", {key: len(value) for key, value in manifest["generalized_classes"].items()})

run([
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
    "ftfy",
    "regex",
    "tensorboard",
    "huggingface-hub",
])

hf_home = Path("/kaggle/working/huggingface")
os.environ.update(
    {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_XET": "1",
    }
)
dfn5b_sha256 = manifest["dfn5b_sha256"]
dfn_candidates = list(dfn5b_dir.rglob("open_clip_pytorch_model.bin"))
if not dfn_candidates:
    raise FileNotFoundError("DFN5B weight is missing from the bundle")
dfn_source = dfn_candidates[0]
repo_cache = hf_home / "hub" / "models--apple--DFN5B-CLIP-ViT-H-14"
blob = repo_cache / "blobs" / dfn5b_sha256
snapshot = repo_cache / "snapshots" / manifest["dfn5b_revision"]
blob.parent.mkdir(parents=True, exist_ok=True)
snapshot.mkdir(parents=True, exist_ok=True)
(repo_cache / "refs").mkdir(parents=True, exist_ok=True)
if not blob.exists():
    shutil.copy2(dfn_source, blob)
snapshot_weight = snapshot / "open_clip_pytorch_model.bin"
if snapshot_weight.exists() or snapshot_weight.is_symlink():
    snapshot_weight.unlink()
try:
    snapshot_weight.symlink_to(os.path.relpath(blob, snapshot))
except OSError:
    shutil.copy2(blob, snapshot_weight)
(repo_cache / "refs" / "main").write_text(manifest["dfn5b_revision"])
(repo_cache / ".no_exist" / manifest["dfn5b_revision"]).mkdir(parents=True, exist_ok=True)
(repo_cache / ".no_exist" / manifest["dfn5b_revision"] / "open_clip_model.safetensors").touch()

student_candidates = list(clip_cache.rglob("ViT-B-32.pt"))
if not student_candidates:
    raise FileNotFoundError("ViT-B-32.pt is missing from the bundle")
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / "ViT-B-32.pt"
if not student_target.exists():
    shutil.copy2(student_candidates[0], student_target)

source_root = source
if not (source_root / "src").is_dir():
    candidates = [path for path in source_root.iterdir() if (path / "src").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify the repository inside source/")
    source_root = candidates[0]
if WORKING_PROJECT.exists():
    shutil.rmtree(WORKING_PROJECT)
shutil.copytree(source_root, WORKING_PROJECT, symlinks=False)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != EXPECTED_COMMIT:
    raise RuntimeError(f"Repository commit mismatch: {actual_commit}")

for path in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not path.is_dir():
        raise FileNotFoundError(path)

run([
    sys.executable,
    "-c",
    "import torch, open_clip, pytorch_lightning, torchmetrics; "
    "print('PyTorch:', torch.__version__); "
    "print('CUDA available:', torch.cuda.is_available())",
], cwd=WORKING_PROJECT, env=os.environ.copy())

print("GZS offline setup complete")
print("Project:", WORKING_PROJECT)
print("Dataset:", SKETCHY_ROOT)
print("Commit:", actual_commit)
print("Run training with --eval_protocol gzs next.")
