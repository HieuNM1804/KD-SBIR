"""ONE OFFLINE Kaggle cell: install compatible wheels and link bundled weights."""
from datetime import datetime, timezone
from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

TASK = "prompted_encoder_cost"
BRANCH = "experiment/prompted-encoder-cost"
SOURCE_COMMIT = "92ff04eb17be9e79e103b20773bd6538d18b5c3e"
BUNDLE_PATH = None  # Set exact offline_bundle folder only if several match.
VERIFY_WEIGHT_HASHES = False  # Online checked hashes; True rereads all model bytes.
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR-encoder-cost"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifests = [Path(BUNDLE_PATH) / "bundle_manifest.json"] if BUNDLE_PATH else sorted({
    Path(path) for depth in range(1, 6)
    for path in glob.glob("/kaggle/input/" + "*/" * depth + "offline_bundle/bundle_manifest.json")
})
matches = []
for manifest in manifests:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (metadata.get("task") == TASK and metadata.get("branch") == BRANCH
            and metadata.get("commit") == SOURCE_COMMIT):
        matches.append((manifest.parent, metadata))
if len(matches) != 1:
    raise RuntimeError(f"Found {len(matches)} matching bundles; attach online output or set BUNDLE_PATH.")
bundle, metadata = matches[0]
print("Bundle:", bundle, flush=True)
if metadata["python"] != list(sys.version_info[:2]):
    print("Online/offline Python differ; pip will select compatible bundled wheels.", flush=True)
for item in metadata["files"]:
    relative = Path(item["path"])
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Invalid/missing bundle path: {relative}")
    if path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Incomplete bundle file: {relative}")
    if (relative.parts[0] != "models" or VERIFY_WEIGHT_HASHES) and sha256(path) != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {relative}")
print("Installing offline dependencies; keeping Kaggle's PyTorch/CUDA.", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
                str(bundle / "wheels"), *metadata["runtime_requirements"]], check=True)
WORKING.mkdir(parents=True, exist_ok=True)
if PROJECT.exists():
    previous = PROJECT.with_name(PROJECT.name + "_previous_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    PROJECT.rename(previous)
    print("Previous project preserved:", previous, flush=True)
shutil.copytree(bundle / "KD-SBIR", PROJECT, ignore=shutil.ignore_patterns("__pycache__"))
(PROJECT / "models").symlink_to((bundle / "models").resolve(), target_is_directory=True)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "src.benchmark_encoder_cost", "--help"], check=True,
                stdout=subprocess.DEVNULL)
print("SETUP COMPLETE:", PROJECT, flush=True)
print("Paste kaggle_encoder_cost_run.ipy next. No datasets needed.")
