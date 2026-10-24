"""ONE OFFLINE Kaggle setup cell: attach the bundle, install wheels, link weights."""

from pathlib import Path
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

TASK = "teacher_zero_shot_classification"
BRANCH = "experiment/teacher-zero-shot-classification"
SOURCE_COMMIT = "c62ea560cfac2724ba01b21ce4497422d58994f0"
BUNDLE_PATH = None  # Set an exact offline_bundle path only if several are attached.
VERIFY_WEIGHT_HASHES = False  # Online already checked all weights; True rereads ~12 GiB.
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR-teacher-classification"


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
for path in manifests:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (metadata.get("task") == TASK and metadata.get("branch") == BRANCH
            and metadata.get("commit") == SOURCE_COMMIT):
        matches.append((path.parent, metadata))
if len(matches) != 1:
    raise RuntimeError(f"Found {len(matches)} matching bundles; attach the new online output or set BUNDLE_PATH.")
bundle, metadata = matches[0]
print("Bundle:", bundle, flush=True)
if metadata["python"] != list(sys.version_info[:2]):
    print("Online/offline Python differ; using compatible wheels for the current interpreter.", flush=True)
for item in metadata["files"]:
    relative = Path(item["path"])
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Invalid/missing bundle path: {relative}")
    if path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Incomplete bundle file: {relative}")
    if (relative.parts[0] != "models" or VERIFY_WEIGHT_HASHES) and sha256(path) != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {relative}")
print("Installing offline runtime dependencies (keeping Kaggle's installed PyTorch).", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
                str(bundle / "wheels"), *metadata["runtime_requirements"]], check=True)
WORKING.mkdir(exist_ok=True, parents=True)
if PROJECT.exists():
    backup = PROJECT.with_name(PROJECT.name + "_previous_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    PROJECT.rename(backup)
    print("Previous project preserved:", backup, flush=True)
shutil.copytree(bundle / "KD-SBIR", PROJECT, ignore=shutil.ignore_patterns("__pycache__"))
# Weights stay on attached input; do not duplicate them in working storage.
(PROJECT / "models").symlink_to((bundle / "models").resolve(), target_is_directory=True)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "src.evaluate_teacher_classification", "--help"], check=True,
                stdout=subprocess.DEVNULL)
print("SETUP COMPLETE:", PROJECT, flush=True)
print("Run the %cd + !python cell next. Inference only; no fine-tuning or prompt training.")
