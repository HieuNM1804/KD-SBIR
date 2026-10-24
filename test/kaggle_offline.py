"""ONE OFFLINE Kaggle setup cell: attach bundle, install wheels and link checkpoints."""

from pathlib import Path
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

TASK = "dfn5b224_pseudo_label_retrieval"
BRANCH = "experiment/dfn5b-pseudo-label-retrieval"
SOURCE_COMMIT = "2ed12a595b429dec2c0069488261784df3945c37"
BUNDLE_PATH = None  # Exact offline_bundle directory if several outputs are attached.
VERIFY_WEIGHT_HASHES = False  # Online verified them; True rereads ~4.3 GiB offline.
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR"


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
    raise RuntimeError(f"Found {len(matches)} matching bundles; attach this branch's online output or set BUNDLE_PATH.")
bundle, metadata = matches[0]
print("Bundle:", bundle, flush=True)
if metadata["python"] != list(sys.version_info[:2]):
    print("Online/offline Python differ; pip will select compatible wheels for the current interpreter.", flush=True)
for item in metadata["files"]:
    relative = Path(item["path"])
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Invalid/missing bundle path: {relative}")
    if path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Incomplete bundle file: {relative}")
    if (relative.parts[0] != "models" or VERIFY_WEIGHT_HASHES) and sha256(path) != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {relative}")
print("Installing offline dependencies; keeping Kaggle's PyTorch/CUDA stack.", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
                str(bundle / "wheels"), *metadata["runtime_requirements"]], check=True)
WORKING.mkdir(exist_ok=True, parents=True)
if PROJECT.exists():
    backup = PROJECT.with_name(PROJECT.name + "_previous_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    PROJECT.rename(backup)
    print("Previous project preserved:", backup, flush=True)
shutil.copytree(bundle / "KD-SBIR", PROJECT, ignore=shutil.ignore_patterns("__pycache__"))
(PROJECT / "models").symlink_to((bundle / "models").resolve(), target_is_directory=True)
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
    KD_SBIR_DFN5B_WEIGHTS=str(PROJECT / "models" / "dfn5b224" / metadata["teacher"]["filename"]),
    KD_SBIR_STUDENT_WEIGHTS=str(PROJECT / "models" / metadata["student"]["filename"]),
)
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "src.train", "--help"], check=True, stdout=subprocess.DEVNULL)
print("SETUP COMPLETE:", PROJECT, flush=True)
print("Run the %cd + !python cell in kaggle_train.ipy. It generates labels, tunes teacher prompts, then distills the student.")
