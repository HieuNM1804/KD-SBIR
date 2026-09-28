"""Validate the five-model bundle and run raw TinyCLIP inference on Kaggle.

Paste/run this after attaching the online notebook output and Sketchy dataset.
It runs no training, optimizer, prompt learning, or teacher model.
"""

from pathlib import Path, PurePosixPath
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/inference-tinyclip-suite"
EXPECTED_COMMIT = "1b091d760019a29aefed6fbb524d275876e12d1f"
EXPECTED_BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
EXPECTED_TASK = "tinyclip_raw_inference_suite"
EXPECTED_ENTRYPOINT = "src.infer_tinyclip_suite"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"
EXPECTED_MODELS = ("8m", "22m", "40m", "45m", "61m")

WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR-tinyclip-inference"
SKETCHY_ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
RESULTS = WORKING / "tinyclip_inference_results_sketchy2"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value):
    if not isinstance(value, str) or Path(value).name != value or value in {"", ".", ".."}:
        raise RuntimeError(f"Unsafe manifest filename: {value!r}")
    return value


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe relative path in manifest: {value!r}")
    return Path(*path.parts)


def validate(path, size, checksum):
    if not path.is_file() or path.stat().st_size != size or sha256(path) != checksum:
        raise RuntimeError(f"Offline bundle checksum validation failed: {path}")


def next_backup(path):
    backup = path.with_name(path.name + "_previous")
    counter = 1
    while backup.exists():
        backup = path.with_name(path.name + f"_previous_{counter}")
        counter += 1
    return backup


manifest_paths = []
for pattern in (
    "/kaggle/input/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/working/offline_bundle/bundle_manifest.json",
):
    manifest_paths.extend(Path(path) for path in glob.glob(pattern))

matches = []
reports = []
for path in sorted(set(manifest_paths)):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reports.append(f"{path}: unreadable ({error})")
        continue
    model_keys = tuple(model.get("key") for model in manifest.get("models", ()))
    reports.append(
        f"{path}: branch={manifest.get('branch')}, commit={manifest.get('commit')}, "
        f"task={manifest.get('task')}, models={model_keys}"
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and manifest.get("commit") == EXPECTED_COMMIT
        and manifest.get("base_commit") == EXPECTED_BASE_COMMIT
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("dataset") == EXPECTED_DATASET
        and model_keys == EXPECTED_MODELS
    ):
        matches.append((path.parent, manifest))
input_matches = [entry for entry in matches if str(entry[0]).startswith("/kaggle/input/")]
if input_matches:
    matches = input_matches
if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one five-model TinyCLIP bundle, found {len(matches)}.\n"
        + ("\n".join(reports) if reports else "No manifests found.")
    )

bundle, manifest = matches[0]
if manifest["python_major_minor"] != list(sys.version_info[:2]):
    raise RuntimeError("Python version differs from the online builder session.")
wheels = bundle / "wheels"
source = bundle / "source" / "KD-SBIR"
models_root = bundle / "tinyclip_models"
required = (
    bundle / "requirements.txt",
    wheels,
    source / ".git",
    source / "src" / "infer_tinyclip_suite.py",
    source / "src" / "tinyclip_inference.py",
    source / "src" / "tinyclip_vendor" / "model.py",
    source / "src" / "tinyclip_vendor" / "LICENSE.upstream",
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Inference bundle is incomplete:\n" + "\n".join(missing))
if not manifest.get("wheels") or len(manifest.get("models", ())) != 5:
    raise RuntimeError("Bundle has an empty wheel manifest or does not contain five models.")
for item in manifest["wheels"]:
    validate(wheels / safe_filename(item["filename"]), item["size"], item["sha256"])
for model in manifest["models"]:
    if not model.get("files"):
        raise RuntimeError(f"Model has an empty file manifest: {model.get('key')}")
    for item in model["files"]:
        validate(
            models_root / safe_relative(item["relative_path"]),
            item["size"],
            item["sha256"],
        )
print("Selected bundle:", bundle)
print("Validated models:", ", ".join(EXPECTED_MODELS))

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(wheels), "-r", str(bundle / "requirements.txt"),
    ],
    cwd=WORKING,
    check=True,
)

if PROJECT.exists():
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT,
        text=True, capture_output=True, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT,
        text=True, capture_output=True, check=False,
    )
    if current.returncode != 0 or current.stdout.strip() != EXPECTED_COMMIT or dirty.stdout.strip():
        backup = next_backup(PROJECT)
        PROJECT.rename(backup)
        print("Previous inference project preserved:", backup)
if not PROJECT.exists():
    shutil.copytree(source, PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
merge_base = subprocess.check_output(
    ["git", "merge-base", "HEAD", EXPECTED_BASE_COMMIT], cwd=PROJECT, text=True
).strip()
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=PROJECT, text=True
).strip()
if actual_commit != EXPECTED_COMMIT or merge_base != EXPECTED_BASE_COMMIT or dirty:
    raise RuntimeError("Restored source failed commit/base/clean-tree validation.")

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing Sketchy directory: {directory}")

offline_env = os.environ.copy()
offline_env.update(
    HF_HOME=str(WORKING / "huggingface_tinyclip_inference"),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    HF_HUB_DISABLE_XET="1",
)
subprocess.run(
    [
        sys.executable, "-m", EXPECTED_ENTRYPOINT,
        "--root", str(SKETCHY_ROOT),
        "--dataset", "sketchy_2",
        "--scope", "unseen",
        "--models-root", str(models_root),
        "--batch-size", "256",
        "--workers", "4",
        "--precision", "auto",
        "--output-dir", str(RESULTS),
    ],
    cwd=PROJECT,
    env=offline_env,
    check=True,
)

print("=" * 72)
print("RAW TINYCLIP 8M/22M/40M/45M/61M SKETCHY-2 INFERENCE COMPLETE")
print("=" * 72)
print("Project:", PROJECT)
print("Commit:", actual_commit)
print("Dataset:", SKETCHY_ROOT)
print("Results JSON:", RESULTS / "tinyclip_inference_results.json")
print("Results CSV:", RESULTS / "tinyclip_inference_results.csv")
