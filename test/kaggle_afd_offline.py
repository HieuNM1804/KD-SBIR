"""Restore and validate the pinned dual-axis AFD project on offline Kaggle."""

import glob
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/clip-kd-dual-axis-afd"
EXPECTED_COMMIT = "__PINNED_AFD_SOURCE_COMMIT__"
BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
TASK = "clip_kd_dual_axis_afd"
WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR-AVKD"
SKETCHY_ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path, size, checksum):
    if not path.is_file() or path.stat().st_size != size or sha256(path) != checksum:
        raise RuntimeError(f"Offline bundle checksum validation failed: {path}")


def safe_filename(value):
    if not isinstance(value, str) or Path(value).name != value or value in {"", ".", ".."}:
        raise RuntimeError(f"Unsafe filename in manifest: {value!r}")
    return value


def find_bundle():
    paths = []
    for pattern in (
        "/kaggle/input/*/offline_bundle/bundle_manifest.json",
        "/kaggle/input/*/*/offline_bundle/bundle_manifest.json",
        "/kaggle/input/*/*/*/offline_bundle/bundle_manifest.json",
        "/kaggle/input/*/*/*/*/offline_bundle/bundle_manifest.json",
        "/kaggle/working/afd_bundle_*/offline_bundle/bundle_manifest.json",
    ):
        paths.extend(Path(path) for path in glob.glob(pattern))
    matches = []
    reports = []
    for path in sorted(set(paths)):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            reports.append(f"{path}: unreadable ({error})")
            continue
        reports.append(
            f"{path}: task={manifest.get('task')}, branch={manifest.get('branch')}, "
            f"commit={manifest.get('commit')}"
        )
        if (
            manifest.get("repository") == REPOSITORY
            and manifest.get("branch") == BRANCH
            and manifest.get("commit") == EXPECTED_COMMIT
            and manifest.get("base_commit") == BASE_COMMIT
            and manifest.get("task") == TASK
            and manifest.get("entrypoint") == "test/kaggle_afd_sweep.py"
            and manifest.get("dataset") == "b20dccn616nguynhutun/sketchy"
        ):
            matches.append((path.parent, manifest))
    input_matches = [entry for entry in matches if str(entry[0]).startswith("/kaggle/input/")]
    if input_matches:
        matches = input_matches
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one AFD bundle for {EXPECTED_COMMIT}, found {len(matches)}.\n"
            + ("\n".join(reports) if reports else "No bundle manifests found.")
        )
    return matches[0]


def preserve_if_different(path, checksum):
    if not path.exists() or (path.is_file() and sha256(path) == checksum):
        return
    backup = path.with_name(path.name + ".previous")
    counter = 1
    while backup.exists():
        backup = path.with_name(path.name + f".previous_{counter}")
        counter += 1
    path.rename(backup)
    print("[Offline] Preserved differing cache file:", backup)


bundle, manifest = find_bundle()
if manifest["python_major_minor"] != list(sys.version_info[:2]):
    raise RuntimeError("Python version differs from the bundle-building Kaggle session.")
wheels = bundle / "wheels"
source = bundle / "source" / "KD-SBIR"
lock = bundle / "requirements.lock.txt"
teacher_file = bundle / "dfn5b_openclip" / safe_filename(manifest["teacher_filename"])
student_file = bundle / "clip_cache" / safe_filename(manifest["student_filename"])
required = (
    wheels,
    lock,
    source / ".git",
    source / "src" / "afd.py",
    source / "src" / "losses.py",
    source / "src" / "model.py",
    source / "src" / "train.py",
    source / "test" / "kaggle_afd_sweep.py",
    source / "tests" / "test_afd.py",
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("AFD bundle is incomplete:\n" + "\n".join(missing))
validate_file(teacher_file, manifest["teacher_size"], manifest["teacher_sha256"])
validate_file(student_file, manifest["student_size"], manifest["student_sha256"])
if not manifest.get("wheels"):
    raise RuntimeError("AFD bundle contains no wheel manifest.")
for wheel in manifest["wheels"]:
    validate_file(
        wheels / safe_filename(wheel["filename"]), wheel["size"], wheel["sha256"]
    )

if WORKING_PROJECT.exists():
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT,
        text=True, capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=WORKING_PROJECT,
        text=True, capture_output=True, check=False,
    )
    if (
        current.returncode != 0
        or current.stdout.strip() != EXPECTED_COMMIT
        or status.returncode != 0
        or status.stdout.strip()
    ):
        backup = WORKING_ROOT / (WORKING_PROJECT.name + "_previous")
        counter = 1
        while backup.exists():
            backup = WORKING_ROOT / f"{WORKING_PROJECT.name}_previous_{counter}"
            counter += 1
        WORKING_PROJECT.rename(backup)
        print("[Offline] Previous project preserved:", backup)
if not WORKING_PROJECT.exists():
    shutil.copytree(source, WORKING_PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
merge_base = subprocess.check_output(
    ["git", "merge-base", "HEAD", BASE_COMMIT], cwd=WORKING_PROJECT, text=True
).strip()
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != EXPECTED_COMMIT or merge_base != BASE_COMMIT or dirty:
    raise RuntimeError("Restored AFD source failed commit/base/clean-tree validation.")

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(wheels), "-r", str(lock),
    ],
    cwd=WORKING_ROOT,
    check=True,
)
for package in ("torch", "torchvision"):
    print(f"[Offline] {package} from GPU runtime: {importlib.metadata.version(package)}")

hf_home = WORKING_ROOT / "huggingface_afd"
hf_hub = hf_home / "hub"
os.environ.update(
    HF_HOME=str(hf_home),
    HF_HUB_CACHE=str(hf_hub),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / student_file.name
preserve_if_different(student_target, manifest["student_sha256"])
if not student_target.exists():
    shutil.copy2(student_file, student_target)

repository_cache = hf_hub / (
    "models--" + manifest["teacher_repository"].replace("/", "--")
)
snapshot = repository_cache / "snapshots" / manifest["teacher_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
teacher_target = snapshot / teacher_file.name
preserve_if_different(teacher_target, manifest["teacher_sha256"])
if not teacher_target.exists():
    shutil.copy2(teacher_file, teacher_target)
refs = repository_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(manifest["teacher_revision"], encoding="utf-8")

sys.path.insert(0, str(WORKING_PROJECT))
import torch
from clip import clip
from open_clip.pretrained import download_pretrained_from_hf

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator.")
if Path(clip.download_model("ViT-B/32")).resolve() != student_target.resolve():
    raise RuntimeError("CLIP did not resolve to the bundled student checkpoint.")
resolved_teacher = Path(download_pretrained_from_hf(
    manifest["teacher_repository"], revision=manifest["teacher_revision"]
))
if resolved_teacher.resolve() != teacher_target.resolve():
    raise RuntimeError("OpenCLIP did not resolve to the bundled teacher checkpoint.")
for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing Sketchy directory: {directory}")
subprocess.run(
    [
        sys.executable, "-m", "unittest", "discover", "-s", "tests",
        "-p", "test_afd.py", "-v",
    ],
    cwd=WORKING_PROJECT,
    env=os.environ.copy(),
    check=True,
)
print("=" * 72)
print("OFFLINE ISOLATED AFD SETUP COMPLETE")
print("Source commit:", actual_commit)
print("Main student losses: disabled and rejected when nonzero")
print("Run:")
print("!python -u test/kaggle_afd_sweep.py --dataset sketchy_1")
print("!python -u test/kaggle_afd_sweep.py --dataset sketchy_2")
