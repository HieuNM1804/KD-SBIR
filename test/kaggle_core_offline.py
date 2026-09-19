"""Restore CoRe-KD in an offline Kaggle GPU notebook."""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/core-cross-modal-margin-kd"
EXPECTED_COMMIT = "5e12c0d28c5e80dddd40f24671cb9087a51c7035"
EXPECTED_TASK = "core_cross_modal_margin_distillation"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR-AVKD"
SKETCHY_ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
CORE_CACHE_NAME = "sketchy2_core_teacher2_v7.pt"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_ROOT)
manifest_paths = []
for pattern in (
    "/kaggle/input/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/offline_bundle/bundle_manifest.json",
    "/kaggle/input/*/*/*/*/offline_bundle/bundle_manifest.json",
):
    manifest_paths.extend(Path(path) for path in glob.glob(pattern))
manifest_paths = sorted(set(manifest_paths))
print("Manifest files found:", len(manifest_paths))

matching = []
reports = []
for manifest_path in manifest_paths:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        reports.append(f"- {manifest_path}: {type(error).__name__}: {error}")
        continue
    reports.append(
        f"- {manifest_path}\n"
        f"  branch: {manifest.get('branch')}\n"
        f"  commit: {manifest.get('commit')}\n"
        f"  task: {manifest.get('task')}"
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and manifest.get("commit") == EXPECTED_COMMIT
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("dataset") == EXPECTED_DATASET
    ):
        matching.append((manifest_path, manifest))

if len(matching) != 1:
    raise RuntimeError(
        f"Expected exactly one CoRe-KD bundle, found {len(matching)}.\n"
        + "\n".join(reports)
    )

manifest_path, manifest = matching[0]
bundle = manifest_path.parent
wheels = bundle / "wheels"
source_project = bundle / "source" / "KD-SBIR"
dfn_source = bundle / "dfn5b_openclip" / manifest["teacher_filename"]
student_source = bundle / "clip_cache" / manifest["student_filename"]
print("Selected bundle:", bundle)

required = (
    bundle / "requirements.txt",
    wheels,
    source_project / ".git",
    source_project / "src" / "losses.py",
    source_project / "src" / "model.py",
    source_project / "src" / "train.py",
    source_project / "tests" / "test_core_kd.py",
    source_project / "test" / "kaggle_core_compare.py",
    dfn_source,
    student_source,
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Offline bundle is incomplete:\n" + "\n".join(missing))

subprocess.run(
    [
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
        "lightning-utilities",
        "huggingface-hub",
        "ftfy",
        "regex",
        "tensorboard",
        "packaging",
        "tqdm",
        "numpy",
        "pillow",
        "matplotlib",
    ],
    cwd=WORKING_ROOT,
    check=True,
)

if dfn_source.stat().st_size != manifest["teacher_size"]:
    raise RuntimeError("DFN5B checkpoint size mismatch.")
if file_sha256(dfn_source) != manifest["teacher_sha256"]:
    raise RuntimeError("DFN5B checkpoint checksum mismatch.")
if student_source.stat().st_size != manifest["student_size"]:
    raise RuntimeError("ViT-B/32 checkpoint size mismatch.")
if file_sha256(student_source) != manifest["student_sha256"]:
    raise RuntimeError("ViT-B/32 checkpoint checksum mismatch.")

student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
shutil.copy2(student_source, student_cache / manifest["student_filename"])

hf_home = WORKING_ROOT / "huggingface"
hf_hub = hf_home / "hub"
repo_cache = hf_hub / ("models--" + manifest["teacher_repository"].replace("/", "--"))
snapshot = repo_cache / "snapshots" / manifest["teacher_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
shutil.copy2(dfn_source, snapshot / manifest["teacher_filename"])
refs = repo_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(manifest["teacher_revision"], encoding="utf-8")

os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_hub)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

previous_project_backup = None
if WORKING_PROJECT.exists():
    existing = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
    ).strip()
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=WORKING_PROJECT,
        check=False,
    ).returncode
    if existing != manifest["commit"] or changed:
        backup = WORKING_ROOT / (
            "KD-SBIR-AVKD_backup_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        )
        WORKING_PROJECT.rename(backup)
        previous_project_backup = backup
        print("Previous project preserved:", backup)
if not WORKING_PROJECT.exists():
    shutil.copytree(source_project, WORKING_PROJECT, symlinks=False)

if previous_project_backup is not None:
    for output_name in ("tb_logs", "saved_models"):
        old = previous_project_backup / output_name
        new = WORKING_PROJECT / output_name
        if old.is_dir() and not new.exists():
            shutil.copytree(old, new, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != manifest["commit"]:
    raise RuntimeError("Restored source commit differs from bundle manifest.")
for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")

# Format-v7 caches are optional. An old main cache is intentionally ignored.
cache_target = WORKING_ROOT / "teacher_cache" / CORE_CACHE_NAME
if not cache_target.is_file():
    candidates = []
    for pattern in (
        f"/kaggle/input/*/teacher_cache/{CORE_CACHE_NAME}",
        f"/kaggle/input/*/*/teacher_cache/{CORE_CACHE_NAME}",
        f"/kaggle/input/*/*/*/teacher_cache/{CORE_CACHE_NAME}",
    ):
        candidates.extend(Path(path) for path in glob.glob(pattern))
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise RuntimeError(
            "Attach at most one CoRe cache:\n" + "\n".join(map(str, candidates))
        )
    if candidates:
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], cache_target)
        print("Restored CoRe cache:", cache_target)
    else:
        print("No format-v7 CoRe cache attached; comparison will build it once.")

smoke_test = """
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_core_kd.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit('CoRe-KD smoke test failed')
"""
subprocess.run(
    [sys.executable, "-c", smoke_test],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print("=" * 70)
print("OFFLINE CORE-KD SETUP COMPLETE")
print("=" * 70)
print("Project:", WORKING_PROJECT)
print("Commit:", actual_commit)
print("Next: run test/kaggle_core_compare.py as one notebook cell.")
