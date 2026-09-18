"""Restore retrieval-vocabulary PromptKD in an offline Kaggle GPU notebook."""

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
EXPECTED_BRANCH = "experiment/promptkd-photo-sketch-prototypes"
EXPECTED_COMMIT = "a536d707a353bbe5b3e4471e3447ce21aed34cbf"
EXPECTED_TASK = "promptkd_photo_sketch_retrieval_vocabulary"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR-AVKD"
SKETCHY_ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
TEACHER_CACHE_NAME = "sketchy1_teacher_1ep.pt"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_ROOT)

# Locate the output saved by kaggle_promptkd_online.py.
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

matching_bundles = []
manifest_reports = []
for manifest_path in manifest_paths:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        manifest_reports.append(
            f"- {manifest_path}\n  unreadable: {type(error).__name__}: {error}"
        )
        continue

    manifest_reports.append(
        f"- {manifest_path}\n"
        f"  branch: {manifest.get('branch')}\n"
        f"  commit: {manifest.get('commit')}\n"
        f"  task: {manifest.get('task')}\n"
        f"  entrypoint: {manifest.get('entrypoint')}\n"
        f"  dataset: {manifest.get('dataset')}"
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and manifest.get("commit") == EXPECTED_COMMIT
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("dataset") == EXPECTED_DATASET
    ):
        matching_bundles.append((manifest_path, manifest))

if not matching_bundles:
    raise FileNotFoundError(
        "Cannot find the required PromptKD bundle.\n\n"
        f"Expected branch: {EXPECTED_BRANCH}\n"
        f"Expected commit: {EXPECTED_COMMIT}\n\n"
        "Manifest files inspected:\n"
        + ("\n".join(manifest_reports) if manifest_reports else "(none)")
    )
if len(matching_bundles) > 1:
    raise RuntimeError(
        "The same PromptKD bundle is attached more than once. Keep one copy only."
    )

manifest_path, manifest = matching_bundles[0]
bundle = manifest_path.parent
wheels = bundle / "wheels"
source_project = bundle / "source" / "KD-SBIR"
clip_cache = bundle / "clip_cache"
dfn_source = bundle / "dfn5b_openclip" / manifest["teacher_filename"]
student_source = clip_cache / manifest["student_filename"]
print("Selected bundle:", bundle)
print("Branch:", manifest["branch"])
print("Commit:", manifest["commit"])

required_bundle_paths = (
    bundle / "requirements.txt",
    wheels,
    source_project / ".git",
    source_project / "clip" / "model.py",
    source_project / "src" / "data_config.py",
    source_project / "src" / "dataset.py",
    source_project / "src" / "losses.py",
    source_project / "src" / "model.py",
    source_project / "src" / "photo_sketch_promptkd.py",
    source_project / "src" / "teacher_prompts.py",
    source_project / "src" / "train.py",
    source_project / "tests" / "test_photo_sketch_promptkd.py",
    source_project / "test" / "kaggle_promptkd_photo_sketch_train.ipy",
    source_project / "docs" / "photo_sketch_promptkd.md",
    dfn_source,
    student_source,
)
missing_paths = [str(path) for path in required_bundle_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )
if not list(wheels.glob("*.whl")):
    raise FileNotFoundError(f"No wheel files found inside {wheels}")

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
print("Offline Python packages installed")

if dfn_source.stat().st_size != manifest["teacher_size"]:
    raise RuntimeError("DFN5B checkpoint size mismatch.")
actual_dfn_sha = file_sha256(dfn_source)
if actual_dfn_sha != manifest["teacher_sha256"]:
    raise RuntimeError(
        "DFN5B checksum mismatch:\n"
        f"Expected: {manifest['teacher_sha256']}\n"
        f"Actual:   {actual_dfn_sha}"
    )
if student_source.stat().st_size != manifest["student_size"]:
    raise RuntimeError("ViT-B/32 checkpoint size mismatch.")
actual_student_sha = file_sha256(student_source)
if actual_student_sha != manifest["student_sha256"]:
    raise RuntimeError(
        "ViT-B/32 checksum mismatch:\n"
        f"Expected: {manifest['student_sha256']}\n"
        f"Actual:   {actual_student_sha}"
    )

# Restore OpenAI CLIP.
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
shutil.copy2(student_source, student_target)

# Restore the exact Hugging Face snapshot for offline open_clip loading.
hf_home = WORKING_ROOT / "huggingface"
hf_hub = hf_home / "hub"
repo_cache_name = "models--" + manifest["teacher_repository"].replace("/", "--")
repo_cache = hf_hub / repo_cache_name
snapshot = repo_cache / "snapshots" / manifest["teacher_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
dfn_target = snapshot / manifest["teacher_filename"]
shutil.copy2(dfn_source, dfn_target)
refs = repo_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(manifest["teacher_revision"], encoding="utf-8")

os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_hub)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print("Student checkpoint:", student_target)
print("DFN5B checkpoint:", dfn_target)

# Install the exact pinned source under the path used by the training cell.
os.chdir(WORKING_ROOT)
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
else:
    print("Reusing exact project:", WORKING_PROJECT)

if previous_project_backup is not None:
    for output_name in ("tb_logs", "saved_models"):
        previous_output = previous_project_backup / output_name
        restored_output = WORKING_PROJECT / output_name
        if previous_output.is_dir() and not restored_output.exists():
            shutil.copytree(previous_output, restored_output, symlinks=False)
            print("Restored previous experiment output:", restored_output)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != manifest["commit"]:
    raise RuntimeError("Restored source commit differs from the bundle manifest")

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")

# Reuse a previously saved teacher cache when exactly one copy is attached.
teacher_cache_target = WORKING_ROOT / "teacher_cache" / TEACHER_CACHE_NAME
if not teacher_cache_target.is_file():
    cache_candidates = []
    for pattern in (
        f"/kaggle/input/*/teacher_cache/{TEACHER_CACHE_NAME}",
        f"/kaggle/input/*/*/teacher_cache/{TEACHER_CACHE_NAME}",
        f"/kaggle/input/*/*/*/teacher_cache/{TEACHER_CACHE_NAME}",
        f"/kaggle/input/*/*/*/*/teacher_cache/{TEACHER_CACHE_NAME}",
    ):
        cache_candidates.extend(Path(path) for path in glob.glob(pattern))
    cache_candidates = sorted(set(cache_candidates))
    if len(cache_candidates) > 1:
        raise RuntimeError(
            "Multiple teacher caches are attached. Keep only one:\n"
            + "\n".join(str(path) for path in cache_candidates)
        )
    if cache_candidates:
        teacher_cache_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_candidates[0], teacher_cache_target)
        print("Restored teacher cache:", teacher_cache_target)
    else:
        print("No saved teacher cache attached; the training cell will build it once.")

smoke_test = """
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch, open_clip, pytorch_lightning, matplotlib
print('PyTorch:', torch.__version__)
print('OpenCLIP:', getattr(open_clip, '__version__', 'unknown'))
print('Lightning:', pytorch_lightning.__version__)
print('Matplotlib:', matplotlib.__version__)
import unittest
suite = unittest.defaultTestLoader.discover(
    'tests', pattern='test_photo_sketch_promptkd.py'
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit('Photo-sketch PromptKD smoke test failed')
print('Paired retrieval vocabulary construction and losses: OK')
"""
subprocess.run(
    [sys.executable, "-c", smoke_test],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print()
print("=" * 70)
print("OFFLINE PROMPTKD SETUP COMPLETE — READY TO TRAIN")
print("=" * 70)
print("Project:", WORKING_PROJECT)
print("Commit:", actual_commit)
print("Dataset:", SKETCHY_ROOT)
print("Next: run test/kaggle_promptkd_photo_sketch_train.ipy as one cell.")
