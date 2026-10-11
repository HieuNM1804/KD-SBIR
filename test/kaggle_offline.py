"""Offline cell 1: restore the attached bidirectional teacher triplet and symmetric student KD bundle."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/teacher-triplet-bidirectional-symmetric-kd"
EXPECTED_BASE_COMMIT = "a8d22f4f95b6f853ff164a744ca9c4b111f54aa1"
EXPECTED_COMMIT = "908e482f7a99cb16f023dd3527fdfe24cfef8ca4"
EXPECTED_TASK = "teacher_triplet_bidirectional_symmetric_kd"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

# Request runtime roots rather than forcing every transitive version from the
# builder's Python image. pip can reuse compatible packages supplied by Kaggle.
RUNTIME_REQUIREMENTS = [
    "open-clip-torch==3.2.0", "pytorch-lightning", "torchmetrics",
    "lightning-utilities", "huggingface-hub", "ftfy", "regex", "tensorboard",
    "packaging", "tqdm", "numpy", "pillow", "matplotlib",
]

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR"
SKETCHY_ROOT = Path(
    "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
)


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

matching_bundles = []
manifest_reports = []
for manifest_path in manifest_paths:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        manifest_reports.append(
            f"- {manifest_path}\n"
            f"  unreadable: {type(error).__name__}: {error}"
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
        and manifest.get("base_commit") == EXPECTED_BASE_COMMIT
    ):
        matching_bundles.append((manifest_path, manifest))

if not matching_bundles:
    raise FileNotFoundError(
        "Cannot find the required symmetric-KD-based bidirectional teacher triplet and symmetric student KD bundle.\n\n"
        f"Expected branch: {EXPECTED_BRANCH}\n"
        f"Expected commit: {EXPECTED_COMMIT}\n"
        f"Expected task: {EXPECTED_TASK}\n"
        f"Expected entrypoint: {EXPECTED_ENTRYPOINT}\n\n"
        "Manifest files inspected:\n"
        + ("\n".join(manifest_reports) if manifest_reports else "(none)")
    )
if len(matching_bundles) > 1:
    print("Warning: matching bundle attached more than once; using the first.")

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
    source_project / "src" / "dataset.py",
    source_project / "src" / "losses.py",
    source_project / "src" / "model.py",
    source_project / "src" / "teacher_prompts.py",
    source_project / "src" / "train.py",
    source_project / "tests" / "test_teacher_triplet.py",
    source_project / "tests" / "test_symmetric_kd.py",
    dfn_source,
    student_source,
)
missing_paths = [
    str(path) for path in required_bundle_paths if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )
wheel_files = list(wheels.glob("*.whl"))
if not wheel_files:
    raise FileNotFoundError(f"No wheel files found inside {wheels}")
print("Offline wheels:", len(wheel_files))

if file_sha256(bundle / "requirements.txt") != manifest["requirements_sha256"]:
    raise RuntimeError("Offline requirements checksum mismatch.")
for record in manifest["wheels"]:
    wheel = wheels / record["filename"]
    if not wheel.is_file() or wheel.stat().st_size != record["size"] or file_sha256(wheel) != record["sha256"]:
        raise RuntimeError(f"Offline wheel integrity check failed: {wheel}")
print("Installing runtime dependencies; reusing compatible Kaggle packages.")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
     str(wheels), "--only-binary=:all:", *RUNTIME_REQUIREMENTS],
    cwd=WORKING_ROOT, check=True,
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

student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
shutil.copy2(student_source, student_target)
print("Student checkpoint:", student_target)

hf_home = WORKING_ROOT / "huggingface"
hf_hub = hf_home / "hub"
repo_cache_name = "models--" + manifest["teacher_repo"].replace("/", "--")
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
print("DFN5B checkpoint:", dfn_target)

subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "from huggingface_hub import hf_hub_download; "
            f"path = hf_hub_download({manifest['teacher_repo']!r}, "
            f"filename={manifest['teacher_filename']!r}, "
            "local_files_only=True); "
            "print('DFN5B offline cache resolved:', path)"
        ),
    ],
    cwd=WORKING_ROOT,
    check=True,
    env=os.environ.copy(),
)

os.chdir(WORKING_ROOT)
if WORKING_PROJECT.exists():
    backup = WORKING_ROOT / ("KD-SBIR_previous_teacher_triplet_bidirectional_symmetric_kd")
    counter = 1
    while backup.exists():
        backup = WORKING_ROOT / f"KD-SBIR_previous_teacher_triplet_bidirectional_symmetric_kd_{counter}"
        counter += 1
    WORKING_PROJECT.rename(backup)
    print("Previous project preserved:", backup)
shutil.copytree(source_project, WORKING_PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != EXPECTED_COMMIT:
    raise RuntimeError(
        "Repository commit mismatch:\n"
        f"Expected: {EXPECTED_COMMIT}\n"
        f"Actual:   {actual_commit}"
    )
print("Repository copied:", WORKING_PROJECT)
if subprocess.check_output(
    ["git", "merge-base", "HEAD", EXPECTED_BASE_COMMIT],
    cwd=WORKING_PROJECT, text=True,
).strip() != EXPECTED_BASE_COMMIT:
    raise RuntimeError("Offline source is not based on the expected symmetric-KD base commit.")

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")

# Verify bidirectional teacher triplets and symmetric KD values/gradients without DFN5B.
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests",
     "-p", "test_*.py", "-v"],
    cwd=WORKING_PROJECT, check=True, env=os.environ.copy(),
)
subprocess.run(
    [sys.executable, "-m", EXPECTED_ENTRYPOINT, "--help"],
    cwd=WORKING_PROJECT, check=True, env=os.environ.copy(), stdout=subprocess.DEVNULL,
)

print()
print("=" * 70)
print("OFFLINE BIDIRECTIONAL TEACHER TRIPLET + SYMMETRIC STUDENT KD SETUP COMPLETE")
print("=" * 70)
print("Project:", WORKING_PROJECT)
print("Dataset:", SKETCHY_ROOT)
print("Branch:", manifest["branch"])
print("Commit:", actual_commit)
print("Entry point:", manifest["entrypoint"])
print("Student checkpoint:", student_target)
print("Teacher checkpoint:", dfn_target)
print()
print("Teacher triplet: 0.5 * (sketch -> photo + photo -> sketch). Domain/photo-text/sketch-text KD: 0.5 * (KL(T||S) + KL(S||T)). Run test/kaggle_train.ipy next.")
