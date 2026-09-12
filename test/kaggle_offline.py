"""Restore patch attention output KD on offline Kaggle; setup only."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/patch-attention-output-kd"
EXPECTED_COMMIT = "ca92ba2149ef9ce7a4c89b15ac4ed7081062147a"
EXPECTED_TASK = "patch_attention_output_kd"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR-AVKD"
SKETCHY_ROOT = Path(
    "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Phase 1: Locate and validate the offline bundle ──────────────────

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
    commit_ok = (
        EXPECTED_COMMIT is None
        or manifest.get("commit") == EXPECTED_COMMIT
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and commit_ok
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("dataset") == EXPECTED_DATASET
    ):
        matching_bundles.append((manifest_path, manifest))

if not matching_bundles:
    raise FileNotFoundError(
        "Cannot find the required patch attention output KD bundle.\n\n"
        f"Expected branch: {EXPECTED_BRANCH}\n"
        f"Expected task: {EXPECTED_TASK}\n\n"
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


# ── Phase 2: Validate bundle contents ────────────────────────────────

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
    source_project / "src" / "attention_output_kd.py",
    source_project / "src" / "attention_output_cache.py",
    source_project / "tests" / "test_attention_output_kd.py",
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


# ── Phase 3: Install packages offline ────────────────────────────────

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


# ── Phase 4: Validate checksums ──────────────────────────────────────

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


# ── Phase 5: Restore model caches ───────────────────────────────────

# OpenAI CLIP cache.
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
shutil.copy2(student_source, student_target)
print("Student checkpoint:", student_target)

# Hugging Face snapshot for offline open_clip.
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


# ── Phase 6: Copy source to working directory ───────────────────────

os.chdir(WORKING_ROOT)
if WORKING_PROJECT.exists():
    existing = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True).strip()
    if existing != manifest["commit"]:
        raise RuntimeError("Existing KD-SBIR-AVKD uses another commit. Choose a new WORKING_PROJECT path.")
    print("Reusing existing project:", WORKING_PROJECT)
else:
    shutil.copytree(source_project, WORKING_PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=WORKING_PROJECT, text=True
).strip()
if actual_commit != manifest["commit"]:
    raise RuntimeError("Restored source commit differs from the bundle manifest")
print("Repository copied:", WORKING_PROJECT)
print("Commit:", actual_commit)


# ── Phase 7: Validate dataset ───────────────────────────────────────

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
print("Dataset:", SKETCHY_ROOT)


# ── Phase 8: Smoke test ─────────────────────────────────────────────

smoke_test = """
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch, open_clip, pytorch_lightning, matplotlib
print('PyTorch:', torch.__version__)
print('OpenCLIP:', getattr(open_clip, '__version__', 'unknown'))
print('Lightning:', pytorch_lightning.__version__)
print('Matplotlib:', matplotlib.__version__)
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_attention_output_kd.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit('Attention output KD smoke test failed')
print('AV extraction, prompt gradients, cached targets and ablations: OK')
"""
subprocess.run(
    [sys.executable, "-c", smoke_test],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)

print()
print("=" * 70)
print("OFFLINE SETUP COMPLETE — READY TO TRAIN")
print("=" * 70)


os.chdir(WORKING_PROJECT)
print("Project:", WORKING_PROJECT)
print("Run the cache preparation and training commands supplied in chat.")
