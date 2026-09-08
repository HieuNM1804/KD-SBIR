"""Restore cross-domain visual ICL on offline Kaggle."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/clip-kd-interactive-contrastive"
EXPECTED_COMMIT = "d056369f0b64f863678b707d5964391d9e9f7a20"
EXPECTED_TASK = "clip_kd_interactive_contrastive"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy"

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
    ):
        matching_bundles.append((manifest_path, manifest))

if not matching_bundles:
    raise FileNotFoundError(
        "Cannot find the required CLIP visual-ICL bundle.\n\n"
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
    shutil.rmtree(WORKING_PROJECT)
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

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")

subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "import torch, open_clip, pytorch_lightning; "
            "from src.losses import multi_positive_contrastive_loss; "
            "from src.model import ICLProjector, ZS_SBIR; "
            "photo_projector = ICLProjector(); "
            "sketch_projector = ICLProjector(); "
            "assert photo_projector(torch.randn(2, 512)).shape == (2, 1024); "
            "assert sketch_projector(torch.randn(2, 512)).shape == (2, 1024); "
            "assert photo_projector.projection.weight.data_ptr() != "
            "sketch_projector.projection.weight.data_ptr(); "
            "print('PyTorch:', torch.__version__); "
            "print('OpenCLIP:', getattr(open_clip, '__version__', 'unknown')); "
            "print('Lightning:', pytorch_lightning.__version__); "
            "print('CUDA available:', torch.cuda.is_available()); "
            "loss = multi_positive_contrastive_loss("
            "torch.randn(2, 1024), torch.randn(2, 1024), "
            "torch.tensor([0, 1]), torch.tensor([0, 1]), 1.0); "
            "assert loss.ndim == 0; "
            "print('Cross-domain visual ICL imports: OK')"
        ),
    ],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
)
subprocess.run(
    [sys.executable, "-m", EXPECTED_ENTRYPOINT, "--help"],
    cwd=WORKING_PROJECT,
    check=True,
    env=os.environ.copy(),
    stdout=subprocess.DEVNULL,
)

print()
print("=" * 70)
print("OFFLINE CLIP-KD CROSS-DOMAIN VISUAL ICL SETUP COMPLETE")
print("=" * 70)
print("Project:", WORKING_PROJECT)
print("Dataset:", SKETCHY_ROOT)
print("Branch:", manifest["branch"])
print("Commit:", actual_commit)
print("Entry point:", manifest["entrypoint"])
print("Student checkpoint:", student_target)
print("Teacher checkpoint:", dfn_target)
print()
print("Run the cross-domain visual ICL src.train cell next.")
