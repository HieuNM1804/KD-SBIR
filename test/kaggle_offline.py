"""Restore staged fine-grained teacher semantic refinement on Kaggle."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/fine-grained-teacher-semantic-refinement"
EXPECTED_COMMIT = "0edd774d5f8503da86fdf5d5bdb63c15053c2aed"
EXPECTED_TASK = "fine_grained_teacher_semantic_refinement"
EXPECTED_ENTRYPOINT = "src.train_fg"
EXPECTED_DATASET = "b20dccn616nguynhutun/sketchy-fg"

WORKING_ROOT = Path("/kaggle/working")
WORKING_PROJECT = WORKING_ROOT / "KD-SBIR"
SKETCHY_ROOT = Path(
    "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_ROOT)

# Search only a few Kaggle input depths. Recursively walking Sketchy-FG is slow.
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
        "Cannot find the required staged FG teacher bundle.\n\n"
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
    source_project / "src" / "dataset_fg.py",
    source_project / "src" / "image_text_prompts.py",
    source_project / "src" / "losses.py",
    source_project / "src" / "losses_fg.py",
    source_project / "src" / "model.py",
    source_project / "src" / "model_fg.py",
    source_project / "src" / "teacher_prompts.py",
    source_project / "src" / "teacher_refinement_report.py",
    source_project / "src" / "train_fg.py",
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

# Restore OpenAI CLIP where the vendored downloader expects it.
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
shutil.copy2(student_source, student_target)
print("Student checkpoint:", student_target)

# Reconstruct the Hugging Face snapshot expected by open_clip in offline mode.
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

# Import and numerical smoke test without loading the multi-gigabyte teacher.
# Run backward on the GPU here so deterministic/autocast errors fail before the
# expensive teacher cache pass starts.
smoke_test = """
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import open_clip
import pytorch_lightning

from src.image_text_prompts import PatchToTextContexts
from src.losses_fg import (
    fine_grained_prompt_infonce_loss,
    fine_grained_teacher_infonce_loss,
    image_conditioned_text_anchor_loss,
    teacher_semantic_refinement_loss,
    teacher_visual_refinement_control_loss,
)
from src.model_fg import FineGrainedCustomCLIP, FineGrainedZS_SBIR
from src.teacher_refinement_report import report_path_for_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
projector = PatchToTextContexts(64, 32, 8, 42).to(device)
patches = torch.randn(2, 49, 64, device=device, requires_grad=True)
base_context = torch.randn(8, 32, device=device, requires_grad=True)
contexts = projector(patches, base_context)
assert contexts.shape == (2, 8, 32)
contexts.square().mean().backward()
assert patches.grad is not None

sketch_images = torch.randn(3, 16, device=device, requires_grad=True)
photo_images = torch.randn(100, 16, device=device, requires_grad=True)
sketch_text = torch.randn(3, 16, device=device, requires_grad=True)
photo_text = torch.randn(100, 16, device=device, requires_grad=True)
targets = torch.tensor([0, 1, 2], device=device)
autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
with torch.autocast(device_type=device.type, dtype=autocast_dtype):
    visual_loss = fine_grained_teacher_infonce_loss(
        sketch_images,
        photo_images,
        targets,
        0.07,
    )
    prompt_loss, prompt_parts = fine_grained_prompt_infonce_loss(
        sketch_images,
        photo_images,
        sketch_text,
        photo_text,
        targets,
        0.07,
    )
    loss = visual_loss + prompt_loss
assert visual_loss.ndim == prompt_loss.ndim == loss.ndim == 0
assert visual_loss.dtype == prompt_loss.dtype == loss.dtype == torch.float32
assert set(prompt_parts) == {
    "sketch_to_photo_text",
    "sketch_text_to_photo",
}
assert torch.isfinite(loss)
loss.backward()
for features in (sketch_images, photo_images, sketch_text, photo_text):
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0

current_sketch = torch.randn(3, 16, device=device, requires_grad=True)
current_photo = torch.randn(100, 16, device=device, requires_grad=True)
source_sketch = torch.randn(3, 16, device=device, requires_grad=True)
source_photo = torch.randn(100, 16, device=device, requires_grad=True)
fixed_sketch_text = torch.randn(3, 16, device=device, requires_grad=True)
fixed_photo_text = torch.randn(100, 16, device=device, requires_grad=True)
with torch.autocast(device_type=device.type, dtype=autocast_dtype):
    refine_loss, refine_parts = teacher_semantic_refinement_loss(
        current_sketch,
        current_photo,
        source_sketch,
        source_photo,
        fixed_sketch_text,
        fixed_photo_text,
        targets,
        0.07,
        0.07,
        1.0,
        0.25,
        0.1,
    )
refine_loss.backward()
assert current_sketch.grad is not None and current_photo.grad is not None
assert source_sketch.grad is None and source_photo.grad is None
assert fixed_sketch_text.grad is None and fixed_photo_text.grad is None
assert set(refine_parts) == {
    "retrieval",
    "semantic",
    "keep",
    "sketch_to_photo_text",
    "sketch_text_to_photo",
}
control_sketch = torch.randn(3, 16, device=device, requires_grad=True)
control_photo = torch.randn(100, 16, device=device, requires_grad=True)
control_source_sketch = torch.randn(3, 16, device=device, requires_grad=True)
control_source_photo = torch.randn(100, 16, device=device, requires_grad=True)
with torch.autocast(device_type=device.type, dtype=autocast_dtype):
    control_loss, control_parts = teacher_visual_refinement_control_loss(
        control_sketch,
        control_photo,
        control_source_sketch,
        control_source_photo,
        targets,
        0.07,
        1.0,
        0.1,
    )
control_loss.backward()
assert control_sketch.grad is not None and control_photo.grad is not None
assert control_source_sketch.grad is None
assert control_source_photo.grad is None
assert set(control_parts) == {"retrieval", "keep"}
anchor_loss = image_conditioned_text_anchor_loss(
    fixed_sketch_text.detach(),
    fixed_photo_text.detach(),
    fixed_sketch_text.detach(),
    fixed_photo_text.detach(),
)
assert abs(anchor_loss.item()) < 1e-5
assert report_path_for_cache("teacher.pt").name == "teacher.pt.metrics.json"

print("PyTorch:", torch.__version__)
print("OpenCLIP:", getattr(open_clip, "__version__", "unknown"))
print("Lightning:", pytorch_lightning.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Deterministic patch-pooling backward: OK")
print("Exact-instance visual/prompt InfoNCE backward: OK")
print("Staged semantic refinement stop-gradient direction: OK")
print("Matched visual-only control stop-gradient direction: OK")
"""
subprocess.run(
    [sys.executable, "-c", smoke_test],
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
print("OFFLINE FG STAGED TEACHER SEMANTIC REFINEMENT SETUP COMPLETE")
print("=" * 70)
print("Project:", WORKING_PROJECT)
print("Dataset:", SKETCHY_ROOT)
print("Branch:", manifest["branch"])
print("Commit:", actual_commit)
print("Entry point:", manifest["entrypoint"])
print("Student checkpoint:", student_target)
print("Teacher checkpoint:", dfn_target)
print()
print("Run the src.train_fg command from README.md next.")
