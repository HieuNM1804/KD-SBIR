"""Build the offline Kaggle bundle for the Sketchy2 SGCD comparison.

Run once with Internet enabled. The output contains pinned source, offline Python
wheels, ViT-B/32 and DFN5B. Save the output and attach it as an input dataset
to the offline GPU notebook.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKING = Path("/kaggle/working")
BUNDLE = (
    WORKING
    / ("sgcd_sketchy2_bundle_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"))
    / "offline_bundle"
)
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
DFN_DIR = BUNDLE / "dfn5b_openclip"

REPO_URL = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/sgcd-native-prompt-learning"
# Pinned training source; this builder is distributed separately.
COMMIT = "a021499ae64a969f0eec816e9c0a0abb57943537"
TASK = "sgcd_sketchy2_where_effect"
ENTRYPOINT = "src.train"
DATASET = "b20dccn616nguynhutun/sketchy"

DFN_REPO = "apple/DFN5B-CLIP-ViT-H-14"
DFN_FILENAME = "open_clip_pytorch_model.bin"
DFN_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
DFN_SHA256 = "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae"
STUDENT_FILENAME = "ViT-B-32.pt"
STUDENT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
for directory in (WHEELS, SOURCE, CLIP_CACHE, DFN_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/6] Clean bundle created:", BUNDLE)

requirements = """
open-clip-torch
pytorch-lightning
torchmetrics
lightning-utilities
huggingface-hub
ftfy
regex
tensorboard
packaging
tqdm
numpy
pillow
matplotlib
"""
requirements_path = BUNDLE / "requirements.txt"
requirements_path.write_text(requirements.strip() + "\n", encoding="utf-8")
run(
    [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(WHEELS),
        "--only-binary=:all:",
        "-r",
        str(requirements_path),
    ],
    WORKING,
)

# Kaggle already supplies its CUDA/PyTorch stack. Do not bundle a second stack.
for pattern in (
    "torch-*.whl",
    "torchvision-*.whl",
    "torchaudio-*.whl",
    "triton-*.whl",
    "nvidia_*.whl",
    "cuda_*.whl",
):
    for wheel in WHEELS.glob(pattern):
        print("Removing Kaggle-provided wheel:", wheel.name)
        wheel.unlink()

required_wheels = (
    "open_clip_torch-*.whl",
    "pytorch_lightning-*.whl",
    "torchmetrics-*.whl",
    "lightning_utilities-*.whl",
    "huggingface_hub-*.whl",
    "ftfy-*.whl",
    "regex-*.whl",
    "matplotlib-*.whl",
)
missing_wheels = [
    pattern for pattern in required_wheels if not list(WHEELS.glob(pattern))
]
if missing_wheels:
    raise FileNotFoundError("Missing required wheels:\n" + "\n".join(missing_wheels))
print("[2/6] Wheels downloaded:", len(list(WHEELS.glob("*.whl"))))

# These lightweight packages are needed by this online builder itself.
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "huggingface-hub",
        "ftfy",
        "regex",
        "packaging",
    ],
    WORKING,
)

project = SOURCE / "KD-SBIR"
clone_args = [
    "git",
    "clone",
    "--branch",
    BRANCH,
    "--single-branch",
    REPO_URL,
    str(project),
]
run(clone_args, WORKING)
if COMMIT is not None:
    run(["git", "checkout", "--detach", COMMIT], project)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=project, text=True
).strip()
if COMMIT is not None and actual_commit != COMMIT:
    raise RuntimeError(f"Repository commit mismatch: {actual_commit} != {COMMIT}")
print("[3/6] Source commit:", actual_commit)

from huggingface_hub import hf_hub_download

downloaded_dfn = Path(
    hf_hub_download(
        repo_id=DFN_REPO,
        filename=DFN_FILENAME,
        revision=DFN_REVISION,
    )
)
dfn_target = DFN_DIR / DFN_FILENAME
shutil.copy2(downloaded_dfn, dfn_target)
actual_dfn_sha = file_sha256(dfn_target)
if actual_dfn_sha != DFN_SHA256:
    raise RuntimeError(f"DFN5B checksum mismatch: {actual_dfn_sha} != {DFN_SHA256}")
print(
    "[4/6] DFN5B downloaded:",
    dfn_target,
    f"({dfn_target.stat().st_size / 1024**3:.3f} GiB)",
)

# Import the repository's vendored CLIP instead of an unrelated `clip` package.
for module_name in list(sys.modules):
    if module_name == "clip" or module_name.startswith("clip."):
        del sys.modules[module_name]
sys.path.insert(0, str(project))
from clip import clip as project_clip

downloaded_student = Path(project_clip.download_model("ViT-B/32"))
student_target = CLIP_CACHE / STUDENT_FILENAME
shutil.copy2(downloaded_student, student_target)
actual_student_sha = file_sha256(student_target)
if actual_student_sha != STUDENT_SHA256:
    raise RuntimeError(
        f"ViT-B/32 checksum mismatch: {actual_student_sha} != {STUDENT_SHA256}"
    )
print(
    "[5/6] ViT-B/32 downloaded:",
    student_target,
    f"({student_target.stat().st_size / 1024**2:.1f} MiB)",
)

required_paths = (
    BUNDLE / "requirements.txt",
    WHEELS,
    project / ".git",
    project / "clip" / "model.py",
    project / "src" / "dataset.py",
    project / "src" / "losses.py",
    project / "src" / "model.py",
    project / "src" / "teacher_prompts.py",
    project / "src" / "train.py",
    project / "src" / "stroke_graph.py",
    project / "src" / "stroke_graph_cache.py",
    project / "src" / "stroke_graph_reports.py",
    project / "src" / "stroke_graph_diagnostics.py",
    project / "src" / "stroke_prompt.py",
    project / "src" / "stroke_prompt_probe.py",
    project / "src" / "stroke_prompt_probe_cli.py",
    project / "tests" / "test_stroke_graph.py",
    project / "tests" / "test_sgcd_commands.py",
    project / "tests" / "test_stroke_prompt.py",
    project / "test" / "RUN_ORDER.txt",
    project / "test" / "kaggle_main_baseline_train.ipy",
    project / "test" / "kaggle_sgcd_audit.ipy",
    project / "test" / "kaggle_sgcd_prepare.ipy",
    project / "test" / "kaggle_sgcd_train.ipy",
    project / "test" / "kaggle_sgcd_native_prompt_train.ipy",
    project / "test" / "kaggle_sgcd_native_where_ablation.ipy",
    project / "test" / "kaggle_sgcd_native_where_what_ablation.ipy",
    project / "test" / "kaggle_sgcd_native_where_effect_ablation.ipy",
    project / "test" / "kaggle_sgcd_native_random_control.ipy",
    project / "test" / "kaggle_sgcd_native_shuffle_control.ipy",
    project / "test" / "kaggle_main_baseline_s43.ipy",
    project / "test" / "kaggle_main_baseline_s44.ipy",
    project / "test" / "kaggle_sgcd_native_where_effect_s43.ipy",
    project / "test" / "kaggle_sgcd_native_where_effect_s44.ipy",
    project / "test" / "kaggle_sgcd_prompt_preflight.py",
    project / "test" / "kaggle_sgcd_restore_caches.py",
    project / "test" / "kaggle_sgcd_local_control.ipy",
    project / "test" / "kaggle_sgcd_random_control.ipy",
    project / "test" / "kaggle_sgcd_shuffle_control.ipy",
    project / "test" / "kaggle_sgcd_standalone.ipy",
    project / "test" / "kaggle_sgcd_report.py",
    project / "test" / "kaggle_sgcd_sketchy2_baseline_where_effect.ipy",
    project / "test" / "kaggle_sgcd_sketchy2_online.py",
    project / "test" / "kaggle_sgcd_sketchy2_offline.py",
    project / "docs" / "sgcd.md",
    project / "docs" / "sgcd_native_prompt.md",
    dfn_target,
    student_target,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Offline bundle is incomplete. Missing:\n" + "\n".join(missing_paths)
    )

manifest = {
    "repository": REPO_URL,
    "branch": BRANCH,
    "commit": actual_commit,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "dataset": DATASET,
    "teacher_repo": DFN_REPO,
    "teacher_revision": DFN_REVISION,
    "teacher_filename": DFN_FILENAME,
    "teacher_sha256": DFN_SHA256,
    "teacher_size": dfn_target.stat().st_size,
    "student_filename": STUDENT_FILENAME,
    "student_sha256": STUDENT_SHA256,
    "student_size": student_target.stat().st_size,
    "python_version": sys.version,
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

bundle_size = sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
print("[6/6] Bundle validated")
print("=" * 70)
print("ONLINE SKETCHY2 SGCD BUNDLE COMPLETE")
print("=" * 70)
print("Bundle:", BUNDLE)
print("Branch:", BRANCH)
print("Commit:", actual_commit)
print("Entry point:", ENTRYPOINT)
print("DFN5B SHA256:", actual_dfn_sha)
print("Student SHA256:", actual_student_sha)
print("Bundle size:", f"{bundle_size / 1024**3:.3f} GiB")
print()
print("Save Version -> Save & Run All -> Always save output.")
