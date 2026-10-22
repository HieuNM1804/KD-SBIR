"""Restore the pinned across-dataset ZS-SBIR project on offline Kaggle.

Attach the online bundle, Sketchy, and the chosen target dataset first. Set
ACROSS_TARGET_DATASET to ``tuberlin`` (default) or ``quickdraw`` before running
this cell if desired. ACROSS_TARGET_ROOT can override automatic root discovery.
This setup performs validation but deliberately runs no test suite.
"""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


EXPECTED_REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
EXPECTED_BRANCH = "experiment/across-dataset-sbir"
EXPECTED_COMMIT = "46c73707897ca386b5e09a750a6f9373f2cb26b4"
EXPECTED_BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
EXPECTED_TASK = "across_dataset_zs_sbir"
EXPECTED_ENTRYPOINT = "src.train"
EXPECTED_PROTOCOL = "across_dataset_zs_sbir"
EXPECTED_SOURCE_DATASET = "sketchy_1"
EXPECTED_TEACHER_REPOSITORY = "apple/DFN5B-CLIP-ViT-H-14"
EXPECTED_TEACHER_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
EXPECTED_STUDENT_MODEL = "ViT-B/32"

WORKING = Path("/kaggle/working")
PROJECT = WORKING / "KD-SBIR"
SKETCHY_ROOT = Path(
    "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
)

# Set these in a preceding notebook cell or edit the defaults here.
TARGET_DATASET = globals().get("ACROSS_TARGET_DATASET", "tuberlin")
TARGET_ROOT_OVERRIDE = globals().get("ACROSS_TARGET_ROOT", "")
TARGET_ROOT_OVERRIDE = (
    Path(TARGET_ROOT_OVERRIDE) if TARGET_ROOT_OVERRIDE else None
)
SUPPORTED_TARGETS = {"tuberlin": 21, "quickdraw": 11}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value):
    if (
        not isinstance(value, str)
        or Path(value).name != value
        or value in {"", ".", ".."}
    ):
        raise RuntimeError(f"Unsafe filename in manifest: {value!r}")
    return value


def validate(path, expected_size, expected_sha256):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: {path.stat().st_size} != "
            f"{expected_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}:\n"
            f"expected: {expected_sha256}\nactual:   {actual_sha256}"
        )


def next_backup(path):
    backup = path.with_name(path.name + "_previous")
    counter = 1
    while backup.exists():
        backup = path.with_name(path.name + f"_previous_{counter}")
        counter += 1
    return backup


def is_dataset_root(path, classes):
    return (
        path.is_dir()
        and (path / "sketch").is_dir()
        and (path / "photo").is_dir()
        and all((path / "sketch" / name).is_dir() for name in classes)
        and all((path / "photo" / name).is_dir() for name in classes)
    )


def find_target_root(classes):
    if TARGET_ROOT_OVERRIDE is not None:
        if not is_dataset_root(TARGET_ROOT_OVERRIDE, classes):
            raise FileNotFoundError(
                "ACROSS_TARGET_ROOT is not a target dataset root with all "
                f"required {TARGET_DATASET} sketch/photo classes: "
                f"{TARGET_ROOT_OVERRIDE}"
            )
        return TARGET_ROOT_OVERRIDE

    # Match only directories named sketch. This remains bounded without ever
    # globbing the image files contained below category directories.
    candidates = []
    for pattern in (
        "/kaggle/input/*/sketch",
        "/kaggle/input/*/*/sketch",
        "/kaggle/input/*/*/*/sketch",
        "/kaggle/input/*/*/*/*/sketch",
        "/kaggle/input/*/*/*/*/*/sketch",
    ):
        candidates.extend(Path(value).parent for value in glob.glob(pattern))
    matches = [
        path for path in sorted(set(candidates))
        if path.resolve() != SKETCHY_ROOT.resolve()
        and is_dataset_root(path, classes)
    ]
    if len(matches) != 1:
        discovered = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"Expected exactly one {TARGET_DATASET} root, found "
            f"{len(matches)}. Set ACROSS_TARGET_ROOT explicitly."
            + ("\nMatches:\n" + discovered if discovered else "")
        )
    return matches[0]


if TARGET_DATASET not in SUPPORTED_TARGETS:
    raise ValueError(
        f"ACROSS_TARGET_DATASET must be one of {sorted(SUPPORTED_TARGETS)}, "
        f"got {TARGET_DATASET!r}."
    )

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
    reports.append(
        f"{path}: branch={manifest.get('branch')}, "
        f"commit={manifest.get('commit')}, task={manifest.get('task')}"
    )
    if (
        manifest.get("repository") == EXPECTED_REPOSITORY
        and manifest.get("branch") == EXPECTED_BRANCH
        and manifest.get("commit") == EXPECTED_COMMIT
        and manifest.get("base_commit") == EXPECTED_BASE_COMMIT
        and manifest.get("task") == EXPECTED_TASK
        and manifest.get("entrypoint") == EXPECTED_ENTRYPOINT
        and manifest.get("evaluation_protocol") == EXPECTED_PROTOCOL
        and manifest.get("source_dataset") == EXPECTED_SOURCE_DATASET
        and manifest.get("teacher_repository")
        == EXPECTED_TEACHER_REPOSITORY
        and manifest.get("teacher_revision") == EXPECTED_TEACHER_REVISION
        and manifest.get("student_model") == EXPECTED_STUDENT_MODEL
    ):
        matches.append((path.parent, manifest))

input_matches = [
    entry for entry in matches if str(entry[0]).startswith("/kaggle/input/")
]
if input_matches:
    matches = input_matches
if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one across-dataset bundle, found {len(matches)}.\n"
        + ("\n".join(reports) if reports else "No manifests found.")
    )

bundle, manifest = matches[0]
print("Builder Python:", manifest["python_major_minor"], "Offline Python:", list(sys.version_info[:2]))

protocol_key = f"{EXPECTED_SOURCE_DATASET}->{TARGET_DATASET}"
target_classes = manifest.get("cross_dataset_classes", {}).get(protocol_key)
if not isinstance(target_classes, list):
    raise RuntimeError(f"Bundle has no protocol class list for {protocol_key}.")
if (
    len(target_classes) != SUPPORTED_TARGETS[TARGET_DATASET]
    or len(target_classes) != len(set(target_classes))
):
    raise RuntimeError(
        f"Invalid {protocol_key} class list in bundle: "
        f"expected {SUPPORTED_TARGETS[TARGET_DATASET]} unique classes."
    )

wheels = bundle / "wheels"
source = bundle / "source" / "KD-SBIR"
teacher_source = (
    bundle
    / "dfn5b_openclip"
    / safe_filename(manifest["teacher_filename"])
)
student_source = (
    bundle / "clip_cache" / safe_filename(manifest["student_filename"])
)
required = (
    bundle / "requirements.txt",
    wheels,
    source / ".git",
    source / "clip" / "model.py",
    source / "src" / "data_config.py",
    source / "src" / "dataset.py",
    source / "src" / "losses.py",
    source / "src" / "model.py",
    source / "src" / "teacher_prompts.py",
    source / "src" / "train.py",
    source / "tests" / "test_cross_dataset_protocol.py",
    teacher_source,
    student_source,
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Across-dataset bundle is incomplete:\n" + "\n".join(missing)
    )

if not manifest.get("wheels"):
    raise RuntimeError("Bundle has an empty wheel manifest.")
validate(
    teacher_source,
    manifest["teacher_size"],
    manifest["teacher_sha256"],
)
validate(
    student_source,
    manifest["student_size"],
    manifest["student_sha256"],
)
for wheel in manifest["wheels"]:
    validate(
        wheels / safe_filename(wheel["filename"]),
        wheel["size"],
        wheel["sha256"],
    )
print("Selected bundle:", bundle)
print("Validated wheels:", len(manifest["wheels"]))

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels),
        *manifest["runtime_requirements"],
    ],
    cwd=WORKING,
    check=True,
)
print("Offline Python packages installed")

# Preserve any unrelated project instead of deleting a previous experiment.
if PROJECT.exists():
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        current.returncode != 0
        or current.stdout.strip() != EXPECTED_COMMIT
        or status.returncode != 0
        or status.stdout.strip()
    ):
        backup = next_backup(PROJECT)
        PROJECT.rename(backup)
        print("Previous project preserved:", backup)
if not PROJECT.exists():
    shutil.copytree(source, PROJECT, symlinks=False)

actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
merge_base = subprocess.check_output(
    ["git", "merge-base", "HEAD", EXPECTED_BASE_COMMIT],
    cwd=PROJECT,
    text=True,
).strip()
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=PROJECT, text=True
).strip()
if actual_commit != EXPECTED_COMMIT or merge_base != EXPECTED_BASE_COMMIT or dirty:
    raise RuntimeError("Restored source failed commit/base/clean-tree validation.")

# Restore the OpenAI CLIP student where the vendored downloader expects it.
student_cache = Path.home() / ".cache" / "clip"
student_cache.mkdir(parents=True, exist_ok=True)
student_target = student_cache / manifest["student_filename"]
if student_target.exists() and sha256(student_target) != manifest["student_sha256"]:
    student_target.rename(next_backup(student_target))
if not student_target.exists():
    shutil.copy2(student_source, student_target)

# Reconstruct the Hugging Face snapshot resolved by OpenCLIP's dfn5b tag.
hf_home = WORKING / "huggingface_across"
hf_hub = hf_home / "hub"
teacher_cache = hf_hub / (
    "models--" + manifest["teacher_repository"].replace("/", "--")
)
snapshot = teacher_cache / "snapshots" / manifest["teacher_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
teacher_target = snapshot / manifest["teacher_filename"]
if teacher_target.exists() and sha256(teacher_target) != manifest["teacher_sha256"]:
    teacher_target.rename(next_backup(teacher_target))
if not teacher_target.exists():
    shutil.copy2(teacher_source, teacher_target)
refs = teacher_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(manifest["teacher_revision"], encoding="utf-8")
no_exist = teacher_cache / ".no_exist" / manifest["teacher_revision"]
no_exist.mkdir(parents=True, exist_ok=True)
(no_exist / "open_clip_model.safetensors").touch()

os.environ.update(
    HF_HOME=str(hf_home),
    HF_HUB_CACHE=str(hf_hub),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    HF_HUB_DISABLE_XET="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)

for directory in (SKETCHY_ROOT / "sketch", SKETCHY_ROOT / "photo"):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing Sketchy source directory: {directory}")
target_root = find_target_root(target_classes)

# Check source protocol data without allocating either model.
config = {}
exec((PROJECT / "src" / "data_config.py").read_text(encoding="utf-8"), config)
source_classes = config.get("CROSS_DATASET_CLASSES", {}).get(
    (EXPECTED_SOURCE_DATASET, TARGET_DATASET)
)
if source_classes != target_classes:
    raise RuntimeError(
        "Restored source class list differs from the bundle manifest."
    )

# Verify both checkpoint resolvers without allocating the 632M-parameter
# teacher or the student. This catches an incorrectly reconstructed cache.
resolver_code = f"""
from pathlib import Path
from huggingface_hub import hf_hub_download
from clip import clip
teacher = Path(hf_hub_download(
    repo_id={manifest['teacher_repository']!r},
    filename={manifest['teacher_filename']!r},
    revision={manifest['teacher_revision']!r},
    local_files_only=True,
))
student = Path(clip.download_model({EXPECTED_STUDENT_MODEL!r}))
assert teacher.resolve() == Path({str(teacher_target)!r}).resolve()
assert student.resolve() == Path({str(student_target)!r}).resolve()
print('Teacher checkpoint resolved:', teacher)
print('Student checkpoint resolved:', student)
"""
subprocess.run(
    [sys.executable, "-c", resolver_code],
    cwd=PROJECT,
    env=os.environ.copy(),
    check=True,
)
subprocess.run(
    [sys.executable, "-m", EXPECTED_ENTRYPOINT, "--help"],
    cwd=PROJECT,
    env=os.environ.copy(),
    stdout=subprocess.DEVNULL,
    check=True,
)

exp_name = f"sketchy1_to_{TARGET_DATASET}"
run_command = (
    "!python -u -m src.train "
    f"--root {SKETCHY_ROOT} "
    "--dataset sketchy_1 "
    f"--target_root {target_root} "
    f"--target_dataset {TARGET_DATASET} "
    "--epochs 1 --workers 8 --batch_size 64 --test_batch_size 1024 "
    "--n_ctx_visual 3 --prompt_depth 12 "
    "--lambda_domain 2 --lambda_modality 2 --kd_temperature 0.07 "
    "--photo_text_kd_temperature 0.15 "
    "--sketch_text_kd_temperature 0.02 "
    "--lr 0.11 --momentum 0.95 --weight_decay 1e-4 "
    "--teacher_pretrain_epochs 1 --teacher_pretrain_batch_size 64 "
    "--teacher_n_ctx_visual 9 --teacher_prompt_depth 12 "
    "--teacher_prompt_std 0.02 --teacher_prompt_lr 0.03 "
    "--teacher_prompt_seed 42 --teacher_momentum 0.9 "
    "--teacher_weight_decay 1e-3 --lambda_teacher_retrieval 1.5 "
    "--teacher_triplet_margin 0.3 --seed 42 "
    f"--exp_name {exp_name} --progress"
)

print("=" * 72)
print("OFFLINE ACROSS-DATASET ZS-SBIR SETUP COMPLETE")
print("=" * 72)
print("Project:", PROJECT)
print("Source:", EXPECTED_SOURCE_DATASET, "at", SKETCHY_ROOT)
print("Target:", TARGET_DATASET, "at", target_root)
print("Target classes:", len(target_classes), target_classes)
print("Branch:", EXPECTED_BRANCH)
print("Commit:", actual_commit)
print("Teacher:", teacher_target)
print("Student:", student_target)
print("Kaggle tests: skipped")
print("Run:")
print(run_command)
