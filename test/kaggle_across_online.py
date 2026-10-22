"""Build the pinned across-dataset ZS-SBIR bundle on online Kaggle.

Run this file as one Internet-enabled Kaggle cell. Save a notebook version
with output, then attach that output and the source/target datasets to the
offline GPU notebook. The bundle supports both Sketchy-1 -> TU-Berlin and
Sketchy-1 -> QuickDraw; datasets are not copied into the bundle.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys


WORKING = Path("/kaggle/working")
BUNDLE = WORKING / "offline_bundle"
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
CLIP_CACHE = BUNDLE / "clip_cache"
TEACHER_DIR = BUNDLE / "dfn5b_openclip"

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/across-dataset-sbir"
SOURCE_COMMIT = "670122186b6f0b3c559c6290637160f6b9ad6d3f"
BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
TASK = "across_dataset_zs_sbir"
ENTRYPOINT = "src.train"
SOURCE_DATASET = "sketchy_1"
TARGET_DATASETS = ("tuberlin", "quickdraw")

TEACHER_MODEL = "ViT-H-14-quickgelu"
TEACHER_PRETRAINED = "dfn5b"
TEACHER_REPOSITORY = "apple/DFN5B-CLIP-ViT-H-14"
TEACHER_FILENAME = "open_clip_pytorch_model.bin"
TEACHER_REVISION = "11738501a1db6d5e0a3451a71ba100be02e577e6"
TEACHER_SHA256 = (
    "d67de50faa7f3ddce52fbab4f4656b046"
    "86a0bb15c26ebd0144d375cfa08b8ae"
)

STUDENT_MODEL = "ViT-B/32"
STUDENT_FILENAME = "ViT-B-32.pt"
STUDENT_SHA256 = (
    "40d365715913c9da98579312b702a82c1"
    "8be219cc2a73407c4526f58eba950af"
)
STUDENT_SIZE = 353_976_522


def run(command, cwd=WORKING, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path, expected_size=None, expected_sha256=None):
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: {path.stat().st_size} != "
            f"{expected_size}"
        )
    actual_sha256 = sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}:\n"
            f"expected: {expected_sha256}\nactual:   {actual_sha256}"
        )
    return actual_sha256


def download_runtime_wheels(roots, python_versions, bundle, wheels, run):
    """Resolve each Python separately while treating Kaggle's torch stack as provided.

    Temporary metadata-only wheels describe the already installed torch stack to
    pip's resolver. They are NEVER installed, downloaded into wheels/, or bundled.
    Real runtime dependencies are resolved and downloaded for the target ABI.
    """
    from importlib.metadata import distribution
    import tempfile
    import zipfile
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    def provided(name):
        return (name in {"torch", "torchvision", "torchaudio", "triton"}
                or name.startswith(("nvidia-", "cuda-")))

    platforms = ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64")
    inventories = {}
    with tempfile.TemporaryDirectory(prefix="kaggle-provided-stack-") as temporary:
        stack = []
        for name in ("torch", "torchvision"):
            installed = distribution(name)
            version = installed.version
            info = f"{name}-{version}.dist-info"
            path = Path(temporary) / f"{name}-{version}-py3-none-any.whl"
            metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            for specification in installed.requires or []:
                if not provided(canonicalize_name(Requirement(specification).name)):
                    metadata += f"Requires-Dist: {specification}\n"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info + "/METADATA", metadata + "\n")
                archive.writestr(info + "/WHEEL",
                                 "Wheel-Version: 1.0\nGenerator: kaggle-bundle-resolver\n"
                                 "Root-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr(info + "/RECORD", "")
            stack.append(str(path))

        for python_version in python_versions:
            abi = "cp" + python_version.replace(".", "")
            target = ["--python-version", python_version, "--implementation", "cp",
                      "--abi", abi, "--abi", "abi3", "--abi", "none"]
            for platform in platforms:
                target.extend(["--platform", platform])
            report = bundle / f"resolution-{abi}.json"
            print(f"Resolving non-CUDA dependencies for Python {python_version}", flush=True)
            run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                 "--only-binary=:all:", "--report", str(report), *target, *roots, *stack])
            resolved = {}
            for item in json.loads(report.read_text(encoding="utf-8"))["install"]:
                metadata = item["metadata"]
                name = canonicalize_name(metadata["name"])
                if not provided(name):
                    resolved[name] = metadata["version"]
            requirements = bundle / f"requirements-{abi}.txt"
            requirements.write_text(
                "".join(f"{name}=={version}\n" for name, version in sorted(resolved.items())),
                encoding="utf-8",
            )
            run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                 *target, "--dest", str(wheels), "-r", str(requirements)])
            inventories[python_version] = resolved
    # Retain this filename for existing manifest/checksum consumers. This file
    # lists runtime roots; exact resolved inventories live in requirements-cp*.txt.
    requirements = bundle / "requirements.txt"
    requirements.write_text("\n".join(roots) + "\n", encoding="utf-8")
    return inventories


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
for directory in (WHEELS, SOURCE, CLIP_CACHE, TEACHER_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/7] Clean bundle created:", BUNDLE)

roots = [
    "open-clip-torch==3.2.0", "pytorch-lightning", "torchmetrics",
    "lightning-utilities", "huggingface-hub", "ftfy", "regex", "tensorboard",
    "packaging", "tqdm", "numpy", "pillow", "matplotlib",
]
run([sys.executable, "-m", "pip", "install", "-q", *roots])
requirements_path = BUNDLE / "requirements.txt"
wheel_inventories = download_runtime_wheels(
    roots, ("3.12", "3.13"), BUNDLE, WHEELS, run,
)
print("[2/6] Non-CUDA dependency wheels:", len(list(WHEELS.glob("*.whl"))))

project = SOURCE / "KD-SBIR"
run(
    [
        "git",
        "clone",
        "--branch",
        BRANCH,
        "--single-branch",
        REPOSITORY,
        str(project),
    ]
)
run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=project)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=project, text=True
).strip()
merge_base = subprocess.check_output(
    ["git", "merge-base", "HEAD", BASE_COMMIT], cwd=project, text=True
).strip()
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=project, text=True
).strip()
if actual_commit != SOURCE_COMMIT or merge_base != BASE_COMMIT or dirty:
    raise RuntimeError("Cloned source failed commit/base/clean-tree validation.")
print("[4/7] Source commit:", actual_commit)

download_env = os.environ.copy()
for variable in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_ENABLE_HF_TRANSFER",
):
    download_env.pop(variable, None)
download_env["HF_HUB_DISABLE_XET"] = "1"

teacher_code = f"""
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id={TEACHER_REPOSITORY!r},
    filename={TEACHER_FILENAME!r},
    revision={TEACHER_REVISION!r},
    local_dir={str(TEACHER_DIR)!r},
))
"""
run([sys.executable, "-c", teacher_code], env=download_env)
teacher_target = TEACHER_DIR / TEACHER_FILENAME
if teacher_target.is_symlink():
    copied_teacher = TEACHER_DIR / (TEACHER_FILENAME + ".copy")
    shutil.copyfile(teacher_target, copied_teacher)
    teacher_target.unlink()
    copied_teacher.replace(teacher_target)
teacher_metadata = TEACHER_DIR / ".cache"
if teacher_metadata.exists():
    shutil.rmtree(teacher_metadata)
teacher_sha256 = validate(teacher_target, expected_sha256=TEACHER_SHA256)
print(
    "[5/7] DFN5B downloaded:",
    teacher_target,
    f"({teacher_target.stat().st_size / 1024**3:.3f} GiB)",
)

# Use the repository's vendored OpenAI CLIP downloader, which is also what
# src.model uses when it restores the ViT-B/32 student offline.
for module_name in list(sys.modules):
    if module_name == "clip" or module_name.startswith("clip."):
        del sys.modules[module_name]
sys.path.insert(0, str(project))
from clip import clip as project_clip


downloaded_student = Path(project_clip.download_model(STUDENT_MODEL))
student_target = CLIP_CACHE / STUDENT_FILENAME
shutil.copy2(downloaded_student, student_target)
student_sha256 = validate(
    student_target,
    expected_size=STUDENT_SIZE,
    expected_sha256=STUDENT_SHA256,
)
print(
    "[6/7] Student ViT-B/32 downloaded:",
    student_target,
    f"({student_target.stat().st_size / 1024**2:.1f} MiB)",
)

config = {}
exec((project / "src" / "data_config.py").read_text(encoding="utf-8"), config)
cross_classes = {
    f"{source}->{target}": classes
    for (source, target), classes in config["CROSS_DATASET_CLASSES"].items()
}
expected_keys = {
    f"{SOURCE_DATASET}->{target}" for target in TARGET_DATASETS
}
if set(cross_classes) != expected_keys:
    raise RuntimeError(
        "Pinned source does not expose exactly the supported across-dataset "
        f"protocols: {sorted(cross_classes)}"
    )
if len(cross_classes["sketchy_1->tuberlin"]) != 21:
    raise RuntimeError("Sketchy-1 -> TU-Berlin must contain 21 classes.")
if len(cross_classes["sketchy_1->quickdraw"]) != 11:
    raise RuntimeError("Sketchy-1 -> QuickDraw must contain 11 classes.")

required_paths = (
    requirements_path,
    WHEELS,
    project / ".git",
    project / "clip" / "model.py",
    project / "src" / "data_config.py",
    project / "src" / "dataset.py",
    project / "src" / "losses.py",
    project / "src" / "model.py",
    project / "src" / "teacher_prompts.py",
    project / "src" / "train.py",
    project / "tests" / "test_cross_dataset_protocol.py",
    teacher_target,
    student_target,
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Across-dataset bundle is incomplete:\n" + "\n".join(missing_paths)
    )

wheel_manifest = [
    {
        "filename": wheel.name,
        "size": wheel.stat().st_size,
        "sha256": sha256(wheel),
    }
    for wheel in sorted(WHEELS.glob("*.whl"))
]
manifest = {
    "runtime_requirements": roots,
    "wheel_python_targets": ["3.12", "3.13"],
    "wheel_inventories": wheel_inventories,
    "teacher_triplet_direction": "sketch_to_photo",
    "student_kd": "symmetric_kl",
    "repository": REPOSITORY,
    "branch": BRANCH,
    "commit": actual_commit,
    "base_commit": BASE_COMMIT,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "evaluation_protocol": "across_dataset_zs_sbir",
    "source_dataset": SOURCE_DATASET,
    "target_datasets": list(TARGET_DATASETS),
    "cross_dataset_classes": cross_classes,
    "teacher_model": TEACHER_MODEL,
    "teacher_pretrained": TEACHER_PRETRAINED,
    "teacher_repository": TEACHER_REPOSITORY,
    "teacher_revision": TEACHER_REVISION,
    "teacher_filename": TEACHER_FILENAME,
    "teacher_size": teacher_target.stat().st_size,
    "teacher_sha256": teacher_sha256,
    "student_model": STUDENT_MODEL,
    "student_filename": STUDENT_FILENAME,
    "student_size": student_target.stat().st_size,
    "student_sha256": student_sha256,
    "python_major_minor": list(sys.version_info[:2]),
    "python_version": sys.version,
    "wheels": wheel_manifest,
}
manifest_path = BUNDLE / "bundle_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

bundle_size = sum(
    path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file()
)
print("[7/7] Bundle validated")
print("=" * 72)
print("ONLINE ACROSS-DATASET ZS-SBIR BUNDLE COMPLETE")
print("=" * 72)
print("Bundle:", BUNDLE)
print("Manifest:", manifest_path)
print("Branch:", BRANCH)
print("Pinned source commit:", actual_commit)
print("Protocols:", {key: len(value) for key, value in cross_classes.items()})
print("Teacher SHA256:", teacher_sha256)
print("Student SHA256:", student_sha256)
print("Wheels:", len(wheel_manifest))
print("Bundle size:", f"{bundle_size / 1024**3:.3f} GiB")
print()
print("Save Version -> Save & Run All -> Always save output.")
