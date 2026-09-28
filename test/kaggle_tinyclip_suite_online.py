"""Build an offline Kaggle bundle for raw inference with five TinyCLIP models.

Run this as one cell in an Internet-enabled Kaggle notebook, then save a
notebook version with output. No teacher checkpoint or training code is run.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request


WORKING = Path("/kaggle/working")
BUNDLE = WORKING / "offline_bundle"
WHEELS = BUNDLE / "wheels"
SOURCE = BUNDLE / "source"
MODELS = BUNDLE / "tinyclip_models"

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/inference-tinyclip-suite"
SOURCE_COMMIT = "1b091d760019a29aefed6fbb524d275876e12d1f"
BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
TASK = "tinyclip_raw_inference_suite"
ENTRYPOINT = "src.infer_tinyclip_suite"
DATASET = "b20dccn616nguynhutun/sketchy"

MODEL_SPECS = (
    {
        "key": "8m", "kind": "huggingface",
        "name": "TinyCLIP-ViT-8M-16-Text-3M",
        "directory": "tinyclip8m_student",
        "repository": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "revision": "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c",
    },
    {
        "key": "22m", "kind": "auto_pruned",
        "name": "TinyCLIP-ViT-22M-32-Text-10M",
        "directory": "tinyclip22m_student",
        "filename": "TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt",
        "url": (
            "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/"
            "checkpoints/TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt"
        ),
        "size": 114_214_705,
        "sha256": "fadfe0486c7eb64208d2cfe4dec08120b284a37a11dc2c63cb5dfbac0ed4f018",
    },
    {
        "key": "40m", "kind": "huggingface",
        "name": "TinyCLIP-ViT-40M-32-Text-19M",
        "directory": "tinyclip40m_student",
        "repository": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "revision": "886b932a36b8fa6c18a8e423a67ca21af5316af8",
    },
    {
        "key": "45m", "kind": "auto_pruned",
        "name": "TinyCLIP-ViT-45M-32-Text-18M",
        "directory": "tinyclip45m_student",
        "filename": "TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt",
        "url": (
            "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/"
            "checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt"
        ),
        "size": 177_369_381,
        "sha256": "9739fe1783f1d75eb6246a276b0865da6a965ac6b730b5ca1e02e1740e6444b7",
    },
    {
        "key": "61m", "kind": "huggingface",
        "name": "TinyCLIP-ViT-61M-32-Text-29M",
        "directory": "tinyclip61m_student",
        "repository": "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
        "revision": "94cbbea5c7949cfe7bdafde64bcea5e403f59852",
    },
)


def run(command, cwd, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
for directory in (WHEELS, SOURCE, MODELS):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/6] Clean inference bundle created:", BUNDLE)

requirements = """
transformers
huggingface-hub
safetensors
tokenizers
tqdm
packaging
"""
requirements_path = BUNDLE / "requirements.txt"
requirements_path.write_text(requirements.strip() + "\n", encoding="utf-8")
run(
    [
        sys.executable, "-m", "pip", "download", "--dest", str(WHEELS),
        "--only-binary=:all:", "-r", str(requirements_path),
    ],
    WORKING,
)
for pattern in (
    "torch-*.whl", "torchvision-*.whl", "torchaudio-*.whl", "triton-*.whl",
    "nvidia_*.whl", "cuda_*.whl",
):
    for wheel in WHEELS.glob(pattern):
        wheel.unlink()
if not list(WHEELS.glob("transformers-*.whl")):
    raise FileNotFoundError("transformers wheel was not downloaded.")
print("[2/6] Dependency wheels:", len(list(WHEELS.glob("*.whl"))))

run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"], WORKING)
print("[3/6] Online download helper installed")

project = SOURCE / "KD-SBIR"
run(
    ["git", "clone", "--branch", BRANCH, "--single-branch", REPOSITORY, str(project)],
    WORKING,
)
run(["git", "checkout", "--detach", SOURCE_COMMIT], project)
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
    raise RuntimeError("Source commit/base/clean-tree validation failed.")
print("[4/6] Source commit:", actual_commit)

download_env = os.environ.copy()
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_XET"):
    download_env.pop(key, None)
download_env["HF_HUB_DISABLE_XET"] = "1"

model_manifest = []
for index, spec in enumerate(MODEL_SPECS, start=1):
    target = MODELS / spec["directory"]
    target.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "huggingface":
        code = f"""
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id={spec['repository']!r}, revision={spec['revision']!r},
    allow_patterns=['config.json', 'model.safetensors'],
    local_dir={str(target)!r},
))
"""
        run([sys.executable, "-c", code], WORKING, env=download_env)
        metadata = target / ".cache"
        if metadata.exists():
            shutil.rmtree(metadata)
        for filename in ("config.json", "model.safetensors"):
            if not (target / filename).is_file():
                raise FileNotFoundError(f"Missing {spec['key']} file: {filename}")
    else:
        checkpoint = target / spec["filename"]
        urllib.request.urlretrieve(spec["url"], checkpoint)
        if checkpoint.stat().st_size != spec["size"] or sha256(checkpoint) != spec["sha256"]:
            raise RuntimeError(f"Checkpoint validation failed: {checkpoint}")
    files = [
        {
            "relative_path": path.relative_to(MODELS).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(target.rglob("*")) if path.is_file()
    ]
    entry = dict(spec)
    entry["files"] = files
    model_manifest.append(entry)
    print(f"[5/6] Model {index}/5 validated: {spec['name']} ({len(files)} files)")

required = (
    project / ".git",
    project / "src" / "infer_tinyclip_suite.py",
    project / "src" / "tinyclip_inference.py",
    project / "src" / "tinyclip_vendor" / "model.py",
    project / "src" / "tinyclip_vendor" / "LICENSE.upstream",
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Bundle source is incomplete:\n" + "\n".join(missing))

wheel_manifest = [
    {"filename": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
    for path in sorted(WHEELS.glob("*.whl"))
]
manifest = {
    "repository": REPOSITORY,
    "branch": BRANCH,
    "commit": actual_commit,
    "base_commit": BASE_COMMIT,
    "task": TASK,
    "entrypoint": ENTRYPOINT,
    "dataset": DATASET,
    "python_major_minor": list(sys.version_info[:2]),
    "python_version": sys.version,
    "wheels": wheel_manifest,
    "models": model_manifest,
}
(BUNDLE / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
bundle_size = sum(path.stat().st_size for path in BUNDLE.rglob("*") if path.is_file())
print("[6/6] Bundle manifest written")
print("=" * 72)
print("ONLINE FIVE-MODEL TINYCLIP INFERENCE BUNDLE COMPLETE")
print("=" * 72)
print("Bundle:", BUNDLE)
print("Branch:", BRANCH)
print("Commit:", actual_commit)
print("Models:", ", ".join(spec["key"] for spec in MODEL_SPECS))
print("Wheels:", len(wheel_manifest))
print("Bundle size:", f"{bundle_size / 1024**3:.3f} GiB")
print("Save Version -> Save & Run All -> Always save output.")
