"""Run as one online Kaggle cell; save its output for offline ensemble training."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
BRANCH = "experiment/clip-kd-teacher-ensemble4"
BASE_COMMIT = "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6"
COMMIT = "d218b11a8bdc7f16d4d697a7157b4cece4e4d034"
TASK = "clip_kd_teacher_ensemble4"
DATASET = "b20dccn616nguynhutun/sketchy"
ENTRYPOINT = "src.train"
STUDENT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
RUNTIME_REQUIREMENTS = [
    "open-clip-torch==3.2.0", "pytorch-lightning", "torchmetrics",
    "lightning-utilities", "huggingface-hub", "safetensors", "ftfy", "regex",
    "tensorboard", "packaging", "tqdm", "numpy", "pillow", "matplotlib",
]
WORKING = Path("/kaggle/working")
BUNDLE = WORKING / ("ensemble4_bundle_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")) / "offline_bundle"
WHEELS = BUNDLE / "wheels"
PROJECT = BUNDLE / "source" / "KD-SBIR"


def run(command, cwd=WORKING):
    subprocess.run(command, cwd=cwd, check=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
for directory in (WHEELS, PROJECT.parent, BUNDLE / "clip_cache", BUNDLE / "teachers"):
    directory.mkdir(parents=True, exist_ok=True)
print("[1/6] New ensemble4 bundle:", BUNDLE)
run([sys.executable, "-m", "pip", "install", "-q", *RUNTIME_REQUIREMENTS])

from importlib.metadata import distribution
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pending = [Requirement(spec).name for spec in RUNTIME_REQUIREMENTS]
resolved = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in resolved or name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith(("nvidia-", "cuda-")):
        continue
    installed = distribution(name)
    resolved[name] = installed.version
    for specification in installed.requires or []:
        dependency = Requirement(specification)
        if dependency.marker is None or dependency.marker.evaluate({"extra": ""}):
            pending.append(dependency.name)
requirements = BUNDLE / "requirements.txt"
requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(resolved.items())), encoding="utf-8")
run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
     "--dest", str(WHEELS), "-r", str(requirements)])
print("[2/6] Non-CUDA dependency wheels:", len(list(WHEELS.glob("*.whl"))))
run(["git", "clone", "--branch", BRANCH, "--single-branch", REPOSITORY, str(PROJECT)])
run(["git", "checkout", "--detach", COMMIT], PROJECT)
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip() != COMMIT:
    raise RuntimeError("Source commit mismatch")
if subprocess.check_output(["git", "merge-base", COMMIT, BASE_COMMIT], cwd=PROJECT, text=True).strip() != BASE_COMMIT:
    raise RuntimeError("Source is not based on the expected main commit")
run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_teacher_ensemble.py", "-v"], PROJECT)
run([sys.executable, "-m", ENTRYPOINT, "--help"], PROJECT)
print("[3/6] Source checked:", COMMIT)

sys.path.insert(0, str(PROJECT))
from src.teacher_ensemble import TEACHERS
from huggingface_hub import hf_hub_download
import open_clip

teachers = []
for key, spec in TEACHERS.items():
    cfg = open_clip.get_pretrained_cfg(spec["model"], spec["pretrained"])
    if cfg.get("hf_hub", "").rstrip("/") != spec["repo"]:
        raise RuntimeError(f"OpenCLIP checkpoint mapping changed for {key}: {cfg}")
    # Direct download to bundle avoids a second multi-GB copy in /kaggle/working.
    target = Path(hf_hub_download(spec["repo"], filename=spec["filename"],
                                 revision=spec["revision"], local_dir=BUNDLE / "teachers" / key))
    if target.is_symlink():
        temporary = target.with_name(target.name + ".copy")
        shutil.copyfile(target, temporary)
        target.unlink()
        temporary.replace(target)
    if file_sha256(target) != spec["sha256"]:
        raise RuntimeError(f"Checkpoint checksum mismatch: {key}")
    teachers.append({"key": key, **spec, "size": target.stat().st_size})
    print("[4/6] Teacher downloaded:", key, f"{target.stat().st_size / 1024**3:.3f} GiB")

for name in list(sys.modules):
    if name == "clip" or name.startswith("clip."):
        del sys.modules[name]
from clip import clip as project_clip

student = BUNDLE / "clip_cache" / "ViT-B-32.pt"
shutil.copy2(project_clip.download_model("ViT-B/32"), student)
if file_sha256(student) != STUDENT_SHA256:
    raise RuntimeError("Student checkpoint checksum mismatch")
print("[5/6] Student checkpoint downloaded:", student)
manifest = {
    "repository": REPOSITORY, "branch": BRANCH, "commit": COMMIT,
    "base_commit": BASE_COMMIT, "task": TASK, "dataset": DATASET, "entrypoint": ENTRYPOINT,
    "teachers": teachers, "student_filename": student.name,
    "student_sha256": STUDENT_SHA256, "student_size": student.stat().st_size,
    "python_version": sys.version, "python_major_minor": list(sys.version_info[:2]),
    "runtime_requirements": RUNTIME_REQUIREMENTS, "requirements_sha256": file_sha256(requirements),
    "wheels": [{"filename": p.name, "size": p.stat().st_size, "sha256": file_sha256(p)}
               for p in sorted(WHEELS.glob("*.whl"))],
}
(BUNDLE / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
print("[6/6] ONLINE ENSEMBLE4 BUNDLE COMPLETE")
print("Bundle:", BUNDLE, f"({size / 1024**3:.3f} GiB)")
print("Branch:", BRANCH, "Commit:", COMMIT)
print("Save Version -> Save & Run All -> Always save output.")
print("Attach this output and Sketchy to the GPU notebook; run kaggle_offline.py next.")
