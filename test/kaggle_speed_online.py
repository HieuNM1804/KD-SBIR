"""ONE ONLINE Kaggle cell: build speed benchmark source, wheels and 5 prompted model backbones."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import distribution
import glob
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

BRANCH = "experiment/inference-speed-clip-tinyclip"
SOURCE_COMMIT = "521be74b26113ec26541a2fe4209546821e2a79c"
REPOSITORY = "https://github.com/HieuNM1804/KD-SBIR.git"
WORKING = Path("/kaggle/working")
BUNDLE = WORKING / ("speed_bundle_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")) / "offline_bundle"
MODELS = BUNDLE / "models"
WHEELS = BUNDLE / "wheels"
PROJECT = BUNDLE / "KD-SBIR"
# Optional exact path to an attached OLD offline_bundle, for weight reuse.
# None: automatically reuse the sole attached speed bundle, if present.
REUSE_BUNDLE = None
TARGET_PYTHONS = ("3.12", "3.13")
TARGET_PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64")
ROOTS = ["transformers==4.57.6", "huggingface-hub==0.36.2", "safetensors==0.8.0",
         "timm==1.0.20", "ftfy", "regex", "numpy", "pillow", "tqdm"]
MODEL_SPECS = ({'key': '22m', 'kind': 'auto_pruned', 'name': 'TinyCLIP-ViT-22M-32-Text-10M', 'directory': 'tinyclip22m_student', 'filename': 'TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt', 'url': 'https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt', 'size': 114214705, 'sha256': 'fadfe0486c7eb64208d2cfe4dec08120b284a37a11dc2c63cb5dfbac0ed4f018'}, {'key': '40m', 'kind': 'huggingface', 'name': 'TinyCLIP-ViT-40M-32-Text-19M', 'directory': 'tinyclip40m_student', 'repository': 'wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M', 'revision': '886b932a36b8fa6c18a8e423a67ca21af5316af8'}, {'key': '45m', 'kind': 'auto_pruned', 'name': 'TinyCLIP-ViT-45M-32-Text-18M', 'directory': 'tinyclip45m_student', 'filename': 'TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt', 'url': 'https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt', 'size': 177369381, 'sha256': '9739fe1783f1d75eb6246a276b0865da6a965ac6b730b5ca1e02e1740e6444b7'}, {'key': '61m', 'kind': 'huggingface', 'name': 'TinyCLIP-ViT-61M-32-Text-29M', 'directory': 'tinyclip61m_student', 'repository': 'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M', 'revision': '94cbbea5c7949cfe7bdafde64bcea5e403f59852'})

def run(cmd, cwd=WORKING):
    subprocess.run(cmd, cwd=cwd, check=True)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


WORKING.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True)
WHEELS.mkdir()
run([sys.executable, "-m", "pip", "install", "-q", *ROOTS])
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
pending = [Requirement(r).name for r in ROOTS]
resolved = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in resolved or name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith(("nvidia-", "cuda-")):
        continue
    installed = distribution(name)
    resolved[name] = installed.version
    for spec in installed.requires or []:
        req = Requirement(spec)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            pending.append(req.name)
requirements = BUNDLE / "requirements.txt"
requirements.write_text("".join(f"{n}=={v}\n" for n,v in sorted(resolved.items())), encoding="utf-8")
# Resolve native pure dependencies above, then fetch binary wheels for BOTH Kaggle
# interpreter versions. Existing PyTorch/CUDA installations are deliberately excluded.
for python_version in TARGET_PYTHONS:
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
           "--python-version", python_version, "--implementation", "cp",
           "--abi", "cp" + python_version.replace(".", ""), "--abi", "abi3", "--abi", "none",
           "--dest", str(WHEELS), "-r", str(requirements)]
    for platform_tag in TARGET_PLATFORMS:
        cmd.extend(["--platform", platform_tag])
    run(cmd)
# Reuse only the exact model files whose hashes were recorded in an attached bundle.
reuse_manifests = [Path(REUSE_BUNDLE)/"bundle_manifest.json"] if REUSE_BUNDLE else sorted({
    Path(path) for depth in range(1, 6)
    for path in glob.glob("/kaggle/input/" + "*/" * depth + "offline_bundle/bundle_manifest.json")
})
reusable = []
for manifest in reuse_manifests:
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("task") == "clip_tinyclip_speed":
        reusable.append((manifest.parent, info))
if len(reusable) > 1:
    raise RuntimeError("Multiple old bundles attached; set REUSE_BUNDLE to one exact offline_bundle folder.")
if reusable:
    import shutil
    previous, info = reusable[0]
    for item in info["files"]:
        relative = Path(item["path"])
        if relative.parts[0] != "models":
            continue
        source = (previous/relative).resolve()
        if not source.is_relative_to(previous.resolve()):
            raise ValueError("Invalid old bundle path")
        if sha(source) != item["sha256"]:
            raise RuntimeError(f"Old model checksum mismatch: {source}")
        target = BUNDLE/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print("Reused model weights from:", previous, flush=True)
run(["git", "clone", "--single-branch", "--branch", BRANCH, REPOSITORY, str(PROJECT)])
run(["git", "checkout", "--detach", SOURCE_COMMIT], PROJECT)
assert subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=PROJECT,text=True).strip()==SOURCE_COMMIT
from huggingface_hub import snapshot_download
for index, spec in enumerate(MODEL_SPECS,1):
    target=MODELS/spec["directory"]
    target.mkdir(exist_ok=True)
    if spec["kind"]=="huggingface":
        if not all((target/name).is_file() for name in ("config.json", "model.safetensors")):
            snapshot_download(repo_id=spec["repository"],revision=spec["revision"],
                          allow_patterns=["config.json","model.safetensors"],local_dir=target)
    else:
        path=target/spec["filename"]
        if not path.is_file():
            urllib.request.urlretrieve(spec["url"],path)
        assert path.stat().st_size==spec["size"] and sha(path)==spec["sha256"]
    print(f"Ready {index}/5: {spec['name']}",flush=True)
sys.path.insert(0,str(PROJECT))
from clip.clip import download_model
import shutil
if not (MODELS/"ViT-B-32.pt").is_file():
    shutil.copy2(download_model("ViT-B/32"),MODELS/"ViT-B-32.pt")
assert sha(MODELS/"ViT-B-32.pt")=="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
files = []
for directory in (MODELS,WHEELS):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(dict(path=path.relative_to(BUNDLE).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
(BUNDLE/"bundle_manifest.json").write_text(json.dumps(dict(task="clip_tinyclip_speed",branch=BRANCH,
    commit=SOURCE_COMMIT,python=list(sys.version_info[:2]),wheel_python_targets=list(TARGET_PYTHONS),
    runtime_requirements=ROOTS,files=files),indent=2),encoding="utf-8")
print("BUNDLE:",BUNDLE)
print("Save Version -> Save & Run All -> Always save output. Attach output to offline GPU notebook.")
