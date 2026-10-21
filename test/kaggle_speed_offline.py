"""CELL 1: restore attached benchmark bundle; run kaggle_speed_run.ipy as CELL 2."""
from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

# Set exact manifest path if multiple online notebook outputs are attached.
MANIFEST = None
# Bounded glob patterns: do not recursively walk millions of dataset image paths.
candidates = [Path(MANIFEST)] if MANIFEST else sorted({
    Path(path) for depth in range(1, 6)
    for path in glob.glob("/kaggle/input/" + "*/" * depth + "offline_bundle/bundle_manifest.json")
})
matches=[]
for path in candidates:
    try:
        data=json.loads(path.read_text())
        if data.get("task")=="clip_tinyclip_speed":
            matches.append((path,data))
    except (OSError,ValueError):
        pass
if len(matches)!=1:
    raise RuntimeError(f"Set MANIFEST to exactly one speed bundle manifest. Found: {[str(p) for p,_ in matches]}")
manifest, metadata=matches[0]
builder_python = ".".join(map(str, metadata.get("python", [])))
print(f"Bundle Python: {builder_python}; offline Python: {sys.version_info.major}.{sys.version_info.minor}", flush=True)
# Version alone does not determine compatibility: pure Python/abi3 wheels and
# compatible runtime packages already installed by Kaggle can work across versions.
bundle=manifest.parent
for item in metadata["files"]:
    path=(bundle/item["path"]).resolve()
    if not path.is_relative_to(bundle.resolve()):
        raise ValueError("Invalid bundle path")
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            h.update(chunk)
    if h.hexdigest()!=item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {path}")
install_command = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(bundle/"wheels"),
                   *metadata["runtime_requirements"]]
subprocess.run(install_command, check=True)
project=Path("/kaggle/working/KD-SBIR-speed")
if project.exists():
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project,text=True).strip()
    if actual!=metadata["commit"] or subprocess.check_output(["git","status","--porcelain"],cwd=project,text=True).strip():
        raise RuntimeError("Existing speed project differs; use a fresh Kaggle session.")
else:
    shutil.copytree(bundle/"KD-SBIR",project)
actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project,text=True).strip()
if actual!=metadata["commit"]:
    raise RuntimeError("Restored source commit differs from manifest")
# Weights stay on attached storage: avoid another local model copy.
os.environ["SPEED_MODELS_ROOT"]=str(bundle/"models")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
print("Ready:",project)
print("Run CELL 2: kaggle_speed_run.ipy")
