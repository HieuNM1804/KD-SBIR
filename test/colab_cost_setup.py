"""ONE Colab Internet cell: clone/update the benchmark branch and install runtime."""
from pathlib import Path
import os
import runpy
import subprocess
import sys

PROJECT = Path("/content/KD-SBIR")
BRANCH = "experiment/prompted-encoder-cost"
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--single-branch", "--branch", BRANCH,
                    "https://github.com/HieuNM1804/KD-SBIR.git", str(PROJECT)], check=True)
else:
    subprocess.run(["git", "fetch", "origin", f"{BRANCH}:refs/remotes/origin/{BRANCH}"], cwd=PROJECT, check=True)
    subprocess.run(["git", "switch", BRANCH], cwd=PROJECT, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=PROJECT, check=True)
roots = runpy.run_path(str(PROJECT / "src/cost_requirements.py"))["RUNTIME_ROOTS"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *roots], check=True)
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(name, None)
os.chdir(PROJECT)
print("COLAB SETUP COMPLETE:", PROJECT, flush=True)
print("Paste the shared inference or training run cell next.")
print("Missing model weights download automatically to /content/encoder_cost_models.")
print("Training requires your dataset; set DATA_ROOT in its run cell if needed.")
