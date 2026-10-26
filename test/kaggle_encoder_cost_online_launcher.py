"""ONE ONLINE Kaggle cell: fetch and run the current fixed online setup."""
from pathlib import Path
import runpy
import subprocess
import tempfile

with tempfile.TemporaryDirectory(prefix="kd-sbir-cost-setup-") as temporary:
    source = Path(temporary) / "KD-SBIR"
    subprocess.run([
        "git", "clone", "--depth", "1", "--single-branch",
        "--branch", "experiment/prompted-encoder-cost",
        "https://github.com/HieuNM1804/KD-SBIR.git", str(source),
    ], check=True)
    runpy.run_path(str(source / "test/kaggle_encoder_cost_online.py"), run_name="__main__")
