"""Check prompt localization on existing seen targets before full experiments.

Use the source from experiment/sgcd-native-prompt-learning. The probe restores
prompts after its temporary updates and writes separate JSON/CSV/PNG files.
"""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
TEACHER_CACHE = "/kaggle/working/teacher_cache/sketchy1_teacher_1ep.pt"
TARGET_CACHE = "/kaggle/working/teacher_cache/sketchy1_pcsgcd_pairwise_s42_k4_m10.pt"
OUT = Path("/kaggle/working") / (
    "native_prompt_preflight_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
)

if not (PROJECT / "src/stroke_prompt_probe_cli.py").is_file():
    raise RuntimeError(
        "Install the source from experiment/sgcd-native-prompt-learning first"
    )

subprocess.run(
    [
        sys.executable,
        "-u",
        "-m",
        "src.stroke_prompt_probe_cli",
        "--root",
        ROOT,
        "--teacher-cache",
        TEACHER_CACHE,
        "--target-cache",
        TARGET_CACHE,
        "--samples",
        "16",
        "--steps",
        "60",
        "--optimizer",
        "sgd",
        "--lr",
        "0.01",
        "--out",
        str(OUT),
    ],
    cwd=PROJECT,
    check=True,
)

from IPython.display import FileLink, Image, display

for name in ("learning.png", "evidence.png"):
    display(Image(filename=str(OUT / name)))
for path in sorted(OUT.iterdir()):
    if path.is_file():
        display(FileLink(str(path)))
print(
    "Send summary.json, localization.csv, prompt_gradients.csv and evidence.png:", OUT
)
