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

for label, path in (("teacher", TEACHER_CACHE), ("target", TARGET_CACHE)):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Missing {label} cache: {path}. Attach or copy the cache before preflight."
        )

command = [
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
]
process = subprocess.Popen(
    command,
    cwd=PROJECT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)
assert process.stdout is not None
for line in process.stdout:
    print(line.rstrip(), flush=True)
code = process.wait()
if code:
    raise RuntimeError(f"Prompt preflight failed with exit code {code}; output: {OUT}")

from IPython.display import FileLink, Image, display

for name in ("learning.png", "evidence.png"):
    display(Image(filename=str(OUT / name)))
for path in sorted(OUT.iterdir()):
    if path.is_file():
        display(FileLink(str(path)))
print(
    "Send summary.json, localization.csv, prompt_gradients.csv and evidence.png:", OUT
)
