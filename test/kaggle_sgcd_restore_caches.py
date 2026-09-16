"""Restore validated SGCD cache files from attached Kaggle datasets."""

import glob
import hashlib
import shutil
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
CACHE_ROOT = Path("/kaggle/working/teacher_cache")
CACHE_NAMES = (
    "sketchy1_teacher_1ep.pt",
    "sketchy1_pcsgcd_pairwise_s42_k4_m10.pt",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CACHE_ROOT.mkdir(parents=True, exist_ok=True)
missing = []
for name in CACHE_NAMES:
    target = CACHE_ROOT / name
    if target.is_file():
        print("Cache already present:", target, sha256(target))
        continue

    candidates = sorted(
        {
            Path(path)
            for path in glob.glob(str(INPUT_ROOT / "**" / name), recursive=True)
            if Path(path).is_file()
        }
    )
    if not candidates:
        missing.append(name)
        continue

    candidates_by_hash = {}
    for candidate in candidates:
        candidates_by_hash.setdefault(sha256(candidate), []).append(candidate)
    if len(candidates_by_hash) != 1:
        details = "\n".join(
            f"{digest}: {', '.join(str(path) for path in paths)}"
            for digest, paths in candidates_by_hash.items()
        )
        raise RuntimeError(
            f"Attached datasets contain conflicting copies of {name}:\n{details}"
        )

    digest, paths = next(iter(candidates_by_hash.items()))
    shutil.copy2(paths[0], target)
    print("Restored:", paths[0], "->", target, digest)

if missing:
    raise FileNotFoundError(
        "Missing attached SGCD caches: "
        + ", ".join(missing)
        + ". Attach the saved cache dataset, or run kaggle_sgcd_audit.ipy "
        "and then kaggle_sgcd_prepare.ipy to regenerate them."
    )

print("SGCD teacher and pairwise target caches are ready:", CACHE_ROOT)
