"""Colab online / Kaggle offline paths and pinned, missing-only model preparation."""
import os
from pathlib import Path

from src.encoder_cost_models import SPECS
from src.kaggle_cost_downloads import download_verified, matches, reuse_verified


def detect_environment():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    # Prefer /content to a stray /kaggle/working created by an old cell in Colab.
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir():
        return "colab"
    return "kaggle" if Path("/kaggle/working").is_dir() else "local"


def working_directory():
    return {"kaggle": Path("/kaggle/working"), "colab": Path("/content")}.get(
        detect_environment(), Path.cwd())


def download_mode(mode="auto"):
    if mode not in {"auto", "online", "offline"}:
        raise ValueError(mode)
    return ("online" if detect_environment() == "colab" else "offline") if mode == "auto" else mode


def models_directory(explicit=None):
    if explicit is not None:
        return Path(explicit).resolve()
    project = Path(__file__).resolve().parents[1]
    for candidate in (project / "models", Path.cwd() / "models"):
        if candidate.is_dir():
            return candidate.resolve()
    return working_directory() / "encoder_cost_models"


def dataset_directory(explicit=None):
    candidates = [Path(explicit)] if explicit else []
    if not explicit:
        if os.environ.get("KD_SBIR_DATA_ROOT"):
            candidates.append(Path(os.environ["KD_SBIR_DATA_ROOT"]))
        candidates += [Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"),
                       Path("/content/Sketchy"), Path("/content/sketchy/Sketchy"),
                       Path("/content/datasets/sketchy/Sketchy")]
    for path in candidates:
        if (path / "sketch").is_dir() and (path / "photo").is_dir():
            return path.resolve()
    raise FileNotFoundError("Dataset not found. Set DATA_ROOT in the training cell, pass --root, "
                            "or set KD_SBIR_DATA_ROOT to the folder containing sketch/ and photo/. "
                            "Datasets are not downloaded automatically.")


def model_files(key):
    spec = SPECS[key]
    files = [(spec["filename"], spec.get("sha256"), None)]
    if key == "tinyclip_61m":
        files.insert(0, ("config.json", None, spec["config_blob_id"]))
    return files


def ensure_models(keys, models_root, mode="auto"):
    """Online: verify/reuse then download missing files. Offline: local files only.

    Preparation happens before model loading and outside all inference/training
    timers. Kaggle defaults to offline, even when its Internet switch is enabled.
    """
    mode = download_mode(mode)
    root = Path(models_root)
    missing = [root / SPECS[key]["directory"] / name for key in keys
               for name, _, _ in model_files(key)
               if not (root / SPECS[key]["directory"] / name).is_file()]
    print(f"Model source: {root} (mode={mode})", flush=True)
    if mode == "offline":
        if missing:
            raise FileNotFoundError("Missing offline model files:\n" + "\n".join(map(str, missing))
                                    + "\nRun the Kaggle offline setup with its attached bundle. "
                                    "On Colab with Internet, use --download-mode online.")
        return
    # This applies to this benchmark process, allowing online Colab even if a
    # previously pasted Kaggle setup left offline flags in the notebook kernel.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(name, None)
    for key in keys:
        spec = SPECS[key]
        directory = root / spec["directory"]
        for name, expected, blob in model_files(key):
            target = directory / name
            candidates = []
            if key == "clip_b32":
                candidates.append(Path.home() / ".cache/clip/ViT-B-32.pt")
            if reuse_verified(target, candidates, expected, blob):
                continue
            print(f"Preparing original weights: {key}/{name}", flush=True)
            if spec.get("url"):
                download_verified([spec["url"], *spec.get("fallback_urls", [])], target, expected)
            else:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=spec["repository"], revision=spec["revision"],
                                filename=name, local_dir=directory, force_download=target.exists())
            if not matches(target, expected, blob):
                raise RuntimeError(f"Pinned model checksum mismatch: {key}/{name}")
