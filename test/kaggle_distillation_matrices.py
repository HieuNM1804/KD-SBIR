"""Paste this entire file into ONE Kaggle cell after ensemble4 training.

Uses a trained student checkpoint and existing teacher feature caches only.
Defaults match the README training command. No downloads or teacher training.
"""

from pathlib import Path
from types import SimpleNamespace
from itertools import islice
import copy
import json
import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

# EDIT: select the student checkpoint you want to inspect (best or last).
PROJECT = Path("/kaggle/working/KD-SBIR")
CHECKPOINT_PATH = PROJECT / "saved_models/ensemble4_sketchy2/last.ckpt"
ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
DATASET = "sketchy_2"
CACHE_DIR = Path("/kaggle/working/teacher_cache")
# Optional explicit paths, e.g. {"dfn5b": "/kaggle/working/teacher_cache/...pt"}.
# Metadata is checked even when paths are provided explicitly.
CACHE_PATHS = {}
TEACHER_MODELS = ["dfn5b", "openai_l14", "dfn2b_l14", "laion_h14"]
TEACHER_WEIGHTS = [1, 1, 1, 1]  # Match the training run.
RELATION = "sketch_photo"  # Or "sketch_text" / "photo_text".
TEMPERATURES = dict(sketch_photo=0.07, sketch_text=0.02, photo_text=0.15)
MAX_SIZE = 224
SEED = 42
BATCH_SIZE = 64  # Match training for the same candidates in the softmax.
EPOCH = 0  # Zero-based sampler epoch; independent of checkpoint epoch.
BATCH_INDEX = 0  # Zero-based batch within this sampler epoch.
ENCODE_BATCH_SIZE = 16
OUTPUT_DIR = Path("/kaggle/working/distillation_visualization")

# Match ALL teacher pretraining settings to locate and validate the exact caches.
TEACHER_TRAINING = dict(
    teacher_pretrain_epochs=2, teacher_pretrain_batch_size=64,
    teacher_n_ctx_visual=10, teacher_prompt_depth=12, teacher_prompt_std=0.02,
    teacher_prompt_lr=3e-2, teacher_prompt_seed=42,
    teacher_prompt_gradient_checkpointing=True,
    teacher_momentum=0.9, teacher_weight_decay=1e-3,
    lambda_teacher_retrieval=1.5, teacher_triplet_margin=0.2,
    teacher_scheduler_step_size=5, teacher_scheduler_gamma=0.1,
)


def load_payload(path, checkpoint=False):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {'student checkpoint' if checkpoint else 'teacher cache'}: {path}")
    # mmap reads selected cache rows without copying the full feature banks.
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        # Only the user's own Lightning checkpoints may require pickle objects.
        return torch.load(path, map_location="cpu", weights_only=not checkpoint)


def restore_student(path, args, classnames, device):
    from clip.model import build_model
    from src.model import CustomCLIP

    payload = load_payload(path, checkpoint=True)
    state = dict(payload.get("state_dict", payload))
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if state and all(key.startswith("model.") for key in state):
        state = {key[6:]: value for key, value in state.items()}
    contexts = [state.get(f"{modality}_visual_prompt.ctx") for modality in ("sketch", "photo")]
    if any(value is None or value.ndim != 2 for value in contexts):
        raise ValueError("Checkpoint must contain both trained visual prompts.")
    if contexts[0].shape != contexts[1].shape:
        raise ValueError("Photo and sketch prompt shapes differ.")
    args.n_ctx_visual = contexts[0].shape[0]
    depths = []
    for modality in ("sketch", "photo"):
        prefix = f"{modality}_visual_prompt.compound_prompts."
        indices = sorted(int(key[len(prefix):]) for key in state if key.startswith(prefix))
        if indices != list(range(len(indices))):
            raise ValueError("Checkpoint has incomplete compound prompts.")
        depths.append(len(indices) + 1)
    if depths[0] != depths[1]:
        raise ValueError("Photo and sketch prompt depths differ.")
    args.prompt_depth = depths[0]
    clip_state = {key[len("clip_model."):]: value for key, value in state.items()
                  if key.startswith("clip_model.")}
    if not clip_state:
        raise ValueError("Checkpoint has no frozen CLIP backbone weights.")
    backbone = build_model(clip_state)  # Reconstruct directly; no pretrained download.
    if backbone.visual.input_resolution != args.max_size:
        raise ValueError("MAX_SIZE must match the checkpoint's image resolution.")
    if device.type == "cpu":
        backbone.float()
    model = CustomCLIP(args, backbone, classnames, teacher=None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def select_batch(dataset, batch_size, epoch, batch_index):
    from src.dataset import WorkerInvariantSampler, sample_seed

    if batch_size < 2 or epoch < 0 or batch_index < 0:
        raise ValueError("Use BATCH_SIZE >= 2 and non-negative EPOCH/BATCH_INDEX.")
    if (batch_index + 1) * batch_size > len(dataset):
        raise ValueError("Selected batch does not exist among training's complete batches.")
    sampler = WorkerInvariantSampler(dataset, dataset.seed)
    sampler.epoch = epoch
    keys = list(islice(iter(sampler), batch_index * batch_size, (batch_index + 1) * batch_size))
    sketches, photos, photo_indices, labels = [], [], [], []
    for sample_epoch, index in keys:
        sketch = dataset.all_sketches_path[index]
        category = Path(sketch).parent.name
        candidates = dataset.all_photos_path[category]
        rng = np.random.default_rng(sample_seed(dataset.seed, sample_epoch, index))
        photo = candidates[rng.integers(len(candidates))]
        sketches.append(sketch)
        photos.append(photo)
        photo_indices.append(dataset.photo_path_to_index[photo])
        labels.append(dataset.category_to_label[category])
    return [index for _, index in keys], photo_indices, sketches, photos, labels


def load_targets(args, dataset, sketch_indices, photo_indices, relation):
    from src.model import CustomCLIP, default_teacher_cache_path, _teacher_training_config
    from src.teacher_ensemble import TEACHERS

    left, right, paths = [], [], {}
    identity = dict(
        dataset=args.dataset, max_size=dataset.max_size, classnames=dataset.all_categories,
        sketch_count=len(dataset.all_sketches_path), photo_count=len(dataset.all_photo_paths),
        sketch_fingerprint=CustomCLIP._path_fingerprint(dataset.all_sketches_path, args.root),
        photo_fingerprint=CustomCLIP._path_fingerprint(dataset.all_photo_paths, args.root),
    )
    for key in args.teacher_models:
        single = copy.copy(args)
        single.teacher_models, single.teacher_ensemble_weights = [key], [1.0]
        path = Path(CACHE_PATHS.get(key) or default_teacher_cache_path(single, dataset))
        payload = load_payload(path)
        expected = dict(identity, format_version=TEACHERS[key]["cache_version"],
                        **_teacher_training_config(single))
        mismatches = [name for name, value in expected.items() if payload.get("metadata", {}).get(name) != value]
        if mismatches:
            raise ValueError(f"Cache {key} differs from this run: {mismatches}. Check TEACHER_TRAINING/ROOT/DATASET.")
        dim = TEACHERS[key]["output_dim"]
        shapes = dict(teacher_sketch_features=(len(dataset), dim),
                      teacher_photo_features=(len(dataset.all_photo_paths), dim),
                      teacher_sketch_text=(len(dataset.all_categories), dim),
                      teacher_photo_text=(len(dataset.all_categories), dim))
        for name, shape in shapes.items():
            if name not in payload or tuple(payload[name].shape) != shape:
                raise ValueError(f"Incorrect {key} cache tensor: {name}, expected {shape}")
        left.append(payload["teacher_photo_features"][photo_indices].float() if relation == "photo_text"
                    else payload["teacher_sketch_features"][sketch_indices].float())
        right.append(payload["teacher_photo_features"][photo_indices].float() if relation == "sketch_photo"
                     else payload[f"teacher_{relation.split('_')[0]}_text"].float())
        paths[key] = str(path.resolve())
        print(f"Teacher target: {key} <- {path}")
        del payload
    packed_left, packed_right = torch.cat(left, -1), torch.cat(right, -1)
    if not torch.isfinite(packed_left).all() or not torch.isfinite(packed_right).all():
        raise ValueError("Selected teacher targets contain NaN/Inf.")
    return packed_left, packed_right, paths


@torch.inference_mode()
def encode_images(model, paths, modality, size, device):
    from src.dataset import TeacherFeatureDataset
    loader = torch.utils.data.DataLoader(TeacherFeatureDataset(paths, size),
                                         batch_size=ENCODE_BATCH_SIZE, shuffle=False, num_workers=0)
    return torch.cat([model.encode_student_image(images.to(device), modality).float() for images in loader])


def main():
    if not (PROJECT / "src/teacher_ensemble.py").is_file():
        raise FileNotFoundError(f"Restore the ensemble4 project first: {PROJECT}")
    sys.path.insert(0, str(PROJECT.resolve()))
    from src.dataset import TrainDataset
    from src.losses import ensemble_teacher_log_probs
    from src.teacher_ensemble import TEACHERS, normalized_weights

    if RELATION not in TEMPERATURES or RELATION not in ("sketch_photo", "sketch_text", "photo_text"):
        raise ValueError("RELATION must be sketch_photo, sketch_text or photo_text.")
    tau = TEMPERATURES[RELATION]
    if not math.isfinite(tau) or tau <= 0 or ENCODE_BATCH_SIZE < 1:
        raise ValueError("Temperature and ENCODE_BATCH_SIZE must be positive.")
    if not TEACHER_MODELS or len(set(TEACHER_MODELS)) != len(TEACHER_MODELS) or any(k not in TEACHERS for k in TEACHER_MODELS):
        raise ValueError("TEACHER_MODELS must contain distinct catalogue keys.")
    weights = normalized_weights(TEACHER_WEIGHTS, len(TEACHER_MODELS))
    active = [(key, weight) for key, weight in zip(TEACHER_MODELS, weights) if weight > 0]
    args = SimpleNamespace(
        root=str(ROOT), dataset=DATASET, max_size=MAX_SIZE, seed=SEED,
        teacher_models=[key for key, _ in active], teacher_ensemble_weights=[w for _, w in active],
        teacher_cache_dir=str(CACHE_DIR), teacher_cache_path="", rebuild_teacher_cache=False,
        lambda_domain=3.0, lambda_modality=1.0, kd_temperature=TEMPERATURES["sketch_photo"],
        sketch_text_kd_temperature=TEMPERATURES["sketch_text"], photo_text_kd_temperature=TEMPERATURES["photo_text"],
        **TEACHER_TRAINING,
    )
    dataset = TrainDataset(args)
    si, pi, sketches, photos, labels = select_batch(dataset, BATCH_SIZE, EPOCH, BATCH_INDEX)
    target_left, target_right, cache_paths = load_targets(args, dataset, si, pi, RELATION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = restore_student(CHECKPOINT_PATH, args, dataset.all_categories, device)
    with torch.inference_mode():
        modality = "photo" if RELATION == "photo_text" else "sketch"
        student_left = encode_images(model, photos if modality == "photo" else sketches, modality, MAX_SIZE, device)
        student_right = (encode_images(model, photos, "photo", MAX_SIZE, device) if RELATION == "sketch_photo"
                         else model.get_student_text_features(modality).float())
        student_log = F.log_softmax(F.normalize(student_left, dim=-1) @ F.normalize(student_right, dim=-1).t() / tau, -1)
        teacher_log = ensemble_teacher_log_probs(target_left.to(device), target_right.to(device), args, tau)
        if not torch.isfinite(student_log).all() or not torch.isfinite(teacher_log).all():
            raise ValueError("Non-finite probabilities; check the checkpoint and caches.")
        forward_kl = F.kl_div(student_log, teacher_log.exp(), reduction="batchmean").item()
        kd = forward_kl if RELATION == "sketch_photo" else 0.5 * (
            forward_kl + (student_log.exp() * (student_log - teacher_log)).sum(-1).mean().item())
        teacher, student = teacher_log.exp().cpu().numpy(), student_log.exp().cpu().numpy()
    names = dataset.all_categories
    row_names = [f"{i}: {names[label]}" for i, label in enumerate(labels)]
    col_names = row_names if RELATION == "sketch_photo" else names
    fig, axes = plt.subplots(1, 2, figsize=(19, 9), layout="constrained")
    vmax = max(float(teacher.max()), float(student.max()))
    for axis, matrix, title in zip(axes, (teacher, student), ("Teacher ensemble", "Trained student")):
        heatmap = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        xstep, ystep = max(1, math.ceil(len(col_names) / 24)), max(1, math.ceil(len(row_names) / 32))
        axis.set_xticks(range(0, len(col_names), xstep), col_names[::xstep], rotation=90, fontsize=7)
        axis.set_yticks(range(0, len(row_names), ystep), row_names[::ystep], fontsize=7)
        axis.set(xlabel="Photo candidates" if RELATION == "sketch_photo" else "All seen text classes",
                 ylabel=f"{modality.title()} queries", title=title)
    fig.colorbar(heatmap, ax=axes, label="Row-softmax probability (shared scale)", shrink=0.8)
    fig.suptitle(f"{RELATION} | seen train batch {BATCH_INDEX}, sampler epoch {EPOCH} | tau={tau}\n"
                 f"shape={teacher.shape} | unweighted KD loss={kd:.6f}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"{RELATION}_epoch{EPOCH}_batch{BATCH_INDEX}"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    np.savez_compressed(stem.with_suffix(".npz"), teacher=teacher, student=student,
                        sketch_indices=si, photo_indices=pi, labels=labels)
    metadata = dict(checkpoint=str(CHECKPOINT_PATH.resolve()), dataset=DATASET, root=str(ROOT),
                    relation=RELATION, temperature=tau, seed=SEED, sampler_epoch=EPOCH,
                    batch_index=BATCH_INDEX, batch_size=BATCH_SIZE, teachers=args.teacher_models,
                    weights=args.teacher_ensemble_weights, caches=cache_paths,
                    teacher_training=TEACHER_TRAINING, n_ctx_visual=args.n_ctx_visual, prompt_depth=args.prompt_depth,
                    sketch_paths=sketches, photo_paths=photos, row_labels=row_names, column_labels=col_names,
                    forward_kl=forward_kl, unweighted_kd_loss=kd)
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Each row sums to 1. Multiple photos from the same class are valid positives.")
    print(f"Unweighted {RELATION} KD loss: {kd:.6f}; teacher -> student KL: {forward_kl:.6f}")
    print("Saved PNG, probability NPZ and batch metadata JSON:", stem)
    plt.show()
    return teacher, student, metadata


if __name__ == "__main__":
    main()
