"""Compare base/student/teacher on identical retrieval pairs, with provenance."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import textwrap
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dataset import normal_transform
from src.data_config import UNSEEN_CLASSES
from src.model import _load_clip_model
from src.attention_diagnostics import (
    student_encoders,
    teacher_encoder,
    pair_attribution,
    native_attention,
    normalize_map,
)


def load_image(path, transform):
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    # Display the actual resized input, not a second Lanczos preprocessing.
    from src.dataset import CLIP_MEAN, CLIP_STD

    pixels = (
        tensor * torch.tensor(CLIP_STD)[:, None, None]
        + torch.tensor(CLIP_MEAN)[:, None, None]
    )
    display = Image.fromarray(
        (pixels.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    )
    return display, tensor[None]


def overlay_heatmap(image, raw, alpha=0.55, signed=False):
    raw = torch.as_tensor(raw).float()
    if signed:
        scale = raw.abs().max()
        heat = raw / scale if scale > 1e-12 else torch.zeros_like(raw)
    else:
        heat = torch.from_numpy(normalize_map(raw))
    heat = F.interpolate(
        heat[None, None],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    cmap = plt.get_cmap("coolwarm" if signed else "viridis")
    colors = cmap((heat + 1) / 2 if signed else heat)[:, :, :3]
    output = (1 - alpha) * np.asarray(image) / 255.0 + alpha * colors
    return Image.fromarray((output.clip(0, 1) * 255).astype(np.uint8))


@torch.no_grad()
def encode_paths(encoder, paths, transform, batch_size):
    result = []
    for start in range(0, len(paths), batch_size):
        images = torch.cat(
            [load_image(p, transform)[1] for p in paths[start : start + batch_size]]
        )
        result.append(encoder.encode(images, "photo").cpu())
    return torch.cat(result)


def build_composite(sketch, photos, maps, names, scores, ranks, title, method, alpha):
    # Every photo column has its own query map: pair attribution must not
    # reuse a single sketch heatmap across different retrieved photos.
    rows = 1 + len(names)
    fig, axes = plt.subplots(
        rows, len(photos), squeeze=False, figsize=(4 * len(photos) + 2, 2.4 * rows)
    )
    for j, photo in enumerate(photos):
        canvas = np.concatenate([np.asarray(sketch), np.asarray(photo)], axis=1)
        axes[0, j].imshow(canvas)
        axes[0, j].set_title(f"Student rank #{ranks[j]} | sketch / photo", fontsize=9)
        for i, name in enumerate(names):
            left, right = maps[i][j]
            canvas = np.concatenate(
                [
                    np.asarray(
                        overlay_heatmap(sketch, left, alpha, method == "pair_grad")
                    ),
                    np.asarray(
                        overlay_heatmap(photo, right, alpha, method == "pair_grad")
                    ),
                ],
                axis=1,
            )
            axes[i + 1, j].imshow(canvas)
            axes[i + 1, j].set_title(f"cos={scores[i][j]:.4f}", fontsize=9)
    for i, label in enumerate(["Raw"] + names):
        axes[i, 0].set_ylabel(label, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    meaning = (
        "Signed token gradient x activation for pair cosine (red positive / blue negative); NOT native attention"
        if method == "pair_grad"
        else "Per-image CLS attention; NOT conditioned on retrieval partner"
    )
    wrap_width = max(55, 35 * len(photos))
    caption = "Each map normalized independently; color intensity is not comparable across models"
    fig.suptitle(
        title
        + "\n"
        + textwrap.fill(meaning, wrap_width)
        + "\n"
        + textwrap.fill(caption, wrap_width),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def run_visualisation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a new --output_dir; previous figures will not be overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    transform = normal_transform(args.max_size)
    model = _load_clip_model(args.backbone).to(device).eval()
    base, student, student_info = student_encoders(model, args.ckpt_path, args.seed)
    if base.size != args.max_size:
        raise ValueError("Input size must match the encoder positional grid")
    encoders = [base, student]
    names = ["Base CLIP", "KD student"]
    teacher_info = None
    if not args.student_only:
        teacher, teacher_info = teacher_encoder(
            args.teacher_cache_path, args.teacher_mode, args.dataset, device
        )
        if teacher.size != args.max_size:
            raise ValueError("Teacher input resolution mismatch")
        encoders.append(teacher)
        names.append("Teacher DFN5B (" + args.teacher_mode + ")")
    print("[Viz] Models:", names)
    print("[Viz] Student checkpoint:", args.ckpt_path)
    print("[Viz] Teacher cache:", teacher_info)
    print("[Viz] Method:", args.method, "; gallery ranked ONLY by prompted student")
    classes = UNSEEN_CLASSES[args.dataset]
    if args.classes and not set(args.classes) <= set(classes):
        raise ValueError("--classes must be a subset of the chosen unseen split")
    render_classes = args.classes or classes
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def paths(modality, cls):
        return sorted(
            str(p)
            for p in (Path(args.root) / modality / cls).glob("*")
            if p.suffix.lower() in extensions
        )

    gallery = [p for cls in classes for p in paths("photo", cls)]
    if not gallery:
        raise ValueError("No gallery photos found")
    features = encode_paths(student, gallery, transform, args.test_batch_size)
    records = []
    for ci, cls in enumerate(classes):
        if cls not in render_classes:
            continue
        sketches = paths("sketch", cls)
        chosen = np.random.default_rng(args.seed + ci).choice(
            len(sketches), min(len(sketches), args.sketches_per_class), replace=False
        )
        for qi, index in enumerate(sorted(chosen)):
            sketch_path = sketches[index]
            sketch, st = load_image(sketch_path, transform)
            with torch.no_grad():
                similarities = (student.encode(st, "sketch").cpu() @ features.t())[0]
            indices = similarities.topk(min(args.top_k, len(gallery))).indices.tolist()
            for start in range(0, len(indices), args.pairs_per_figure):
                subset = indices[start : start + args.pairs_per_figure]
                photos = []
                maps = [[] for _ in encoders]
                values = [[] for _ in encoders]
                arrays = {}
                pair_records = []
                for j, pi in enumerate(subset):
                    photo, pt = load_image(gallery[pi], transform)
                    photos.append(photo)
                    record = {
                        "sketch": sketch_path,
                        "photo": gallery[pi],
                        "student_rank": start + j + 1,
                        "ranking_cosine": float(similarities[pi]),
                        "models": {},
                    }
                    for mi, (name, encoder) in enumerate(zip(names, encoders)):
                        if args.method == "pair_grad":
                            pair, score = pair_attribution(encoder, st, pt)
                        else:
                            pair = [
                                native_attention(encoder, st, "sketch", args.method),
                                native_attention(encoder, pt, "photo", args.method),
                            ]
                            with torch.no_grad():
                                score = float(
                                    (
                                        encoder.encode(st, "sketch")
                                        * encoder.encode(pt, "photo")
                                    ).sum()
                                )
                        if not all(torch.isfinite(m).all() for m in pair):
                            raise RuntimeError("Nonfinite diagnostic map")
                        maps[mi].append(pair)
                        values[mi].append(score)
                        record["models"][name] = {
                            "cosine": score,
                            "grid": encoder.grid,
                            "map_minmax": [
                                [float(m.min()), float(m.max())] for m in pair
                            ],
                        }
                        for side, m in zip(("sketch", "photo"), pair):
                            arrays[f"pair{j}_model{mi}_{side}"] = m.numpy()
                    pair_records.append(record)
                stem = f"{cls}_sketch{qi+1}_ranks{start+1}-{start+len(subset)}"
                fig = build_composite(
                    sketch,
                    photos,
                    maps,
                    names,
                    values,
                    list(range(start + 1, start + 1 + len(subset))),
                    f"{cls} / sketch {qi+1}",
                    args.method,
                    args.alpha,
                )
                fig.savefig(output / (stem + ".png"), dpi=120)
                plt.close(fig)
                np.savez_compressed(output / (stem + ".npz"), **arrays)
                for record in pair_records:
                    record["figure"] = stem + ".png"
                records.extend(pair_records)
            print(
                f"[Viz] {cls} sketch {qi+1}: {len(indices)} pairs, {len(encoders)} models"
            )
    import open_clip

    manifest = {
        "method": args.method,
        "ranking_model": "KD student",
        "normalization": "per-map",
        "student": student_info,
        "teacher": teacher_info,
        "arguments": vars(args),
        "versions": {
            "torch": str(torch.__version__),
            "open_clip": getattr(open_clip, "__version__", "unknown"),
        },
        "pairs": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(str(output) + ".zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.iterdir()):
            archive.write(p, arcname=p.name)
    print("[Viz] Saved", str(output) + ".zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--dataset", choices=sorted(UNSEEN_CLASSES), default="sketchy_2"
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--backbone", default="ViT-B/32")
    parser.add_argument("--teacher_cache_path", default="")
    parser.add_argument("--teacher_mode", choices=["tuned", "raw"], default="tuned")
    parser.add_argument("--student_only", action="store_true")
    parser.add_argument(
        "--method", choices=["pair_grad", "last", "rollout"], default="pair_grad"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--sketches_per_class", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--pairs_per_figure", type=int, default=5)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--output_dir", default="/kaggle/working/pair_heatmaps")
    args = parser.parse_args()
    if (
        min(
            args.test_batch_size,
            args.sketches_per_class,
            args.top_k,
            args.pairs_per_figure,
        )
        < 1
        or not 0 <= args.alpha <= 1
    ):
        parser.error("Counts must be positive and alpha must lie in [0,1]")
    if (
        not args.student_only
        and args.teacher_mode == "tuned"
        and not args.teacher_cache_path
    ):
        parser.error(
            "Specify --teacher_cache_path explicitly, or choose --teacher_mode raw / --student_only"
        )
    run_visualisation(args)


if __name__ == "__main__":
    main()
