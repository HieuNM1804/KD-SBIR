"""Attention heatmap visualization for KD-SBIR retrieval.

Extracts CLS→patch attention from the student (CLIP ViT-B/32) and
optionally the teacher (DFN5B ViT-H/14) using forward hooks on
``nn.MultiheadAttention``.  No model code is modified.

The retrieval pipeline encodes all unseen sketches and photos, ranks
photos by cosine similarity for each query sketch, and renders
composite heatmap images comparing *base CLIP* (no prompts) against
the *KD-prompted student*.

Usage on Kaggle (after training)::

    python -m src.visualize_attention \\
        --root /kaggle/input/datasets/.../Sketchy \\
        --dataset sketchy_2 \\
        --ckpt_path saved_models/.../last.ckpt \\
        --output_dir /kaggle/working/heatmaps \\
        --sketches_per_class 5 \\
        --top_k 10
"""

import argparse
import os
import shutil
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import glob
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from clip import clip
from clip.model import build_model
from src.dataset import normal_transform
from src.data_config import UNSEEN_CLASSES
from src.model import (
    _load_clip_model,
    IndependentVisualPromptLearner,
    freeze_clip,
)

# ── Attention extraction ──────────────────────────────────────────────


class AttentionCapture:
    """Context manager that captures attention weights from ViT blocks.

    Uses PyTorch forward hooks on ``nn.MultiheadAttention`` to force
    ``need_weights=True`` and collect the resulting weight matrices
    without touching any model source code.

    Requires PyTorch ≥ 2.0 for ``register_forward_pre_hook(with_kwargs=True)``.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.attentions = []
        self._hooks = []

    def __enter__(self):
        self.attentions.clear()
        self._hooks.clear()

        for block in self.blocks:
            attn_module = block.attn
            attn_store = self.attentions

            def _make_pre_hook():
                def pre_hook(_module, args, kwargs):
                    kwargs["need_weights"] = True
                    return args, kwargs

                return pre_hook

            def _make_post_hook(store):
                def post_hook(_module, _input, output):
                    if isinstance(output, tuple) and len(output) > 1:
                        weights = output[1]
                        if weights is not None:
                            store.append(weights.detach().cpu())

                return post_hook

            h1 = attn_module.register_forward_pre_hook(
                _make_pre_hook(), with_kwargs=True
            )
            h2 = attn_module.register_forward_hook(
                _make_post_hook(attn_store)
            )
            self._hooks.extend([h1, h2])

        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False

    def spatial_attention(self, grid_size, method="rollout"):
        """Extract CLS→patch spatial attention map.

        Args:
            grid_size: spatial grid side length (e.g. 7 for ViT-B/32).
            method: ``'last'`` (last-layer only) or ``'rollout'``
                (attention rollout across all layers).

        Returns:
            numpy array of shape ``[grid_size, grid_size]`` in ``[0, 1]``.
        """
        patch_count = grid_size * grid_size

        if method == "last":
            attn = self.attentions[-1]  # [batch, tgt_seq, src_seq]
            cls_attn = attn[0, 0, 1 : 1 + patch_count].float()

        elif method == "rollout":
            result = None
            for attn in self.attentions:
                a = attn[0].float()  # [tgt_seq, src_seq]
                identity = torch.eye(a.shape[0])
                a = 0.5 * a + 0.5 * identity
                a = a / a.sum(dim=-1, keepdim=True)
                result = a if result is None else result @ a
            cls_attn = result[0, 1 : 1 + patch_count]

        else:
            raise ValueError(f"Unknown attention method: {method!r}")

        attn_map = cls_attn.numpy().reshape(grid_size, grid_size)
        lo, hi = attn_map.min(), attn_map.max()
        if hi - lo > 1e-8:
            attn_map = (attn_map - lo) / (hi - lo)
        else:
            attn_map = np.zeros_like(attn_map)

        return attn_map


# ── Heatmap overlay ──────────────────────────────────────────────────


def overlay_heatmap(image, attn_map, alpha=0.55, cmap_name="jet"):
    """Blend a spatial attention map onto a PIL image as a colour heatmap.

    Higher attention regions appear as yellow/red, lower regions as blue.
    """
    img = np.asarray(image, dtype=np.float32) / 255.0
    h, w = img.shape[:2]

    # Bicubic upsample from grid to image resolution.
    t = torch.from_numpy(attn_map)[None, None].float()
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
    heat = t[0, 0].numpy()

    lo, hi = heat.min(), heat.max()
    if hi - lo > 1e-8:
        heat = (heat - lo) / (hi - lo)

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(heat)[:, :, :3].astype(np.float32)

    blended = (1.0 - alpha) * img + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── Attention extractors ─────────────────────────────────────────────


def student_attention(clip_model, tensor, prompt_learner, grid_size, method):
    """Extract spatial attention from the student CLIP visual encoder."""
    blocks = list(clip_model.visual.transformer.resblocks)

    prompt, compounds = (None, [])
    if prompt_learner is not None:
        ctx, comps = prompt_learner()
        prompt, compounds = ctx, comps

    with torch.no_grad(), AttentionCapture(blocks) as cap:
        clip_model.visual(tensor.type(clip_model.dtype), prompt, compounds)
        return cap.spatial_attention(grid_size, method)


# ── Feature extraction helpers ───────────────────────────────────────


@torch.no_grad()
def encode_all_images(clip_model, paths, prompt_learner, transform,
                      batch_size, device, modality_label="image"):
    """Encode a list of image paths into L2-normalised features."""
    dtype = clip_model.dtype
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(transform(img.convert("RGB")))
        tensor = torch.stack(images).to(device=device, dtype=dtype)

        prompt, compounds = (None, [])
        if prompt_learner is not None:
            ctx, comps = prompt_learner()
            prompt, compounds = ctx, comps

        feat = clip_model.visual(tensor, prompt, compounds)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu().float())

    return torch.cat(features, dim=0)


# ── Composite figure builder ─────────────────────────────────────────


def build_composite(
    sketch_img,
    photo_imgs,
    sketch_base_heat,
    sketch_kd_heat,
    photo_base_heats,
    photo_kd_heats,
    class_name,
    sketch_index,
    similarities,
    alpha=0.55,
):
    """Build a 3-row composite: Raw / Base CLIP heatmap / KD Student heatmap.

    Columns: [query sketch, top-1 photo, top-2 photo, …, top-K photo].
    """
    n_cols = 1 + len(photo_imgs)
    cell_size = 2.2
    label_width = 1.8

    fig_width = label_width + cell_size * n_cols
    fig_height = cell_size * 3 + 0.5  # 3 rows + title

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    fig.suptitle(
        f"{class_name.replace('_', ' ')}  —  sketch #{sketch_index + 1}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(
        3,
        n_cols + 1,
        width_ratios=[label_width / cell_size] + [1] * n_cols,
        wspace=0.04,
        hspace=0.08,
        left=0.01,
        right=0.99,
        top=0.92,
        bottom=0.01,
    )

    row_labels = ["Raw image", "Base CLIP\n(no prompts)", "KD Student\n(with prompts)"]
    row_colors = ["#222222", "#d32f2f", "#2e7d32"]

    # Combine all images and heatmaps into rows.
    all_raw = [sketch_img] + photo_imgs
    all_base = [overlay_heatmap(sketch_img, sketch_base_heat, alpha)] + [
        overlay_heatmap(p, h, alpha) for p, h in zip(photo_imgs, photo_base_heats)
    ]
    all_kd = [overlay_heatmap(sketch_img, sketch_kd_heat, alpha)] + [
        overlay_heatmap(p, h, alpha) for p, h in zip(photo_imgs, photo_kd_heats)
    ]
    rows = [all_raw, all_base, all_kd]

    for row_idx in range(3):
        # Row label on the left.
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.95,
            0.5,
            row_labels[row_idx],
            transform=ax_label.transAxes,
            fontsize=9,
            fontweight="bold",
            color=row_colors[row_idx],
            ha="right",
            va="center",
            linespacing=1.4,
        )

        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.imshow(rows[row_idx][col_idx])
            ax.axis("off")

            # Column headers (only on the first row).
            if row_idx == 0:
                if col_idx == 0:
                    header = "Query"
                else:
                    sim_value = similarities[col_idx - 1]
                    header = f"#{col_idx} ({sim_value:.3f})"
                ax.set_title(header, fontsize=7, pad=2)

    return fig


# ── Main retrieval + visualisation pipeline ──────────────────────────


def run_visualisation(args):
    """Encode unseen images, retrieve, and render heatmap composites."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = normal_transform(args.max_size)

    unseen_classes = UNSEEN_CLASSES[args.dataset]

    # ── Load student model ───────────────────────────────────────────
    print(f"[Viz] Loading CLIP {args.backbone} ...")
    clip_model = _load_clip_model(args.backbone).to(device).eval()
    freeze_clip(clip_model)

    patch_size = clip_model.visual.conv1.kernel_size[0]
    grid_size = args.max_size // patch_size
    vis_width = clip_model.visual.ln_pre.normalized_shape[0]
    print(f"[Viz] Student grid {grid_size}×{grid_size} (patch {patch_size})")

    # Build prompt learners.
    prompt_depth = min(args.prompt_depth, clip_model.visual.transformer.layers)
    photo_pl = IndependentVisualPromptLearner(
        args.n_ctx_visual, vis_width, args.seed + 201, prompt_depth
    ).to(device)
    sketch_pl = IndependentVisualPromptLearner(
        args.n_ctx_visual, vis_width, args.seed + 202, prompt_depth
    ).to(device)

    # Load checkpoint.
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        print(f"[Viz] Loading checkpoint {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        for name, learner in [("photo", photo_pl), ("sketch", sketch_pl)]:
            prefix = f"model.{name}_visual_prompt."
            sub = {
                k[len(prefix) :]: v
                for k, v in sd.items()
                if k.startswith(prefix)
            }
            if sub:
                learner.load_state_dict(sub)
                print(f"[Viz]   loaded {name} prompt ({len(sub)} params)")
    else:
        print("[Viz] ⚠  No checkpoint — using random prompts (demo mode)")

    # ── Gather unseen paths ──────────────────────────────────────────
    class_sketch_paths = {}
    class_photo_paths = {}
    for cls in unseen_classes:
        sketches = sorted(glob.glob(os.path.join(args.root, "sketch", cls, "*")))
        photos = sorted(glob.glob(os.path.join(args.root, "photo", cls, "*")))
        class_sketch_paths[cls] = sketches
        class_photo_paths[cls] = photos

    all_photo_paths = []
    photo_labels = []
    for cls in unseen_classes:
        paths = class_photo_paths[cls]
        all_photo_paths.extend(paths)
        photo_labels.extend([cls] * len(paths))

    print(
        f"[Viz] Unseen classes: {len(unseen_classes)}, "
        f"total photos: {len(all_photo_paths)}"
    )

    # ── Encode all unseen photos (prompted student) ──────────────────
    print("[Viz] Encoding all unseen photos ...")
    photo_features = encode_all_images(
        clip_model,
        all_photo_paths,
        photo_pl,
        transform,
        batch_size=args.test_batch_size,
        device=device,
    )

    # ── Output directory ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_composites = 0

    for cls_idx, cls in enumerate(unseen_classes):
        sketches = class_sketch_paths[cls]
        if not sketches:
            print(f"[Viz] ⚠  No sketches for class '{cls}', skipping.")
            continue

        n_sketches = min(args.sketches_per_class, len(sketches))

        # Deterministic selection of sketches.
        rng = np.random.default_rng(args.seed + cls_idx)
        chosen_indices = rng.choice(len(sketches), size=n_sketches, replace=False)
        chosen_indices.sort()

        print(
            f"[Viz] Class {cls_idx + 1}/{len(unseen_classes)}: "
            f"'{cls}' ({n_sketches} sketches)"
        )

        class_figures = []

        for sketch_i, sketch_idx in enumerate(chosen_indices):
            sketch_path = sketches[sketch_idx]

            # Encode this sketch (prompted student).
            with Image.open(sketch_path) as sk_pil:
                sk_pil = sk_pil.convert("RGB")
            sk_display = sk_pil.resize(
                (args.max_size, args.max_size), Image.LANCZOS
            )
            sk_tensor = transform(sk_pil).unsqueeze(0).to(device)

            sk_feat = encode_all_images(
                clip_model,
                [sketch_path],
                sketch_pl,
                transform,
                batch_size=1,
                device=device,
            )

            # Cosine similarity → top-K photos.
            sims = (sk_feat @ photo_features.t()).squeeze(0)
            top_k_values, top_k_indices = sims.topk(args.top_k)
            top_k_values = top_k_values.tolist()
            top_k_indices = top_k_indices.tolist()

            # Load top-K photo images.
            photo_imgs = []
            for pi in top_k_indices:
                with Image.open(all_photo_paths[pi]) as ph_pil:
                    ph_pil = ph_pil.convert("RGB")
                photo_imgs.append(
                    ph_pil.resize(
                        (args.max_size, args.max_size), Image.LANCZOS
                    )
                )

            # ── Attention heatmaps ───────────────────────────────────
            # Sketch heatmaps.
            sketch_base_heat = student_attention(
                clip_model, sk_tensor, None, grid_size, args.method
            )
            sketch_kd_heat = student_attention(
                clip_model, sk_tensor, sketch_pl, grid_size, args.method
            )

            # Photo heatmaps (all top-K).
            photo_base_heats = []
            photo_kd_heats = []
            for pi in top_k_indices:
                with Image.open(all_photo_paths[pi]) as ph_pil:
                    ph_pil = ph_pil.convert("RGB")
                ph_tensor = transform(ph_pil).unsqueeze(0).to(device)

                photo_base_heats.append(
                    student_attention(
                        clip_model, ph_tensor, None, grid_size, args.method
                    )
                )
                photo_kd_heats.append(
                    student_attention(
                        clip_model, ph_tensor, photo_pl, grid_size, args.method
                    )
                )

            # Build composite figure.
            fig = build_composite(
                sketch_img=sk_display,
                photo_imgs=photo_imgs,
                sketch_base_heat=sketch_base_heat,
                sketch_kd_heat=sketch_kd_heat,
                photo_base_heats=photo_base_heats,
                photo_kd_heats=photo_kd_heats,
                class_name=cls,
                sketch_index=sketch_i,
                similarities=top_k_values,
                alpha=args.alpha,
            )
            class_figures.append(fig)
            total_composites += 1

        # ── Stack all sketch composites for this class vertically ────
        # Save each figure as a temporary image, then vertically stack.
        temp_imgs = []
        for fig_i, fig in enumerate(class_figures):
            temp_path = output_dir / f"_temp_{cls}_{fig_i}.png"
            fig.savefig(temp_path, dpi=120, facecolor="white")
            plt.close(fig)
            temp_imgs.append(Image.open(temp_path))

        # Vertical concatenation.
        widths = [img.width for img in temp_imgs]
        max_width = max(widths)
        total_height = sum(img.height for img in temp_imgs)

        combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in temp_imgs:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
            img.close()

        out_path = output_dir / f"{cls}.png"
        combined.save(out_path, quality=95)
        combined.close()

        # Clean up temp files.
        for fig_i in range(len(class_figures)):
            temp_path = output_dir / f"_temp_{cls}_{fig_i}.png"
            if temp_path.exists():
                temp_path.unlink()

        print(f"[Viz]   → {out_path.name}")

    print(f"[Viz] Generated {total_composites} composites across {len(unseen_classes)} classes")

    # ── ZIP output ───────────────────────────────────────────────────
    zip_path = Path(args.output_dir).with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in sorted(output_dir.glob("*.png")):
            zf.write(png, arcname=png.name)
    print(f"[Viz] ✓ Zipped → {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Attention heatmap visualization for KD-SBIR retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root containing sketch/ and photo/.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sketchy_2",
        choices=sorted(UNSEEN_CLASSES),
        help="Zero-shot split.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Student Lightning checkpoint (.ckpt) path.",
    )
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx_visual", type=int, default=3)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        type=str,
        default="rollout",
        choices=["last", "rollout"],
        help="Attention extraction method.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument(
        "--sketches_per_class",
        type=int,
        default=5,
        help="Number of query sketches per unseen class.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top retrieved photos per sketch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="heatmaps",
        help="Directory for output PNG files and ZIP archive.",
    )

    args = parser.parse_args()
    run_visualisation(args)


if __name__ == "__main__":
    main()
