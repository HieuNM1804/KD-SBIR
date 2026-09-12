"""Paste this entire file into ONE Kaggle cell; reuses existing weights offline."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import gc
import csv
import json
import math
import zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only edit this configuration. No downloads, training or source replacement.
PROJECT = Path("/kaggle/working/KD-SBIR")
ROOT = Path("/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy")
CKPT = PROJECT / "saved_models/heatmap_viz_sketchy2/epoch=01-precision=0.7862.ckpt"
TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/heatmap_viz_sketchy2_teacher_1ep.pt"
)
PREVIOUS_AUDIT = Path(
    "/kaggle/working/teacher_student_audit_20260912_115723/manifest.json"
)
DATASET = "sketchy_2"
N_CLASSES = 8
IMAGES_PER_CLASS = 5  # 40 sketch queries / 40 photos by default
SEED = 42
OUT_ROOT = Path("/kaggle/working")
MODEL_NAMES = ("Student", "Base_CLIP", "Teacher_raw", "Teacher_tuned")
GROUPS = ("patch", "cls", "prompt")


@torch.no_grad()
def decompose_cls_attention(module, args, kwargs, output, patch_count):
    """c[h,j] = A[h,CLS,j] * V[h,j] @ W_O[:,head].T.

    Exact MHA branch decomposition (up to floating point), before residual,
    LayerScale and MLP. Output projection bias is separate, not per head/token.
    No subtraction of tokens, masking, or changes to the model output.
    """
    if not isinstance(module, torch.nn.MultiheadAttention):
        raise TypeError("Only torch.nn.MultiheadAttention is supported")
    if (
        module.training
        or module.bias_k is not None
        or module.bias_v is not None
        or module.add_zero_attn
    ):
        raise ValueError("Expected evaluation MHA without extra bias/zero tokens")
    value = args[2] if len(args) > 2 else kwargs["value"]
    if not module.batch_first:
        value = value.transpose(0, 1)
    if value.shape[0] != 1:
        raise ValueError("Audit processes one image at a time")
    weights = output[1]
    if weights is None or weights.ndim != 4:
        raise RuntimeError("Expected per-head attention [B,H,Q,K]")
    a = weights[0, :, 0, :].float()
    embed = module.embed_dim
    heads = module.num_heads
    dim = embed // heads
    wv = (
        module.in_proj_weight[2 * embed :]
        if module.in_proj_weight is not None
        else module.v_proj_weight
    )
    bv = module.in_proj_bias[2 * embed :] if module.in_proj_bias is not None else None
    v = F.linear(value, wv, bv)[0].reshape(-1, heads, dim).transpose(0, 1).float()
    if a.shape != v.shape[:2] or a.shape[1] < patch_count + 1:
        raise RuntimeError("Attention/value token order or count mismatch")
    slices = (slice(1, 1 + patch_count), slice(0, 1), slice(1 + patch_count, None))
    vectors, norms, masses, cancellations = [], [], [], []
    # Project one head at a time: avoids allocating H x tokens x full width.
    for h in range(heads):
        projected = v[h] @ module.out_proj.weight[:, h * dim : (h + 1) * dim].float().T
        c = a[h, :, None] * projected
        n = c.norm(dim=-1)
        grouped = torch.stack([c[s].sum(0) for s in slices])
        group_norm_sums = torch.stack([n[s].sum() for s in slices])
        vectors.append(grouped.cpu())
        norms.append(n.cpu())
        masses.append(torch.stack([a[h, s].sum() for s in slices]).cpu())
        cancellations.append(
            (grouped.norm(dim=-1) / group_norm_sums.clamp_min(1e-12)).cpu()
        )
    vectors = torch.stack(vectors)
    reconstructed = vectors.sum((0, 1))
    bias = (
        module.out_proj.bias.detach().float().cpu()
        if module.out_proj.bias is not None
        else torch.zeros(embed)
    )
    actual = output[0][0, 0].detach().float().cpu()
    error = float(
        (reconstructed + bias - actual).norm() / actual.norm().clamp_min(1e-8)
    )
    if not math.isfinite(error) or error > 0.02:
        raise RuntimeError(
            f"MHA reconstruction error {error:.5g}; do not interpret these maps"
        )
    result = {
        "attention": a.cpu().numpy(),
        "contribution_norm": torch.stack(norms).numpy(),
        "group_vectors": vectors.numpy(),
        "group_mass": torch.stack(masses).numpy(),
        "cancellation_ratio": torch.stack(cancellations).numpy(),
        "out_bias": bias.numpy(),
        "actual_output": actual.numpy(),
        "relative_error": error,
    }
    if not all(np.isfinite(x).all() for x in result.values()):
        raise RuntimeError("Nonfinite decomposition")
    return result


@torch.no_grad()
def capture(encoder, tensor, modality, layers):
    saved, handles = {}, []

    def pre(module, args, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def post_for(layer):
        def post(module, args, kwargs, output):
            saved[layer] = decompose_cls_attention(
                module, args, kwargs, output, encoder.grid**2
            )

        return post

    try:
        for layer in layers:
            attn = encoder.visual.transformer.resblocks[layer - 1].attn
            handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(
                attn.register_forward_hook(post_for(layer), with_kwargs=True)
            )
        # Returns embedding from hooked execution, whose MHA path exposes weights.
        embedding = encoder.encode(tensor, modality).float().cpu().numpy()[0]
    finally:
        for handle in handles:
            handle.remove()
    if set(saved) != set(layers):
        raise RuntimeError("Missing attention layer capture")
    return embedding, saved


def metrics(query, gallery, query_labels, gallery_labels):
    """Untrained cosine diagnostic; rejects degenerate zero representations."""
    qn = np.linalg.norm(query, axis=-1, keepdims=True)
    gn = np.linalg.norm(gallery, axis=-1, keepdims=True)
    valid = qn[:, 0] > 1e-8
    vg = gn[:, 0] > 1e-8
    if not valid.any() or not vg.any():
        return {
            "valid_queries": 0,
            "valid_gallery": int(vg.sum()),
            "P5": None,
            "top1": None,
            "positive_negative_margin": None,
        }
    similarity = (query[valid] / qn[valid]) @ (gallery[vg] / gn[vg]).T
    positive = (
        np.asarray(query_labels)[valid, None] == np.asarray(gallery_labels)[None, vg]
    )
    usable = positive.any(1) & (~positive).any(1)
    if not usable.any():
        return {
            "valid_queries": 0,
            "valid_gallery": int(vg.sum()),
            "P5": None,
            "top1": None,
            "positive_negative_margin": None,
        }
    similarity, positive = similarity[usable], positive[usable]
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranked = np.take_along_axis(positive, order, axis=1)
    margin = np.where(positive, similarity, -np.inf).max(1) - np.where(
        ~positive, similarity, -np.inf
    ).max(1)
    return {
        "valid_queries": int(usable.sum()),
        "valid_gallery": int(vg.sum()),
        "P5": float(ranked[:, : min(5, ranked.shape[1])].mean()),
        "top1": float(ranked[:, 0].mean()),
        "positive_negative_margin": float(margin.mean()),
    }


def draw_map(ax, image, raw, title):
    heat = torch.as_tensor(raw).float()
    maximum = float(heat.max())
    heat = heat / maximum if maximum > 0 else torch.zeros_like(heat)
    heat = F.interpolate(
        heat[None, None],
        (image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    ax.imshow(image)
    ax.imshow(heat, cmap="inferno", vmin=0, vmax=1, alpha=0.7 * heat)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def savefig(fig, path):
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def plot_image(out, name, encoder, image, captures):
    layers = sorted(captures)
    g = encoder.grid
    fig, axes = plt.subplots(2, len(layers) + 1, figsize=(16, 6))
    for ax in axes[:, 0]:
        ax.imshow(image)
        ax.axis("off")
    axes[0, 0].set_title("Native CLS attention A")
    axes[1, 0].set_title("Sum of per-token head norms ||AVW_O||")
    arrays = {}
    for column, layer in enumerate(layers, 1):
        data = captures[layer]
        a = data["attention"][:, 1 : 1 + g * g]
        n = data["contribution_norm"][:, 1 : 1 + g * g]
        draw_map(
            axes[0, column],
            image,
            a.mean(0).reshape(g, g),
            f"L{layer} | raw max={a.mean(0).max():.3g}",
        )
        draw_map(
            axes[1, column],
            image,
            n.sum(0).reshape(g, g),
            f"L{layer} | raw max={n.sum(0).max():.3g}",
        )
        for key, value in data.items():
            arrays[f"L{layer}_{key}"] = value
    fig.suptitle(
        name
        + "\nPer-panel scaling. Contribution norm is not causal importance or final-score attribution.",
        fontsize=11,
    )
    savefig(fig, out / (name + "_overview.png"))
    np.savez_compressed(out / (name + "_raw.npz"), **arrays)
    # Show every head in middle/final sampled layers, not cherry-picked heads.
    for layer in sorted(set([layers[len(layers) // 2 - 1], layers[-1]])):
        data = captures[layer]
        heads = data["attention"].shape[0]
        rows = math.ceil(heads / 4)
        fig, axes = plt.subplots(rows, 8, figsize=(20, 2.7 * rows), squeeze=False)
        for h in range(rows * 4):
            row, col = h // 4, 2 * (h % 4)
            if h >= heads:
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")
                continue
            draw_map(
                axes[row, col],
                image,
                data["attention"][h, 1 : 1 + g * g].reshape(g, g),
                f"H{h} A | patch mass={data['group_mass'][h,0]:.2f}",
            )
            draw_map(
                axes[row, col + 1],
                image,
                data["contribution_norm"][h, 1 : 1 + g * g].reshape(g, g),
                f"H{h} ||AVW_O||",
            )
        fig.suptitle(
            f"{name} | L{layer}\nEach adjacent pair is A / contribution magnitude; each panel independently scaled"
        )
        savefig(fig, out / (name + f"_L{layer}_heads.png"))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        x = np.arange(heads)
        for j, label in enumerate(GROUPS):
            axes[0].bar(x + (j - 1) * 0.25, data["group_mass"][:, j], 0.25, label=label)
            axes[1].plot(
                x,
                np.linalg.norm(data["group_vectors"][:, j], axis=-1),
                marker=".",
                label=label,
            )
            axes[2].plot(x, data["cancellation_ratio"][:, j], marker=".", label=label)
        for ax, title in zip(
            axes,
            [
                "CLS attention probability mass",
                "Norm of group vector (not probability)",
                "||sum c|| / sum ||c||; low = cancellation",
            ],
        ):
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("head (zero based)")
            ax.legend()
        fig.suptitle(f"{name} | L{layer} | output bias excluded")
        savefig(fig, out / (name + f"_L{layer}_groups.png"))


def main():
    os.chdir(PROJECT)
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from src.model import _load_clip_model
    from src.dataset import normal_transform
    from src.data_config import UNSEEN_CLASSES
    from src.attention_diagnostics import Encoder, student_encoders, teacher_encoder
    from src.visualize_attention import load_image

    assert CKPT.is_file(), CKPT
    assert TEACHER_CACHE.is_file(), TEACHER_CACHE
    assert N_CLASSES >= 2 and IMAGES_PER_CLASS >= 2
    out = OUT_ROOT / (
        "attention_value_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(TEACHER_CACHE, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    assert meta["dataset"] == DATASET
    seen = sorted(meta["classnames"])
    assert not set(seen) & set(UNSEEN_CLASSES[DATASET])
    del payload
    rng = np.random.default_rng(SEED)

    def paths(modality, cls):
        return sorted(
            p
            for p in (ROOT / modality / cls).glob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )

    eligible = [
        c
        for c in seen
        if all(len(paths(m, c)) >= IMAGES_PER_CLASS for m in ("sketch", "photo"))
    ]
    assert len(eligible) >= N_CLASSES
    selected = sorted(rng.choice(eligible, N_CLASSES, replace=False).tolist())
    samples = []
    for modality in ("sketch", "photo"):
        for cls in selected:
            pool = paths(modality, cls)
            for i in sorted(rng.choice(len(pool), IMAGES_PER_CLASS, replace=False)):
                samples.append(
                    {"path": str(pool[i]), "modality": modality, "class": cls}
                )
    # Optional: retain the EXACT earlier unseen pairs as illustration only.
    illustration, old = [], None
    if PREVIOUS_AUDIT.is_file():
        old = json.loads(PREVIOUS_AUDIT.read_text())
        assert old["dataset"] == DATASET
        for qi, query in enumerate(old["queries"]):
            illustration.append(
                {
                    "path": query,
                    "modality": "sketch",
                    "class": Path(query).parent.name,
                    "name": f"q{qi}_sketch",
                }
            )
            for kind, index in old["common_pairs_gallery_indices"][str(qi)].items():
                photo = old["gallery"][index]
                illustration.append(
                    {
                        "path": photo,
                        "modality": "photo",
                        "class": Path(photo).parent.name,
                        "name": f"q{qi}_{kind}_photo",
                    }
                )
    else:
        # Fixed images across all models; no pretending these are hard negatives.
        for modality in ("sketch", "photo"):
            for cls in selected[:2]:
                row = next(
                    s
                    for s in samples
                    if s["modality"] == modality and s["class"] == cls
                )
                illustration.append(dict(row, name=f"{cls}_{modality}"))
    for row in samples + illustration:
        assert Path(row["path"]).is_file(), row["path"]
    transform = normal_transform(224)
    nq = N_CLASSES * IMAGES_PER_CLASS
    qlabels = [s["class"] for s in samples[:nq]]
    glabels = [s["class"] for s in samples[nq:]]
    rows, checks, summary, pair_rows, provenance = [], [], [], [], {}
    diagram_data = {}
    print("Output:", out)
    print("Seen-class diagnostic sample:", selected, "|", nq, "queries /", nq, "photos")
    print(
        "These images/classes were NOT held out when the existing weights were trained."
    )

    def run_model(name, encoder, info):
        provenance[name] = info
        previous_key = {"Student": "KD student", "Teacher_tuned": "Teacher tuned"}.get(
            name
        )
        if old and previous_key:
            expected = old["models"][previous_key].get("sha256")
            if expected and info.get("sha256") != expected:
                raise ValueError(
                    f"{name} weights differ from the previous audit; check the configured paths"
                )
        depth = len(encoder.visual.transformer.resblocks)
        layers = sorted(
            set([max(1, depth // 4), max(1, depth // 2), max(1, 3 * depth // 4), depth])
        )
        vectors = {l: [] for l in layers}
        embeddings = []
        pair_features = {}
        for i, sample in enumerate(samples + illustration):
            image, tensor = load_image(sample["path"], transform)
            emb, data = capture(encoder, tensor, sample["modality"], layers)
            # Compare with normal, unhooked inference on first sample per model.
            if i == 0:
                with torch.no_grad():
                    normal = encoder.encode(tensor, sample["modality"]).cpu().numpy()[0]
                delta = float(np.max(np.abs(normal - emb)))
                checks.append(
                    {
                        "model": name,
                        "image": sample["path"],
                        "layer": "embedding",
                        "relative_error": delta,
                    }
                )
                if delta > 0.02:
                    raise RuntimeError(f"Hook changed embedding too much: {delta}")
            for layer, d in data.items():
                checks.append(
                    {
                        "model": name,
                        "image": sample["path"],
                        "layer": layer,
                        "relative_error": d["relative_error"],
                    }
                )
            if i < len(samples):
                embeddings.append(emb)
                for layer in layers:
                    vectors[layer].append(data[layer]["group_vectors"])
            else:
                stem = f"{name}_{sample['name']}"
                plot_image(out, stem, encoder, image, data)
                pair_features[sample["name"]] = (
                    emb,
                    {l: d["group_vectors"] for l, d in data.items()},
                )
                diagram_data[(name, sample["name"])] = (
                    image,
                    {
                        l: (
                            d["attention"]
                            .mean(0)[1 : 1 + encoder.grid**2]
                            .reshape(encoder.grid, encoder.grid),
                            d["contribution_norm"]
                            .sum(0)[1 : 1 + encoder.grid**2]
                            .reshape(encoder.grid, encoder.grid),
                        )
                        for l, d in data.items()
                    },
                )
            if i % 8 == 0:
                print(name, i, "/", len(samples) + len(illustration), flush=True)
        embeddings = np.stack(embeddings)
        values = metrics(embeddings[:nq], embeddings[nq:], qlabels, glabels)
        summary.append(
            dict(model=name, layer="final", head="all", group="embedding", **values)
        )
        arrays = {"embedding": embeddings}
        for layer, items in vectors.items():
            v = np.stack(items)
            arrays[f"L{layer}_group_vectors"] = v
            # One vector per image/head/group, in that model's own output space.
            for group_index, group in enumerate(GROUPS + ("all_tokens",)):
                f = v[:, :, group_index] if group_index < 3 else v.sum(2)
                for h in range(f.shape[1]):
                    rows.append(
                        dict(
                            model=name,
                            layer=layer,
                            head=h,
                            group=group,
                            **metrics(f[:nq, h], f[nq:, h], qlabels, glabels),
                        )
                    )
                summed = f.sum(1)
                summary.append(
                    dict(
                        model=name,
                        layer=layer,
                        head="sum",
                        group=group,
                        **metrics(summed[:nq], summed[nq:], qlabels, glabels),
                    )
                )
        np.savez_compressed(out / (name + "_probe_features.npz"), **arrays)
        if old:

            def cosine(a, b):
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                return (
                    float(np.dot(a, b) / denominator) if denominator > 1e-12 else None
                )

            def add_pair(qi, layer, head, group, q, p, n):
                positive, negative = cosine(q, p), cosine(q, n)
                pair_rows.append(
                    {
                        "model": name,
                        "query": old["queries"][qi],
                        "same_class_photo": old["gallery"][
                            old["common_pairs_gallery_indices"][str(qi)]["same_class"]
                        ],
                        "different_class_photo": old["gallery"][
                            old["common_pairs_gallery_indices"][str(qi)][
                                "different_class"
                            ]
                        ],
                        "layer": layer,
                        "head": head,
                        "group": group,
                        "same_cosine": positive,
                        "different_cosine": negative,
                        "margin": (
                            positive - negative
                            if positive is not None and negative is not None
                            else None
                        ),
                    }
                )

            for qi in range(len(old["queries"])):
                q, p, n = [
                    pair_features[f"q{qi}_{suffix}"]
                    for suffix in [
                        "sketch",
                        "same_class_photo",
                        "different_class_photo",
                    ]
                ]
                add_pair(qi, "final", "all", "embedding", q[0], p[0], n[0])
                for layer in layers:
                    for j, group in enumerate(GROUPS + ("all_tokens",)):
                        features = [
                            item[1][layer][:, j] if j < 3 else item[1][layer].sum(1)
                            for item in (q, p, n)
                        ]
                        for h in range(features[0].shape[0]):
                            add_pair(qi, layer, h, group, *[f[h] for f in features])
                        add_pair(qi, layer, "sum", group, *[f.sum(0) for f in features])
        print(name, "final embedding diagnostic:", values, flush=True)

    # Function scope releases weights even in a notebook kernel. Existing user
    # variables are not deleted; their GPU allocations remain the user's own.
    def student_stage():
        backbone = _load_clip_model("ViT-B/32").to(device).eval()
        base, student, info = student_encoders(backbone, str(CKPT), SEED)
        run_model("Student", student, info)
        run_model("Base_CLIP", base, dict(info, mode="raw_no_prompts"))

    def teacher_stage():
        tuned, info = teacher_encoder(str(TEACHER_CACHE), "tuned", DATASET, device)
        # Share frozen weights, but raw runs through native encode_image with NO prompts.
        raw = Encoder(tuned.model, openclip=True)
        run_model(
            "Teacher_raw",
            raw,
            dict(mode="raw_no_prompts", backbone=meta["teacher_model"]),
        )
        run_model("Teacher_tuned", tuned, info)

    student_stage()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    teacher_stage()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Direct raw/tuned teacher comparison, identical grid, layer, and image.
    for sample in illustration:
        key = sample["name"]
        image, raw = diagram_data[("Teacher_raw", key)]
        tuned = diagram_data[("Teacher_tuned", key)][1]
        layers = sorted(raw)
        fig, axes = plt.subplots(4, len(layers), figsize=(15, 11), squeeze=False)
        for col, l in enumerate(layers):
            for row, (label, heat) in enumerate(
                [
                    ("raw A", raw[l][0]),
                    ("tuned A", tuned[l][0]),
                    ("raw ||c||", raw[l][1]),
                    ("tuned ||c||", tuned[l][1]),
                ]
            ):
                draw_map(axes[row, col], image, heat, f"{label} | L{l}")
        fig.suptitle(
            f"Teacher raw / tuned | {key}\nPer-panel scaling; use raw NPZ for quantitative comparisons"
        )
        savefig(fig, out / (key + "_teacher_comparison.png"))
    # All heads shown: a descriptive diagnostic, not a head-selection result.
    for name in MODEL_NAMES:
        subset = [r for r in rows if r["model"] == name and r["group"] == "patch"]
        layers = sorted(set(r["layer"] for r in subset))
        heads = max(r["head"] for r in subset) + 1
        matrix = np.full((len(layers), heads), np.nan)
        for r in subset:
            if r["P5"] is not None:
                matrix[layers.index(r["layer"]), r["head"]] = r["P5"]
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(layers)), [f"L{l}" for l in layers])
        ax.set_xticks(range(heads))
        ax.set_xlabel("Head (zero based)")
        ax.set_title(
            name
            + " | patch-group vector cosine P@5\nSeen diagnostic subset; not held-out validation"
        )
        fig.colorbar(im, ax=ax)
        savefig(fig, out / (name + "_head_probe.png"))

    def write_csv(filename, data):
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write_csv("head_probe.csv", rows)
    write_csv("group_probe.csv", summary)
    write_csv("reconstruction_checks.csv", checks)
    if pair_rows:
        write_csv("fixed_pair_cosines.csv", pair_rows)
    manifest = {
        "dataset": DATASET,
        "seed": SEED,
        "scope": "seen-class sampled diagnostic; NOT held-out validation",
        "probe_samples": samples,
        "illustration_images": illustration,
        "models": provenance,
        "groups": GROUPS,
        "group_vector_axes": "image, head, group, output_width",
        "group_vector_definition": "A_cls,j * V_j @ W_O_head.T, summed within group; output bias separate",
        "limitations": [
            "MHA branch only: excludes residual, LayerScale, MLP and downstream blocks",
            "Norms are not probabilities or causal attribution; cancellation is reported",
            "Cosine probes are untrained and not cross-model feature comparisons",
            "No layer/head selected from unseen examples; no benchmark claims",
        ],
        "torch": str(torch.__version__),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = Path(str(out) + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            z.write(path, arcname=path.name)
    print("DONE:", archive)
    print("Send this ZIP for review. No training or source changes were performed.")
    try:
        from IPython.display import display, Image

        display(
            Image(
                filename=str(
                    out / (illustration[0]["name"] + "_teacher_comparison.png")
                )
            )
        )
        display(Image(filename=str(out / "Teacher_tuned_head_probe.png")))
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    main()
