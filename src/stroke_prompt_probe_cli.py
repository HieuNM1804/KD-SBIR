"""Run a small GPU localization probe before scheduling full SGCD training.

python -m src.stroke_prompt_probe_cli --synthetic
python -m src.stroke_prompt_probe_cli --root ... --teacher-cache ... --target-cache ...
"""

import argparse
import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from clip.model import build_model
from src.dataset import TrainDataset, normal_transform
from src.model import CustomCLIP
from src.stroke_graph import StrokeGraphEvidenceHead, patch_ink_mass
from src.stroke_graph_cache import file_sha256, validate_payload
from src.stroke_prompt_probe import localization_probe, save_probe
from src.train import seed_everything


def synthetic_batch():
    """Two disjoint shapes per image with alternating left/right targets."""
    images, targets = [], []
    for index in range(6):
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        shift = index * 4
        draw.ellipse((20, 30 + shift, 90, 170 + shift), outline="black", width=5)
        draw.line(
            [(140, 40 + shift), (200, 100), (140, 180 - shift)], fill="black", width=5
        )
        tensor = normal_transform(224)(image)
        ink = patch_ink_mass(tensor[None], 7)[0]
        columns = torch.arange(49) % 7
        support = columns < 3 if index % 2 == 0 else columns >= 4
        target = ink * support
        images.append(tensor)
        targets.append(target / target.sum())
    return torch.stack(images), torch.stack(targets), torch.ones(6)


def cached_batch(options):
    """Read historical targets with data/teacher checks; never rewrite the cache.

    Source hashes are retained as provenance, since this is deliberately a test
    of a different student against fixed historical teacher targets. Full cache
    construction continues to use prepare_cache's strict source validation.
    """
    cache_path = Path(options.target_cache)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    validate_payload(payload, metadata)
    if file_sha256(options.teacher_cache) != metadata["teacher_cache_sha256"]:
        raise ValueError("Target cache was built from a different main teacher cache")
    cfg = Namespace(
        root=options.root,
        dataset=metadata["dataset"],
        max_size=metadata["max_size"],
        seed=metadata["seed"],
    )
    dataset = TrainDataset(cfg)
    if dataset.all_categories != metadata["classnames"]:
        raise ValueError("Seen class lists differ")
    for key, paths in (
        ("sketch", dataset.all_sketches_path),
        ("photo", dataset.all_photo_paths),
    ):
        digest = CustomCLIP._path_fingerprint(paths, cfg.root)
        if digest != metadata[key + "_fingerprint"]:
            raise ValueError(f"{key} path order differs from the target cache")
    if metadata["student_grid"] != 7:
        raise ValueError("This probe expects the original ViT-B/32 patch grid")
    generator = torch.Generator().manual_seed(cfg.seed + 9400)
    indices = torch.randperm(len(dataset), generator=generator)[
        : options.samples
    ].tolist()
    images = torch.stack([dataset[(0, index)][1] for index in indices])
    return (
        images,
        payload["maps"][indices, 0].float(),
        payload["confidence"][indices, 0],
        metadata,
        indices,
        file_sha256(cache_path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--root", default="")
    parser.add_argument("--teacher-cache", default="")
    parser.add_argument("--target-cache", default="")
    parser.add_argument(
        "--clip-weights", default=str(Path.home() / ".cache/clip/ViT-B-32.pt")
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    options = parser.parse_args()
    if options.samples < 2:
        parser.error("At least two samples are required")
    if not options.synthetic and not all(
        (options.root, options.teacher_cache, options.target_cache)
    ):
        parser.error(
            "Provide --root, --teacher-cache and --target-cache, or --synthetic"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU for the pretrained CLIP probe")
    seed_everything(options.seed)
    torch.set_num_threads(4)
    if options.synthetic:
        images, targets, confidence = synthetic_batch()
        provenance = {
            "target_kind": "synthetic left/right stroke targets; no teacher or dataset"
        }
    else:
        images, targets, confidence, metadata, indices, digest = cached_batch(options)
        options.seed = metadata["seed"]
        seed_everything(options.seed)
        provenance = {
            "target_kind": "historical seen teacher targets",
            "target_cache_sha256": digest,
            "target_metadata": metadata,
            "indices": indices,
        }
    weights_path = Path(options.clip_weights)
    loaded = torch.jit.load(str(weights_path), map_location="cpu").eval()
    backbone = build_model(loaded.state_dict())
    del loaded
    cfg = Namespace(
        n_ctx_visual=3,
        prompt_depth=12,
        seed=options.seed,
        retrieval_head="sgcd",
        sgcd_student_mode="native_prompt",
        lambda_domain=0.0,
        lambda_modality=0.0,
        kd_temperature=0.07,
        photo_text_kd_temperature=0.15,
        sketch_text_kd_temperature=0.02,
        teacher_cache_path="",
        rebuild_teacher_cache=False,
        teacher_pretrain_epochs=0,
        sgcd_beta=0.0,
        sgcd_target="verified",
        lambda_sgcd=1.0,
        sgcd_ink_threshold=0.08,
        sgcd_ink_softness=0.12,
    )
    model = CustomCLIP(cfg, backbone, ["object"]).cuda().eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        output = model.encode_student_image_details(images.cuda(), "sketch")
        legacy = StrokeGraphEvidenceHead(512, 7, graph_steps=0, graph_mix=0).cuda()
        native, dense, ink = output["native"], output["dense"], output["ink_mass"]
        old = legacy(native, dense, ink)
        logits = torch.einsum(
            "bnd,bd->bn",
            legacy.key(dense),
            legacy.query(F.normalize(native.float(), dim=-1)),
        ) / (legacy.key.out_features**0.5 * legacy.temperature)
        prior = ink / ink.sum(-1, keepdim=True).clamp_min(1e-8)
        provenance["legacy_initial_map_ink_cosine"] = (
            F.cosine_similarity(old["weights"], prior, dim=-1).mean().item()
        )
        provenance["legacy_initial_semantic_logit_std"] = logits.std(-1).mean().item()
        del legacy, output, old
    provenance["clip_weights_sha256"] = file_sha256(weights_path)
    provenance["seed"] = options.seed
    provenance["device"] = torch.cuda.get_device_name()
    provenance["torch"] = torch.__version__
    result = localization_probe(
        model, images, targets, confidence, options.steps, options.lr, options.optimizer
    )
    out = (
        Path(options.out)
        if options.out
        else Path("artifacts")
        / ("native_prompt_probe_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"))
    )
    summary = save_probe(out, result, images, targets, provenance)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("Probe files:", out.resolve())


if __name__ == "__main__":
    main()
