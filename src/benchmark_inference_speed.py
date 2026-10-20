"""Training-free latency/throughput benchmark; no retrieval accuracy claims."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.data_config import UNSEEN_CLASSES
from src.dataset import TeacherFeatureDataset
from src.tinyclip_inference import MODEL_SPECS, load_frozen_image_encoder
from src.speed_prompts import wrap_student

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MODELS = ("clip_b32", "22m", "40m", "45m", "61m")


def collect(root, dataset):
    result = {}
    for modality in ("sketch", "photo"):
        paths = []
        for name in UNSEEN_CLASSES[dataset]:
            folder = Path(root) / modality / name
            with os.scandir(folder) as entries:
                current = sorted(str(folder / e.name) for e in entries
                                 if not e.name.startswith(".") and Path(e.name).suffix.lower() in EXTENSIONS)
            if not current:
                raise ValueError(f"Empty class: {folder}")
            paths.extend(current)
        result[modality] = paths
    return result


def sample_images(paths, count, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))[:min(count, len(paths))]
    selected = [paths[int(i)] for i in indices]
    dataset = TeacherFeatureDataset(selected, 224)
    return torch.stack([dataset[i] for i in range(len(dataset))]), selected


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(fn, device, warmup, iterations):
    for _ in range(warmup):
        fn()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(iterations):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        values.append((time.perf_counter() - start) * 1000)
    return dict(samples_ms=values, mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), std_ms=float(np.std(values)),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)


@torch.inference_mode()
def throughput(fn, batch_size, device, warmup, iterations):
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync(device)
    seconds = time.perf_counter() - start
    return dict(block_seconds=seconds, images_per_second=batch_size * iterations / seconds,
                amortized_ms_per_image=1000 * seconds / (batch_size * iterations))


def load_encoder(key, args, device):
    if key != "clip_b32":
        encoder = load_frozen_image_encoder(key, args.models_root, device, args.precision)
        info = dict(name=MODEL_SPECS[key].name, implementation=encoder.encoder_kind,
                    vision_parameters=encoder.vision_parameter_count(),
                    weights="official pretrained; no SBIR training")
        student = wrap_student(encoder, encoder.encoder_kind, args, device)
        info.update(prompt_parameters=sum(p.numel() for name,p in student.named_parameters() if "visual_prompt" in name),
                    n_ctx_visual=args.n_ctx_visual, prompt_depth=args.prompt_depth,
                    prompt_weights="seeded main initialization; timing only, not trained quality")
        return student, info
    from clip.model import build_model
    path = Path(args.models_root) / "ViT-B-32.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.jit.load(str(path), map_location="cpu").state_dict()
    model = build_model(dict(state))
    # build_model applies OpenAI's selective fp16 conversion; LayerNorm stays FP32 as in main.
    if args.precision == "fp32":
        model.float()
    model = model.eval().requires_grad_(False).to(device=device)
    # Only the visual tower participates in these timings.
    student = wrap_student(model.visual, "openai", args, device)
    return student, dict(name="CLIP-ViT-B/32 + main visual prompts", implementation="project OpenAI CLIP",
                         vision_parameters=sum(p.numel() for p in model.visual.parameters()),
                         prompt_parameters=sum(p.numel() for name,p in student.named_parameters() if "visual_prompt" in name),
                         n_ctx_visual=args.n_ctx_visual, prompt_depth=args.prompt_depth,
                         prompt_weights="seeded main initialization; timing only, not trained quality",
                         layernorm_precision="fp32 (main/OpenAI mixed precision)",
                         weights="OpenAI pretrained backbone; no SBIR training")


@torch.inference_mode()
def encode_gallery(encoder, paths, device, dtype, batch_size):
    dataset = TeacherFeatureDataset(paths, 224)
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    chunks = []
    sync(device)
    started = time.perf_counter()
    for images in tqdm(loader, desc="Encode actual unseen gallery", leave=False):
        features = encoder(images.to(device=device, dtype=dtype), "photo").float()
        if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
            raise ValueError("Invalid gallery features")
        chunks.append(F.normalize(features, dim=-1).cpu())
    sync(device)
    seconds = time.perf_counter() - started
    return torch.cat(chunks).to(device), seconds


def write_reports(output, report):
    temporary = output / "summary.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "summary.json")
    rows = [{k: v for k, v in row.items() if k != "samples_ms"} for row in report["measurements"]]
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def environment(device):
    packages = {}
    for package in ("torch", "torchvision", "transformers", "timm", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu_driver=command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                source_commit=command(["git", "rev-parse", "HEAD"]),
                tf32=False, cpu_threads=torch.get_num_threads(),
                hf_attention_implementation="eager (explicitly selected by loader)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-json", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sample-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ctx-visual", type=int, default=3)
    parser.add_argument("--prompt-depth", type=int, default=12)
    parser.add_argument("--gallery", choices=["synthetic", "real", "none"], default="synthetic")
    parser.add_argument("--gallery-batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/inference_speed"))
    args = parser.parse_args(argv)
    if min(args.batch_sizes + [args.warmup, args.iterations, args.sample_images,
                               args.gallery_batch_size, args.top_k, args.n_ctx_visual, args.prompt_depth]) < 1:
        parser.error("Batch sizes, sample count, warmup, iterations and K must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no silent CPU fallback")
    if args.device == "cpu" and args.precision == "fp16":
        parser.error("CPU runs must use --precision fp32")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    configs = json.loads(args.datasets_json.read_text(encoding="utf-8"))
    if not configs or any(name not in UNSEEN_CLASSES for name in configs):
        parser.error("Dataset JSON must map known dataset names to their roots")
    output = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  environment=environment(device), datasets={}, models={}, measurements=[], errors=[],
                  scope="Main-style independent deep visual prompts (FP32) + pretrained visual backbones. Seeded untrained prompt values: architecture timing only, no accuracy claims. No teacher/text forward. Synthetic galleries measure search workload, not real retrieval quality.")
    data = {}
    for name, root in configs.items():
        print(f"Listing {name}: {root}", flush=True)
        paths = collect(root, name)
        data[name] = dict(paths=paths, samples={})
        info = dict(root=root, unseen_classes=UNSEEN_CLASSES[name], sketch_count=len(paths["sketch"]),
                    photo_count=len(paths["photo"]), sample_paths={}, path_fingerprints={})
        for modality in ("sketch", "photo"):
            images, selected = sample_images(paths[modality], max(args.sample_images, max(args.batch_sizes)), args.seed)
            data[name]["samples"][modality] = images
            info["sample_paths"][modality] = [str(Path(p).relative_to(root)) for p in selected]
            info["path_fingerprints"][modality] = hashlib.sha256("\n".join(str(Path(p).relative_to(root)) for p in paths[modality]).encode()).hexdigest()
        report["datasets"][name] = info
    write_reports(output, report)
    for key in args.models:
        encoder = None
        try:
            print(f"Loading {key}", flush=True)
            encoder, info = load_encoder(key, args, device)
            report["models"][key] = info
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            with torch.inference_mode():
                for name, datum in data.items():
                    for modality, bank in datum["samples"].items():
                        for batch_size in args.batch_sizes:
                            indices = torch.arange(batch_size) % len(bank)
                            inputs = bank[indices].to(device=device, dtype=dtype)
                            output_features = encoder(inputs, modality)
                            if not torch.isfinite(output_features).all() or (output_features.float().norm(dim=-1) == 0).any():
                                raise ValueError("Encoder produced invalid features")
                            fn = lambda: encoder(inputs, modality)
                            measured = measure(fn, device, args.warmup, args.iterations)
                            row = dict(model=key, dataset=name, modality=modality, stage="encoder_only",
                                       batch_size=batch_size, precision=args.precision, **measured,
                                       **throughput(fn, batch_size, device, args.warmup, args.iterations))
                            report["measurements"].append(row)
                            print(f"{key} {name} {modality} B={batch_size}: median {row['median_ms']:.3f} ms/batch, {row['images_per_second']:.1f} images/s", flush=True)
                            del inputs, output_features, fn
                    if args.gallery != "none":
                        # Search timing always uses a single preprocessed query, with gallery resident on device.
                        query_image = datum["samples"]["sketch"][:1].to(device=device, dtype=dtype)
                        q = F.normalize(encoder(query_image, "sketch").float(), dim=-1)
                        n = len(datum["paths"]["photo"])
                        if args.gallery == "real":
                            gallery, build_seconds = encode_gallery(encoder, datum["paths"]["photo"], device, dtype, args.gallery_batch_size)
                        else:
                            generator = torch.Generator(device=device).manual_seed(args.seed)
                            gallery = F.normalize(torch.randn(n, q.shape[1], generator=generator, device=device), dim=-1)
                            build_seconds = None
                        search = lambda query: (query @ gallery.T).topk(min(args.top_k, n), dim=-1)
                        for stage, fn in (("search_only", lambda: search(q)),
                                          ("encoder_plus_search", lambda: search(F.normalize(encoder(query_image, "sketch").float(), dim=-1)))):
                            row = dict(model=key, dataset=name, modality="sketch", stage=stage, batch_size=1,
                                       precision=args.precision, gallery_kind=args.gallery, gallery_photos=n,
                                       top_k=min(args.top_k, n), gallery_build_seconds=build_seconds,
                                       **measure(fn, device, args.warmup, args.iterations))
                            report["measurements"].append(row)
                        del gallery, q, query_image, search, fn
                    write_reports(output, report)
        except Exception as error:
            report["errors"].append(dict(model=key, error=f"{type(error).__name__}: {error}"))
            print(f"FAILED {key}: {error}", flush=True)
        finally:
            del encoder
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_reports(output, report)
    report["complete"] = not report["errors"]
    write_reports(output, report)
    print(f"Results: {output}", flush=True)
    if report["errors"]:
        raise RuntimeError("Some models failed; see errors in summary.json. Partial results retained.")
    return report


if __name__ == "__main__":
    main()
