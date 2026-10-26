"""Batch-1 prompted encoder params, FLOPs, and synchronized GPU/CPU latency."""
import argparse
import csv
from datetime import datetime, timezone
import gc
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from src.encoder_cost_models import SPECS, load_model, parameter_counts, set_precision, vit_flops
from src.cost_runtime import download_mode, ensure_models, models_directory, working_directory


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(fn, device, warmup, iterations):
    output = fn()
    if not torch.isfinite(output).all() or (output.float().norm(dim=-1) == 0).any():
        raise ValueError("Invalid encoder output")
    for _ in range(warmup):
        fn()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iterations):
        sync(device)
        started = time.perf_counter()
        fn()
        sync(device)
        samples.append(1000 * (time.perf_counter() - started))
    ordered = sorted(samples)
    position = .95 * (len(ordered) - 1)
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    p95 = ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])
    return dict(median_ms=statistics.median(samples), mean_ms=statistics.mean(samples),
                p95_ms=p95, std_ms=statistics.pstdev(samples), samples_ms=samples,
                output_dim=int(output.shape[-1]),
                peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                    if device.type == "cuda" else None))


@torch.inference_mode()
def profile_flops(model, images, modality):
    """Untimed math-attention pass; check the analytical count against real ops.

    Disable fused/native attention ONLY here so all supported matmuls appear.
    Latency uses the normal implementation and is measured before this pass.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.utils.flop_counter import FlopCounterMode
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
            model(images, modality)
        operations = {str(op): int(count) for op, count in counter.get_flop_counts()["Global"].items()}
        return dict(total=int(counter.get_total_flops()), operations=operations)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    packages = {}
    for name in ("torch", "torchvision", "open-clip-torch", "transformers", "timm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cpu = (command(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).Name"]) if os.name == "nt"
           else command(["lscpu"]))
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                cpu=cpu or platform.processor(), cpu_threads=torch.get_num_threads(),
                cpu_interop_threads=torch.get_num_interop_threads(), logical_cpus=os.cpu_count(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu_driver=command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                    "--format=csv,noheader"]),
                source_commit=command(["git", "rev-parse", "HEAD"]),
                tf32=False, cudnn_benchmark=False, compile=False,
                mha_fastpath=torch.backends.mha.get_fastpath_enabled())


def save_reports(output, report):
    temporary = output / "summary.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    rows = []
    for row in report["measurements"]:
        info = report["models"][row["model"]]
        counts = info["parameters"]
        rows.append({**{k: v for k, v in row.items() if k != "samples_ms"},
                     **counts, "query_params_m": counts["query_parameters"] / 1e6,
                     "deployed_params_m": counts["deployed_parameters"] / 1e6,
                     "gflops": info["flops"]["total"] / 1e9,
                     "n_ctx": info["n_ctx"], "prompt_depth": info["depth"]})
    if rows:
        with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # One paper-ready row per model, with device/modalities in separate columns.
    table = []
    for key, info in report["models"].items():
        counts = info["parameters"]
        row = dict(model=key, query_params_m=counts["query_parameters"] / 1e6,
                   deployed_params_m=counts["deployed_parameters"] / 1e6,
                   gflops=info["flops"]["total"] / 1e9, dim=SPECS[key]["dim"],
                   n_ctx=info["n_ctx"], prompt_depth=info["depth"])
        for device in report["config"]["devices"]:
            for modality in report["config"]["modalities"]:
                matching = [r for r in report["measurements"] if r["model"] == key
                            and r["device"] == device and r["modality"] == modality]
                row[f"{device}_{modality}_median_ms"] = matching[0]["median_ms"] if matching else None
        teacher = report["models"].get("dfn5b_h14")
        row["flops_speedup_vs_teacher"] = teacher["flops"]["total"] / info["flops"]["total"] if teacher else None
        for device in report["config"]["devices"]:
            for modality in report["config"]["modalities"]:
                baseline = next((r for r in report["measurements"] if r["model"] == "dfn5b_h14"
                                 and r["device"] == device and r["modality"] == modality), None)
                value = row[f"{device}_{modality}_median_ms"]
                row[f"{device}_{modality}_speedup_vs_teacher"] = baseline["median_ms"] / value if baseline and value else None
        table.append(row)
    if table:
        with (output / "table4.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--download-mode", choices=["auto", "online", "offline"], default="auto")
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--devices", nargs="+", choices=["cuda", "cpu"], default=["cuda", "cpu"])
    parser.add_argument("--modalities", nargs="+", choices=["sketch", "photo"], default=["sketch"])
    parser.add_argument("--gpu-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--cpu-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--teacher-n-ctx", type=int, default=9)
    parser.add_argument("--student-n-ctx", type=int, default=3)
    parser.add_argument("--teacher-depth", type=int, default=12)
    parser.add_argument("--student-depth", type=int, default=12)
    parser.add_argument("--teacher-prompts", type=Path)
    parser.add_argument("--clip-prompts", type=Path)
    parser.add_argument("--tiny-prompts", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.iterations, args.cpu_threads, args.teacher_n_ctx, args.student_n_ctx) < 1 or args.warmup < 0:
        parser.error("Iterations, threads, prompt counts must be positive; warmup must be nonnegative")
    if not 1 <= args.teacher_depth <= 32 or not 1 <= args.student_depth <= 12:
        parser.error("Teacher depth must be 1..32; student depth must be 1..12")
    for name in ("models", "devices", "modalities"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA unavailable; enable the notebook GPU or pass --devices cpu")
    args.models_root = models_directory(args.models_root)
    args.output_dir = args.output_dir or working_directory() / "encoder_cost"
    args.download_mode = download_mode(args.download_mode)
    ensure_models(args.models, args.models_root, args.download_mode)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(config=config, environment=environment(), models={}, measurements=[],
                  protocol=dict(batch_size=1, image_size=224, input="seeded synthetic tensor resident on device",
                                scope="encoder + visual prompts + feature normalization; no I/O, H2D, gallery or search",
                                params="query: vision + sketch bank; deployed: vision + both banks; no text tower",
                                flops="conv/linear/QK^T/AV; 2 FLOPs per MAC; excludes bias, normalization, activation and softmax",
                                accuracy="not measured; prompt initialization/checkpoint source recorded per model"))
    save_reports(output, report)
    print("Results:", output, flush=True)
    for key in args.models:
        print(f"Loading {key} on CPU ...", flush=True)
        teacher = key == "dfn5b_h14"
        n_ctx = args.teacher_n_ctx if teacher else args.student_n_ctx
        depth = args.teacher_depth if teacher else args.student_depth
        prompt_path = {"dfn5b_h14": args.teacher_prompts, "clip_b32": args.clip_prompts,
                       "tinyclip_61m": args.tiny_prompts}[key]
        model, prompt_source = load_model(key, args.models_root, n_ctx, depth, args.seed, prompt_path)
        report["models"][key] = dict(name=SPECS[key]["name"], n_ctx=n_ctx, depth=depth,
                                    prompt_source=prompt_source, parameters=parameter_counts(model, teacher),
                                    attention=("transformers SDPA" if key == "tinyclip_61m"
                                               else "PyTorch MultiheadAttention (default dispatch)"),
                                    flops=vit_flops(SPECS[key], n_ctx))
        for name in args.devices:
            device = torch.device(name)
            precision = args.gpu_precision if name == "cuda" else args.cpu_precision
            set_precision(model, key, device, precision)
            # A single resident image. No dataset workers or gallery scans.
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            images = torch.randn(1, 3, 224, 224, generator=generator).to(
                device=device, dtype=torch.float16 if precision == "fp16" else torch.float32)
            print(f"  {key}: {name}, {precision}, B=1, ctx={n_ctx}, depth={depth}", flush=True)
            for modality in args.modalities:
                values = measure(lambda: model(images, modality), device, args.warmup, args.iterations)
                if values["output_dim"] != SPECS[key]["dim"]:
                    raise ValueError(f"Unexpected output dimension for {key}")
                row = dict(model=key, device=name, precision=precision, modality=modality,
                           batch_size=1, warmup=args.warmup, iterations=args.iterations, **values)
                report["measurements"].append(row)
                save_reports(output, report)
                print(f"    {modality}: median {values['median_ms']:.3f} ms; p95 {values['p95_ms']:.3f} ms", flush=True)
            # Validate once/model after timing; restored math attention does not affect latency.
            if "profile" not in report["models"][key]:
                measured = profile_flops(model, images, args.modalities[0])
                expected = report["models"][key]["flops"]["total"]
                measured["relative_difference"] = abs(measured["total"] - expected) / expected
                report["models"][key]["profile"] = measured
                if measured["relative_difference"] > .001:
                    save_reports(output, report)
                    raise RuntimeError(f"FLOP coverage mismatch for {key}; inspect summary.json before using results")
            del images
            model.cpu().float()
            gc.collect()
            if name == "cuda":
                torch.cuda.empty_cache()
            save_reports(output, report)
        del model
        gc.collect()
    print("COMPLETE:", output, flush=True)
    print("Return table4.csv, measurements.csv and summary.json for analysis.", flush=True)


if __name__ == "__main__":
    main()
