"""Measure main's teacher -> cache -> student pipeline without changing its losses."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch

from src.training_cost import TrainingCostRecorder, instrument_training, local_training_weights


def build_training_argv(args, remaining, run_id):
    cache_dir = args.teacher_cache_dir
    cache = args.teacher_cache_path or cache_dir / ("teacher_" + run_id + ".pt")
    cache = cache.resolve()
    if cache.is_file() and not (args.allow_cached_teacher or args.rebuild_teacher_cache):
        raise ValueError("Teacher cache already exists. Use a fresh path, --rebuild_teacher_cache, "
                         "or explicitly --allow-cached-teacher for a student-only timing run.")
    remaining = remaining[1:] if remaining[:1] == ["--"] else list(remaining)
    result = remaining + ["--teacher_pretrain_epochs", str(args.teacher_pretrain_epochs),
                          "--teacher_cache_path", str(cache),
                          "--exp_name", args.exp_name + "_" + run_id]
    if args.rebuild_teacher_cache:
        result.append("--rebuild_teacher_cache")
    return result, cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                    epilog="Other arguments are forwarded unchanged to src.train (e.g. --root, --dataset, --epochs).")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/training_cost"))
    parser.add_argument("--teacher_cache_dir", type=Path, default=Path("/kaggle/working/teacher_cache_training_cost"))
    parser.add_argument("--teacher_cache_path", type=Path)
    parser.add_argument("--teacher_pretrain_epochs", type=int, default=1)
    parser.add_argument("--exp_name", default="training_cost")
    parser.add_argument("--allow-cached-teacher", action="store_true")
    parser.add_argument("--rebuild_teacher_cache", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.teacher_pretrain_epochs < 1:
        parser.error("Use --teacher_pretrain_epochs >= 1 for a training-cost configuration")
    if not torch.cuda.is_available():
        parser.error("Training cost runs main's single-GPU pipeline; enable Kaggle GPU")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    try:
        train_argv, cache = build_training_argv(args, remaining, run_id)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir / run_id
    # Record the unchanged training source hashes, as well as full effective args.
    source_root = Path(__file__).resolve().parents[1]
    sources = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
               for name in ("src/train.py", "src/model.py", "src/losses.py", "src/teacher_prompts.py",
                            "src/dataset.py", "src/data_config.py")}
    recorder = TrainingCostRecorder(output, metadata=dict(
        training_base="741ea272b258c9e7b92a46173618e2fe1d401eaa", training_source_sha256=sources,
        requested_argv=list(argv if argv is not None else sys.argv[1:]), effective_train_argv=train_argv,
        models_root=str(args.models_root.resolve()), teacher_cache_path=str(cache)))
    print("Training reports:", output, flush=True)
    print("Teacher cache:", cache, flush=True)
    try:
        with recorder.stage("pipeline"):
            from src.benchmark_encoder_cost import environment
            from src import model as model_module
            from pytorch_lightning import Callback, Trainer
            recorder.report["environment"] = environment()
            with local_training_weights(model_module, args.models_root), \
                    instrument_training(recorder, model_module, Trainer, Callback), \
                    patch.object(sys, "argv", ["src.train", *train_argv]):
                runpy.run_module("src.train", run_name="__main__")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        recorder.finish(status, f"{type(error).__name__}: {error}")
        raise
    else:
        recorder.finish("interrupted" if recorder.report.get("trainer_interrupted") else "completed")
    print("Training cost:", output / "training_cost.csv", flush=True)
    print("Full report:", output / "training_cost.json", flush=True)


if __name__ == "__main__":
    main()
