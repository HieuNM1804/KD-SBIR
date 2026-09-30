"""Run five text-template pairs with the user's Sketchy-2 training settings."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.experiment_results import best_validation_epoch, write_json
from src.text_prompts import TEXT_PROMPT_PAIRS, prompt_pair_config

DEFAULT_ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"


def common_arguments(args):
    return [
        "--root", args.root,
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--batch_size", "64",
        "--test_batch_size", "1024",
        "--n_ctx_visual", "3",
        "--prompt_depth", "12",
        "--teacher_pretrain_epochs", "1",
        "--teacher_pretrain_batch_size", "64",
        "--teacher_n_ctx_visual", "10",
        "--teacher_prompt_depth", "12",
        "--teacher_prompt_std", "0.02",
        "--teacher_prompt_lr", "3e-2",
        "--teacher_prompt_seed", "42",
        "--teacher_prompt_gradient_checkpointing",
        "--teacher_momentum", "0.9",
        "--teacher_weight_decay", "1e-3",
        "--lambda_teacher_retrieval", "1.5",
        "--teacher_triplet_margin", "0.2",
        "--lambda_domain", "3.0",
        "--lambda_modality", "1.0",
        "--photo_text_kd_temperature", "0.15",
        "--sketch_text_kd_temperature", "0.02",
        "--lr", "1e-2",
        "--momentum", "0.9",
        "--weight_decay", "1e-5",
        "--seed", "42",
        "--progress",
    ]


def source_fingerprint(project=PROJECT):
    digest = hashlib.sha256()
    paths = sorted((project / "src").glob("*.py"))
    paths += sorted((project / "clip").glob("*.py"))
    paths.append(project / "test" / "kaggle_text_prompt_sweep.py")
    for path in paths:
        digest.update(path.relative_to(project).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command_for_pair(args, pair, output):
    return [
        sys.executable, "-u", "-m", "src.train",
        *common_arguments(args),
        "--text_prompt_pair", pair,
        "--results_path", str(output / "runs" / f"{pair}.json"),
        "--teacher_cache_path", str(output / "teacher_cache.pt"),
        # Retrain with identical seeds each time and overwrite one cache to
        # avoid keeping five copies of all teacher image features on disk.
        "--rebuild_teacher_cache",
        "--exp_name", f"text_prompt_{args.dataset}_{output.name}_{pair}",
    ]


def run_training(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code != 0:
        raise RuntimeError(f"Training failed with exit code {code}. See {log_path}")


def load_result(path, pair, args):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"Incomplete training result: {path}")
    if result.get("text_prompt_pair") != prompt_pair_config(pair):
        raise ValueError(f"Template mismatch: {path}")
    if result.get("dataset") != args.dataset:
        raise ValueError(f"Dataset mismatch: {path}")
    history = result.get("history", [])
    if len(history) != args.epochs:
        raise ValueError(f"Expected {args.epochs} trained epochs in {path}")
    selected = best_validation_epoch(history)
    if result.get("best_epoch") != selected:
        raise ValueError(f"Best epoch mismatch: {path}")
    if not Path(result["best_checkpoint"]).is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {path}")
    return result


def export_summary(output, results):
    rows = []
    for result in results:
        pair = result["text_prompt_pair"]
        selected = result["best_epoch"]
        rows.append({
            "pair": pair["name"],
            "photo_template": pair["photo"],
            "sketch_template": pair["sketch"],
            "best_epoch": selected["epoch"],
            "best_precision": selected["precision"],
            "mAP_at_best_precision": selected["mAP"],
            "p_k": selected["p_k"],
            "map_k": selected["map_k"],
            "best_checkpoint": result["best_checkpoint"],
        })
    temporary = output / "summary.csv.tmp"
    fields = [
        "pair", "photo_template", "sketch_template", "best_epoch",
        "best_precision", "mAP_at_best_precision", "p_k", "map_k",
        "best_checkpoint",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    write_json(output / "all_results.json", results)
    if rows:
        # Python max keeps the first pair on exact precision ties.
        winner_index = max(range(len(rows)), key=lambda i: rows[i]["best_precision"])
        write_json(output / "best_run.json", {
            "status": "completed" if len(rows) == len(TEXT_PROMPT_PAIRS) else "partial",
            "completed_pairs": len(rows),
            "total_pairs": len(TEXT_PROMPT_PAIRS),
            "selection_metric": results[winner_index]["selection_metric"],
            "winner": results[winner_index],
        })
        return rows[winner_index]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=("sketchy_1", "sketchy_2"), default="sketchy_2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume", action="store_true", help="Skip completed pairs in an existing output_dir.")
    parser.add_argument("--dry_run", action="store_true", help="Print five commands without loading models or data.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.workers < 0:
        parser.error("epochs must be positive and workers must be non-negative.")
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output_dir.")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir or f"/kaggle/working/text_prompt_sweep_{args.dataset}_{timestamp}").resolve()
    if args.dry_run:
        for pair in TEXT_PROMPT_PAIRS:
            print(prompt_pair_config(pair))
            print(subprocess.list2cmdline(command_for_pair(args, pair, output)))
        return
    for modality in ("photo", "sketch"):
        if not (Path(args.root) / modality).is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {Path(args.root) / modality}")
    protocol = {
        "source_sha256": source_fingerprint(),
        "arguments": common_arguments(args),
        "pairs": TEXT_PROMPT_PAIRS,
        "selection_metric": "P@200" if args.dataset == "sketchy_2" else "P@100",
    }
    manifest_path = output / "sweep_manifest.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Cannot resume: source/config/templates differ or the manifest is missing.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, protocol)
    (output / "runs").mkdir(exist_ok=True)
    results = []
    export_summary(output, results)
    print(f"Output: {output}", flush=True)
    for index, pair in enumerate(TEXT_PROMPT_PAIRS, start=1):
        print(f"[{index}/5] {prompt_pair_config(pair)}", flush=True)
        result_path = output / "runs" / f"{pair}.json"
        previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        if args.resume and previous.get("status") == "completed":
            result = load_result(result_path, pair, args)
            print(f"Reusing completed pair: {pair}", flush=True)
        else:
            command = command_for_pair(args, pair, output)
            write_json(output / "runs" / f"{pair}_command.json", command)
            run_training(command, output / "logs" / f"{pair}.log")
            result = load_result(result_path, pair, args)
        results.append(result)
        winner = export_summary(output, results)
        print(f"Best so far: {winner['pair']} P@{winner['p_k']}={winner['best_precision']:.6f}", flush=True)
    shutil.copy2(winner["best_checkpoint"], output / "best.ckpt")
    best = json.loads((output / "best_run.json").read_text(encoding="utf-8"))
    best["exported_checkpoint"] = str(output / "best.ckpt")
    write_json(output / "best_run.json", best)
    print(f"BEST PAIR: {winner['pair']}")
    print(f"P@{winner['p_k']}: {winner['best_precision']:.6f}")
    print(f"mAP at the same epoch: {winner['mAP_at_best_precision']:.6f}")
    print(f"Summary: {output / 'summary.csv'}")
    print(f"Best result: {output / 'best_run.json'}")
    print(f"Best checkpoint: {output / 'best.ckpt'}")


if __name__ == "__main__":
    main()
