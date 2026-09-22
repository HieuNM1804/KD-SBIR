"""Run the isolated dual-axis AFD study on Kaggle and export compact evidence.

The student objective never invokes the domain/modality losses from main.  The
student-only and shuffled-teacher runs are controls for causal teacher use.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# TensorBoard versions bundled by some Kaggle images still use NumPy 1 aliases.
if "string_" not in np.__dict__:
    np.string_ = np.bytes_
if "unicode_" not in np.__dict__:
    np.unicode_ = np.str_


PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
DEFAULT_ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"


def run_stage(label, command, log_dir):
    log_path = log_dir / f"{label}.log"
    print("=" * 80, flush=True)
    print(label.upper(), flush=True)
    print(" ".join(command), flush=True)
    print("=" * 80, flush=True)
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
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}: {log_path}")
    return log_path


def read_scalars(run_name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    versions = sorted((PROJECT / "tb_logs" / run_name).glob("version_*"))
    if not versions:
        raise RuntimeError(f"No TensorBoard output found for {run_name}")
    accumulator = EventAccumulator(str(versions[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves = {
        tag: [
            {"step": item.step, "value": item.value}
            for item in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags().get("scalars", [])
    }
    if not curves.get("precision") or not curves.get("mAP"):
        raise RuntimeError(f"Retrieval metrics are missing for {run_name}")
    maps = {item["step"]: item["value"] for item in curves["mAP"]}
    trained_precisions = [item for item in curves["precision"] if item["step"] > 0]
    selected = max(
        trained_precisions or curves["precision"],
        key=lambda item: item["value"],
    )
    row = {
        "run": run_name,
        "selected_step": selected["step"],
        "selected_precision": selected["value"],
        "selected_mAP": maps[selected["step"]],
        "last_precision": curves["precision"][-1]["value"],
        "last_mAP": curves["mAP"][-1]["value"],
    }
    diagnostic_tags = (
        "train_loss",
        "AFD_SP",
        "AFD_IT",
        "afd_grad_prompt",
        "afd_grad_fusion",
        "afd_grad_prompt_sp",
        "afd_grad_prompt_it",
        "afd_grad_prompt_sp_it_cosine",
        "afd_image_student_weight_norm",
        "afd_image_teacher_weight_norm",
        "afd_text_student_weight_norm",
        "afd_text_teacher_weight_norm",
    )
    for tag in diagnostic_tags:
        if curves.get(tag):
            finite = [item["value"] for item in curves[tag] if item["value"] == item["value"]]
            if finite:
                row[tag + "_last"] = finite[-1]
                row[tag + "_mean"] = sum(finite) / len(finite)
    return row, curves


def condition_grid():
    conditions = [
        {"name": "student_only_sp", "sp": 1.0, "it": 0.0, "control": "student_only"},
        {"name": "student_only_it", "sp": 0.0, "it": 1.0, "control": "student_only"},
        {"name": "student_only_full", "sp": 0.3, "it": 0.3, "control": "student_only"},
    ]
    for weight in (0.1, 0.3, 1.0):
        suffix = str(weight).replace(".", "p")
        conditions.append(
            {"name": f"verified_sp_w{suffix}", "sp": weight, "it": 0.0, "control": "verified"}
        )
        conditions.append(
            {"name": f"verified_it_w{suffix}", "sp": 0.0, "it": weight, "control": "verified"}
        )
    for sp_weight in (0.1, 0.3, 1.0):
        for it_weight in (0.1, 0.3, 1.0):
            sp = str(sp_weight).replace(".", "p")
            it = str(it_weight).replace(".", "p")
            conditions.append(
                {"name": f"verified_full_sp{sp}_it{it}", "sp": sp_weight, "it": it_weight, "control": "verified"}
            )
    for control in ("shuffled_image", "shuffled_text", "shuffled_both", "teacher_only"):
        conditions.append(
            {"name": f"control_{control}", "sp": 0.3, "it": 0.3, "control": control}
        )
    conditions.extend(
        (
            {"name": "full_temp_0p05", "sp": 0.3, "it": 0.3, "control": "verified", "temp": 0.05},
            {"name": "full_temp_0p10", "sp": 0.3, "it": 0.3, "control": "verified", "temp": 0.10},
            {"name": "full_xavier", "sp": 0.3, "it": 0.3, "control": "verified", "init": "xavier"},
        )
    )
    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sketchy_1", "sketchy_2"), required=True)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output = Path("/kaggle/working") / f"afd_sweep_{args.dataset}_{timestamp}"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    cache = Path("/kaggle/working/teacher_cache") / (
        f"{args.dataset}_main_teacher2_afd_v6.pt"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"afd_{args.dataset}_s{args.seed}_{timestamp}"
    common = [
        "--root", args.root,
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--batch_size", "64",
        "--test_batch_size", "1024",
        "--n_ctx_visual", "3",
        "--prompt_depth", "12",
        "--teacher_pretrain_epochs", "2",
        "--teacher_pretrain_batch_size", "64",
        "--teacher_n_ctx_visual", "10",
        "--teacher_prompt_depth", "12",
        "--teacher_prompt_std", "0.02",
        "--teacher_prompt_lr", "3e-2",
        "--teacher_prompt_seed", str(args.seed),
        "--teacher_prompt_gradient_checkpointing",
        "--teacher_momentum", "0.9",
        "--teacher_weight_decay", "1e-3",
        "--lambda_teacher_retrieval", "1.5",
        "--teacher_triplet_margin", "0.2",
        "--teacher_cache_path", str(cache),
        "--lambda_domain", "0",
        "--lambda_modality", "0",
        "--lr", "1e-2",
        "--afd_fusion_lr", "1e-3",
        "--momentum", "0.9",
        "--weight_decay", "5e-4",
        "--seed", str(args.seed),
        "--afd_grad_every", "50",
        "--no_progress",
        "--no_checkpoints",
    ]
    base_command = [sys.executable, "-u", "-m", "src.train", *common]
    run_stage(
        "prepare_teacher_cache",
        [
            *base_command,
            "--lambda_afd_sp", "1",
            "--lambda_afd_it", "1",
            "--teacher_cache_only",
            "--exp_name", prefix + "_cache",
        ],
        logs,
    )

    rows = []
    curves_by_condition = {}
    conditions = condition_grid()
    for index, condition in enumerate(conditions, start=1):
        name = condition["name"]
        run_name = f"{prefix}_{index:02d}_{name}"
        temperature = condition.get("temp", 0.07)
        command = [
            *base_command,
            "--lambda_afd_sp", str(condition["sp"]),
            "--lambda_afd_it", str(condition["it"]),
            "--afd_temperature_sp", str(temperature),
            "--afd_temperature_it", str(temperature),
            "--afd_control", condition["control"],
            "--afd_init", condition.get("init", "student_identity"),
            "--exp_name", run_name,
        ]
        run_stage(f"{index:02d}_{name}", command, logs)
        row, curves = read_scalars(run_name)
        row.update(condition)
        row["temperature"] = temperature
        rows.append(row)
        curves_by_condition[name] = curves

    by_name = {row["name"]: row for row in rows}
    comparisons = {
        "verified_sp_w1_minus_student_only_sp_mAP_pp": 100 * (
            by_name["verified_sp_w1p0"]["selected_mAP"]
            - by_name["student_only_sp"]["selected_mAP"]
        ),
        "verified_it_w1_minus_student_only_it_mAP_pp": 100 * (
            by_name["verified_it_w1p0"]["selected_mAP"]
            - by_name["student_only_it"]["selected_mAP"]
        ),
    }
    verified_full = [row for row in rows if row["name"].startswith("verified_full")]
    best = max(verified_full, key=lambda row: (row["selected_precision"], row["selected_mAP"]))
    comparisons["best_verified_full_minus_student_only_full_mAP_pp"] = 100 * (
        best["selected_mAP"] - by_name["student_only_full"]["selected_mAP"]
    )
    matched_full = by_name["verified_full_sp0p3_it0p3"]
    for control in (
        "student_only_full",
        "control_shuffled_image",
        "control_shuffled_text",
        "control_shuffled_both",
        "control_teacher_only",
    ):
        comparisons[f"verified_full_0p3_minus_{control}_mAP_pp"] = 100 * (
            matched_full["selected_mAP"] - by_name[control]["selected_mAP"]
        )
    mechanism_checks = {
        "teacher_improves_sp_axis": (
            by_name["verified_sp_w1p0"]["selected_mAP"]
            > by_name["student_only_sp"]["selected_mAP"]
        ),
        "teacher_improves_it_axis": (
            by_name["verified_it_w1p0"]["selected_mAP"]
            > by_name["student_only_it"]["selected_mAP"]
        ),
        "matched_full_beats_student_only": (
            matched_full["selected_mAP"]
            > by_name["student_only_full"]["selected_mAP"]
        ),
        "matched_full_beats_shuffled_both": (
            matched_full["selected_mAP"]
            > by_name["control_shuffled_both"]["selected_mAP"]
        ),
    }
    analysis = {
        "protocol": "one-seed AFD mechanism/hyperparameter study; no significance claim",
        "student_objective": "lambda_afd_sp * AFD_SP + lambda_afd_it * AFD_IT",
        "legacy_main_student_losses": {"lambda_domain": 0, "lambda_modality": 0},
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "run_count": len(rows),
        "source_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT, text=True
        ).strip(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT, text=True
        ).strip()),
        "main_base": "b2d50842f7831c9eb14f06ddb6cbe5bbd22255b6",
        "best_verified_full": best,
        "comparisons": comparisons,
        "mechanism_checks": mechanism_checks,
        "rows": rows,
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (output / "scalar_curves.json").write_text(
        json.dumps(curves_by_condition), encoding="utf-8"
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("report") / path.relative_to(output))
    for run_name in (row["run"] for row in rows):
        tensorboard_run = PROJECT / "tb_logs" / run_name
        if tensorboard_run.is_dir():
            shutil.rmtree(tensorboard_run)
    print("Completed AFD runs:", len(rows))
    print("Best verified full:", best["name"], best["selected_mAP"])
    print("Teacher-use comparisons (mAP pp):", comparisons)
    print("Mechanism checks:", mechanism_checks)
    print("Main student losses used: no")
    print("Send this ZIP:", archive)
    try:
        from IPython.display import FileLink, display

        display(FileLink(str(archive)))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
