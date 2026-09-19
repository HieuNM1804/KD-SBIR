"""Tune Gap-CoRe, then run checkpoint-free paired confirmation on new seeds.

Paste this entire file into one offline Kaggle GPU cell after setup. Search is
performed on student seed 42. The selected configuration is frozen before
paired main/verified/shuffled confirmation on seeds 43--47.
"""

import csv
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
LEGACY_TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/sketchy2_core_teacher1_v7.pt"
)
TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/sketchy2_gap_core_teacher1_v8.pt"
)
STAMP = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
RUN_PREFIX = "gap_core_significance_" + STAMP
OUT = Path("/kaggle/working") / RUN_PREFIX
WORK_LOGS = OUT / "work_logs"
OUT.mkdir(parents=True, exist_ok=False)
WORK_LOGS.mkdir(parents=True, exist_ok=False)
TEACHER_CACHE.parent.mkdir(parents=True, exist_ok=True)

# Existing Kaggle sessions may contain the implementation immediately before
# fixed teacher/student seeds were separated. Patch only the cache metadata key
# in that case; new pinned bundles already include the CLI contract.
train_source = (PROJECT / "src" / "train.py").read_text(encoding="utf-8")
model_path = PROJECT / "src" / "model.py"
model_source = model_path.read_text(encoding="utf-8")
TRAIN_SUPPORTS_NO_CHECKPOINTS = "--no_checkpoints" in train_source
TRAIN_SUPPORTS_TEACHER_TRAINING_SEED = "--teacher_training_seed" in train_source
if not TRAIN_SUPPORTS_TEACHER_TRAINING_SEED:
    old = '        "seed": args.seed,\n'
    new = '        "seed": getattr(args, "teacher_training_seed", 42),\n'
    if old in model_source:
        model_path.write_text(model_source.replace(old, new, 1), encoding="utf-8")
        print("[Compatibility] fixed teacher cache identity at seed 42.")
    elif new not in model_source:
        raise RuntimeError("Cannot apply fixed-teacher compatibility patch.")


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def run_stage(label, command):
    log_path = WORK_LOGS / f"{label}.log"
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
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            log.write(line)
        code = process.wait()
    print(f"[{label}] exit code: {code}", flush=True)
    if code != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-120:]), flush=True)
    return code, log_path


def latest_version(run_name):
    versions = list((PROJECT / "tb_logs" / run_name).glob("version_*"))
    if not versions:
        return None
    return max(versions, key=lambda path: int(path.name.split("_")[-1]))


def read_scalars(version):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(version), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [
            {"step": value.step, "value": value.value, "wall_time": value.wall_time}
            for value in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags().get("scalars", [])
    }


MECHANISM_TAGS = (
    "gap_core_coverage",
    "gap_core_teacher_correction",
    "gap_core_student_correction",
    "gap_core_agreement",
    "gap_core_absolute_error",
    "gap_core_common_margin",
    "gap_core_full_margin",
    "gap_core_grad_ratio",
    "gap_core_grad_cosine",
)


def summarize(run_name, condition, return_code, configuration):
    version = latest_version(run_name)
    row = {
        "condition": condition,
        "run": run_name,
        "return_code": return_code,
        **configuration,
    }
    if version is None:
        return row, {}, None
    row["version"] = version.name
    metrics = read_scalars(version)
    maps = metrics.get("mAP", [])
    precisions = metrics.get("precision", [])
    if maps and precisions:
        map_by_step = {value["step"]: value["value"] for value in maps}
        selected_index, selected_precision = max(
            enumerate(precisions), key=lambda item: item[1]["value"]
        )
        selected_step = selected_precision["step"]
        row.update(
            {
                "selected_validation_epoch": selected_index + 1,
                "selected_step": selected_step,
                "selected_mAP200": map_by_step[selected_step],
                "selected_P200": selected_precision["value"],
                "final_mAP200": maps[-1]["value"],
                "final_P200": precisions[-1]["value"],
                "best_mAP200": max(value["value"] for value in maps),
                "best_P200": max(value["value"] for value in precisions),
                "completed_validation_epochs": len(maps),
            }
        )
    for tag in MECHANISM_TAGS:
        values = metrics.get(tag, []) or metrics.get(tag + "_epoch", [])
        row[tag + "_final"] = values[-1]["value"] if values else None
        row[tag + "_mean"] = (
            sum(value["value"] for value in values) / len(values)
            if values
            else None
        )
    return row, metrics, version


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


fixed = [
    "--root",
    ROOT,
    "--dataset",
    "sketchy_2",
    "--batch_size",
    "64",
    "--test_batch_size",
    "1024",
    "--workers",
    "8",
    "--teacher_pretrain_epochs",
    "1",
    "--teacher_cache_path",
    str(TEACHER_CACHE),
    "--teacher_pretrain_batch_size",
    "64",
    "--teacher_n_ctx_visual",
    "10",
    "--teacher_prompt_depth",
    "12",
    "--teacher_prompt_std",
    "0.02",
    "--teacher_prompt_lr",
    "3e-2",
    "--teacher_prompt_seed",
    "42",
    "--teacher_prompt_gradient_checkpointing",
    "--teacher_momentum",
    "0.9",
    "--teacher_weight_decay",
    "1e-3",
    "--lambda_teacher_retrieval",
    "1.5",
    "--teacher_triplet_margin",
    "0.2",
    "--photo_text_kd_temperature",
    "0.15",
    "--sketch_text_kd_temperature",
    "0.02",
    "--momentum",
    "0.9",
    "--lambda_core",
    "0.0",
    "--no_progress",
]
if TRAIN_SUPPORTS_TEACHER_TRAINING_SEED:
    fixed.extend(["--teacher_training_seed", "42"])
if TRAIN_SUPPORTS_NO_CHECKPOINTS:
    fixed.append("--no_checkpoints")
else:
    print(
        "[Compatibility] checkpoints will be deleted immediately after each run.",
        flush=True,
    )


if TEACHER_CACHE.is_file():
    cache_code = 0
    cache_log = WORK_LOGS / "gap_cache.log"
    cache_log.write_text(
        f"Reusing format-v8 cache: {TEACHER_CACHE}\n", encoding="utf-8"
    )
elif LEGACY_TEACHER_CACHE.is_file():
    cache_code, cache_log = run_stage(
        "gap_cache",
        [
            sys.executable,
            "-u",
            "-m",
            "src.gap_core_cache",
            "--source",
            str(LEGACY_TEACHER_CACHE),
            "--destination",
            str(TEACHER_CACHE),
            "--root",
            ROOT,
            "--dataset",
            "sketchy_2",
            "--batch_size",
            "64",
            "--workers",
            "8",
        ],
    )
else:
    cache_code, cache_log = run_stage(
        "gap_cache",
        [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            *fixed,
            "--epochs",
            "3",
            "--n_ctx_visual",
            "3",
            "--prompt_depth",
            "12",
            "--lr",
            "1e-2",
            "--weight_decay",
            "5e-4",
            "--seed",
            "42",
            "--lambda_domain",
            "3.0",
            "--lambda_modality",
            "1.0",
            "--lambda_gap_core",
            "0.0",
            "--teacher_cache_only",
            "--exp_name",
            RUN_PREFIX + "_cache",
        ],
    )
if cache_code != 0 or not TEACHER_CACHE.is_file():
    shutil.copy2(cache_log, OUT / "gap_cache_failed.log")
    raise RuntimeError("Teacher cache preparation failed.")


all_rows = []
all_runs = []


def launch(condition, configuration, retain_metrics=False):
    index = len(all_runs)
    label = f"{index:03d}_{condition}"
    run_name = RUN_PREFIX + f"_{index:03d}_{condition}"
    arguments = [
        "--epochs",
        str(configuration["epochs"]),
        "--n_ctx_visual",
        str(configuration["n_ctx_visual"]),
        "--prompt_depth",
        str(configuration["prompt_depth"]),
        "--lr",
        str(configuration["lr"]),
        "--weight_decay",
        str(configuration["weight_decay"]),
        "--seed",
        str(configuration["seed"]),
        "--lambda_domain",
        str(configuration["lambda_domain"]),
        "--lambda_modality",
        str(configuration["lambda_modality"]),
        "--lambda_gap_core",
        str(configuration["lambda_gap_core"]),
    ]
    if configuration["lambda_gap_core"] > 0:
        arguments.extend(
            [
                "--gap_core_control",
                configuration["control"],
                "--gap_core_direction",
                configuration["direction"],
                "--gap_core_huber_beta",
                str(configuration["huber_beta"]),
                "--gap_core_min_correction",
                str(configuration["min_correction"]),
                "--gap_core_max_weight",
                str(configuration["max_weight"]),
            ]
        )
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        *fixed,
        *arguments,
        "--exp_name",
        run_name,
    ]
    code, log_path = run_stage(label, command)
    row, metrics, version = summarize(
        run_name, condition, code, configuration
    )
    checkpoint_directory = PROJECT / "saved_models" / run_name
    checkpoint_bytes_removed = 0
    if checkpoint_directory.is_dir():
        checkpoint_bytes_removed = sum(
            path.stat().st_size
            for path in checkpoint_directory.rglob("*")
            if path.is_file()
        )
        shutil.rmtree(checkpoint_directory)
    tensorboard_directory = PROJECT / "tb_logs" / run_name
    if tensorboard_directory.is_dir():
        shutil.rmtree(tensorboard_directory)
    record = {
        "condition": condition,
        "run": run_name,
        "configuration": configuration,
        "command": command,
        "return_code": code,
        "log_path": str(log_path),
        "checkpoint_bytes_removed": checkpoint_bytes_removed,
    }
    all_rows.append(row)
    all_runs.append(record)
    return row, (metrics if retain_metrics else {}), record


def base_configuration(**changes):
    configuration = {
        "epochs": 3,
        "seed": 42,
        "n_ctx_visual": 3,
        "prompt_depth": 12,
        "lr": 1e-2,
        "weight_decay": 5e-4,
        "lambda_domain": 3.0,
        "lambda_modality": 1.0,
        "lambda_gap_core": 0.0,
        "direction": "none",
        "min_correction": 0.0,
        "huber_beta": 0.05,
        "max_weight": 0.25,
        "control": "none",
        "stage": "baseline_search",
    }
    configuration.update(changes)
    return configuration


# Stage A: tune the main model without observing Gap-CoRe results.
base_candidates = []
for domain, modality in itertools.product(
    (1.0, 1.5, 2.0, 3.0),
    (0.25, 0.5, 1.0),
):
    base_candidates.append(
        base_configuration(lambda_domain=domain, lambda_modality=modality)
    )
for n_ctx, depth in (
    (2, 6),
    (2, 12),
    (3, 6),
    (3, 9),
    (3, 12),
    (4, 6),
    (4, 9),
    (4, 12),
    (6, 12),
):
    base_candidates.append(
        base_configuration(n_ctx_visual=n_ctx, prompt_depth=depth)
    )
for learning_rate, decay in itertools.product(
    (5e-3, 1e-2, 2e-2),
    (1e-4, 5e-4, 1e-3),
):
    base_candidates.append(
        base_configuration(lr=learning_rate, weight_decay=decay)
    )

unique_base_candidates = []
seen_base_keys = set()
base_names = (
    "n_ctx_visual",
    "prompt_depth",
    "lr",
    "weight_decay",
    "lambda_domain",
    "lambda_modality",
)
for configuration in base_candidates:
    key = tuple(configuration[name] for name in base_names)
    if key not in seen_base_keys:
        seen_base_keys.add(key)
        configuration["base_id"] = "base_" + str(len(unique_base_candidates))
        unique_base_candidates.append(configuration)

base_results = []
for configuration in unique_base_candidates:
    row, _, record = launch(configuration["base_id"], configuration)
    if record["return_code"] == 0 and "selected_mAP200" in row:
        base_results.append((row, record))
if len(base_results) < 2:
    raise RuntimeError("Fewer than two baseline-search runs completed.")
base_results.sort(
    key=lambda item: (item[0]["selected_mAP200"], item[0]["selected_P200"]),
    reverse=True,
)
top_bases = base_results[:2]
matched_base_rows = {row["base_id"]: row for row, _ in base_results}


# Stage B: tune Gap-CoRe only on the two bases selected without using Gap loss.
gap_results = []
for base_row, base_record in top_bases:
    base = dict(base_record["configuration"])
    for lambda_value, direction, minimum in itertools.product(
        (1.0, 2.0, 4.0),
        ("bidirectional", "sketch_to_photo"),
        (0.0, 0.02),
    ):
        configuration = {
            **base,
            "lambda_gap_core": lambda_value,
            "direction": direction,
            "min_correction": minimum,
            "huber_beta": 0.05,
            "max_weight": 0.25,
            "control": "verified",
            "stage": "gap_search",
        }
        condition = (
            base["base_id"]
            + "_gap_l"
            + safe_name(lambda_value)
            + "_"
            + direction
            + "_m"
            + safe_name(minimum)
        )
        row, _, record = launch(condition, configuration)
        if record["return_code"] == 0 and "selected_mAP200" in row:
            matched = matched_base_rows[base["base_id"]]
            row["delta_selected_mAP200_vs_matched_main"] = (
                row["selected_mAP200"] - matched["selected_mAP200"]
            )
            row["delta_selected_P200_vs_matched_main"] = (
                row["selected_P200"] - matched["selected_P200"]
            )
            gap_results.append((row, record))
if not gap_results:
    raise RuntimeError("Every Gap-CoRe search run failed.")
gap_results.sort(
    key=lambda item: (item[0]["selected_mAP200"], item[0]["selected_P200"]),
    reverse=True,
)


# Stage C: refine loss curvature and correction clipping around top two.
refinement_results = []
seen_gap_keys = set()
gap_names = (
    "base_id",
    "lambda_gap_core",
    "direction",
    "min_correction",
    "huber_beta",
    "max_weight",
)
for row, _ in gap_results:
    seen_gap_keys.add(tuple(row[name] for name in gap_names))
for top_row, top_record in gap_results[:2]:
    base = dict(top_record["configuration"])
    variants = []
    for beta in (0.02, 0.10):
        variants.append({**base, "huber_beta": beta})
    for maximum in (0.15, 0.35):
        variants.append({**base, "max_weight": maximum})
    for configuration in variants:
        configuration["stage"] = "gap_refinement"
        key = tuple(configuration[name] for name in gap_names)
        if key in seen_gap_keys:
            continue
        seen_gap_keys.add(key)
        condition = (
            configuration["base_id"]
            + "_refine_b"
            + safe_name(configuration["huber_beta"])
            + "_w"
            + safe_name(configuration["max_weight"])
        )
        row, _, record = launch(condition, configuration)
        if record["return_code"] == 0 and "selected_mAP200" in row:
            matched = matched_base_rows[configuration["base_id"]]
            row["delta_selected_mAP200_vs_matched_main"] = (
                row["selected_mAP200"] - matched["selected_mAP200"]
            )
            row["delta_selected_P200_vs_matched_main"] = (
                row["selected_P200"] - matched["selected_P200"]
            )
            refinement_results.append((row, record))

all_gap_candidates = gap_results + refinement_results
all_gap_candidates.sort(
    key=lambda item: (item[0]["selected_mAP200"], item[0]["selected_P200"]),
    reverse=True,
)
best_search_row, best_search_record = all_gap_candidates[0]
selected_configuration = dict(best_search_record["configuration"])


# Stage D: freeze hyperparameters and confirm on five unseen student seeds.
confirmation_rows = []
confirmation_metric_sets = []
confirmation_records = []
for seed in (43, 44, 45, 46, 47):
    for control in ("main", "verified", "shuffled"):
        configuration = {
            **selected_configuration,
            "epochs": 5,
            "seed": seed,
            "stage": "confirmation",
            "control": control,
            "lambda_gap_core": (
                0.0 if control == "main" else selected_configuration["lambda_gap_core"]
            ),
            "direction": (
                "none" if control == "main" else selected_configuration["direction"]
            ),
        }
        condition = f"confirm_s{seed}_{control}"
        row, metrics, record = launch(condition, configuration, retain_metrics=True)
        confirmation_rows.append(row)
        confirmation_metric_sets.append((condition, record, metrics))
        confirmation_records.append(record)

# One reversed run is diagnostic; significance is assessed against shuffled.
reverse_configuration = {
    **selected_configuration,
    "epochs": 5,
    "seed": 43,
    "stage": "confirmation_control",
    "control": "reversed",
}
reverse_row, reverse_metrics, reverse_record = launch(
    "confirm_s43_reversed", reverse_configuration, retain_metrics=True
)
confirmation_rows.append(reverse_row)
confirmation_metric_sets.append(
    ("confirm_s43_reversed", reverse_record, reverse_metrics)
)
confirmation_records.append(reverse_record)


def exact_sign_flip_p(values):
    """Exact one-sided paired randomization p-value for a positive mean."""
    values = [float(value) for value in values]
    observed = statistics.mean(values)
    if not values or observed <= 0:
        return 1.0
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = statistics.mean(
            sign * value for sign, value in zip(signs, values)
        )
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / total


def paired_statistics(values):
    values = [float(value) for value in values]
    count = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    # t(0.975, 4) for the preregistered five confirmation seeds.
    critical = 2.776 if count == 5 else 1.96
    half_width = critical * std / math.sqrt(count) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "positive_seeds": sum(value > 0 for value in values),
        "exact_one_sided_sign_flip_p": exact_sign_flip_p(values),
        "values": values,
    }


by_seed_condition = {
    (int(row["seed"]), row["control"]): row
    for row in confirmation_rows
    if "selected_mAP200" in row and row["control"] in {"main", "verified", "shuffled"}
}
paired_report = {}
for metric in (
    "selected_mAP200",
    "selected_P200",
    "final_mAP200",
    "final_P200",
):
    verified_main = []
    verified_shuffled = []
    for seed in (43, 44, 45, 46, 47):
        main = by_seed_condition.get((seed, "main"))
        verified = by_seed_condition.get((seed, "verified"))
        shuffled = by_seed_condition.get((seed, "shuffled"))
        if main is not None and verified is not None:
            verified_main.append(verified[metric] - main[metric])
        if shuffled is not None and verified is not None:
            verified_shuffled.append(verified[metric] - shuffled[metric])
    paired_report[metric] = {
        "verified_minus_main": paired_statistics(verified_main),
        "verified_minus_shuffled": paired_statistics(verified_shuffled),
    }

primary_main = paired_report["selected_mAP200"]["verified_minus_main"]
primary_shuffle = paired_report["selected_mAP200"]["verified_minus_shuffled"]
claim_checks = {
    "five_confirmation_seeds_complete": primary_main["n"] == 5
    and primary_shuffle["n"] == 5,
    "verified_improves_selected_mAP_mean": primary_main["mean"] > 0,
    "verified_vs_main_exact_p_le_0.05": (
        primary_main["exact_one_sided_sign_flip_p"] <= 0.05
    ),
    "verified_vs_main_ci95_above_zero": primary_main["ci95_low"] > 0,
    "verified_beats_shuffled_selected_mAP_mean": primary_shuffle["mean"] > 0,
    "verified_vs_shuffled_exact_p_le_0.05": (
        primary_shuffle["exact_one_sided_sign_flip_p"] <= 0.05
    ),
    "verified_vs_shuffled_ci95_above_zero": primary_shuffle["ci95_low"] > 0,
}
claim_checks["significant"] = all(claim_checks.values())


base_rows = [row for row, _ in base_results]
base_rows.sort(
    key=lambda row: (row["selected_mAP200"], row["selected_P200"]),
    reverse=True,
)
gap_rows = [row for row, _ in all_gap_candidates]
gap_rows.sort(
    key=lambda row: (row["selected_mAP200"], row["selected_P200"]),
    reverse=True,
)
write_csv(OUT / "baseline_search.csv", base_rows)
write_csv(OUT / "gap_search.csv", gap_rows)
write_csv(OUT / "confirmation_runs.csv", confirmation_rows)
write_csv(OUT / "all_runs.csv", all_rows)

analysis = {
    "selection_protocol": {
        "tuning_seed": 42,
        "confirmation_seeds": [43, 44, 45, 46, 47],
        "baseline_selection": (
            "top two selected mAP@200/P@200 using only lambda_gap_core=0"
        ),
        "gap_selection": (
            "highest selected mAP@200, then P@200, within the two locked bases"
        ),
        "primary_confirmation_metric": "selected_mAP200",
        "significance_test": "exact one-sided paired sign-flip over five new seeds",
    },
    "selected_search_result": best_search_row,
    "selected_configuration": selected_configuration,
    "paired_confirmation": paired_report,
    "claim_checks": claim_checks,
    "interpretation": (
        "claim_supported"
        if claim_checks["significant"]
        else "claim_not_yet_supported"
    ),
}
(OUT / "analysis.json").write_text(
    json.dumps(analysis, indent=2), encoding="utf-8"
)
(OUT / "commands.json").write_text(
    json.dumps(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"log_path"}
            }
            for record in all_runs
        ],
        indent=2,
    ),
    encoding="utf-8",
)


# Detailed curves and logs are kept only for confirmation runs and failures.
confirmation_curves = []
for condition, record, metrics in confirmation_metric_sets:
    for tag in sorted(metrics):
        for value in metrics[tag]:
            confirmation_curves.append(
                {
                    "condition": condition,
                    "run": record["run"],
                    "tag": tag,
                    **value,
                }
            )
write_csv(OUT / "confirmation_scalar_curves.csv", confirmation_curves)
shutil.copy2(cache_log, OUT / "gap_cache.log")
for record in confirmation_records:
    shutil.copy2(
        record["log_path"], OUT / (record["condition"] + ".log")
    )
failed_records = [record for record in all_runs if record["return_code"] != 0]
for index, record in enumerate(failed_records):
    shutil.copy2(
        record["log_path"],
        OUT / f"failed_{index:02d}_{record['condition']}.log",
    )


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

seeds = [43, 44, 45, 46, 47]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for condition, color in (
    ("main", "#7f7f7f"),
    ("verified", "#2ca02c"),
    ("shuffled", "#d62728"),
):
    values = [
        100 * by_seed_condition[(seed, condition)]["selected_mAP200"]
        for seed in seeds
        if (seed, condition) in by_seed_condition
    ]
    axes[0].plot(seeds[: len(values)], values, marker="o", label=condition, color=color)
axes[0].set_xlabel("Confirmation student seed")
axes[0].set_ylabel("Selected mAP@200 (%)")
axes[0].legend()
axes[0].grid(alpha=0.2)
verified_main_pp = [100 * value for value in primary_main["values"]]
verified_shuffle_pp = [100 * value for value in primary_shuffle["values"]]
x = list(range(len(seeds)))
width = 0.36
axes[1].bar(
    [value - width / 2 for value in x],
    verified_main_pp,
    width,
    label="verified - main",
)
axes[1].bar(
    [value + width / 2 for value in x],
    verified_shuffle_pp,
    width,
    label="verified - shuffled",
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xticks(x, [str(seed) for seed in seeds])
axes[1].set_xlabel("Confirmation student seed")
axes[1].set_ylabel("Selected mAP@200 delta (pp)")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "confirmation.png", dpi=170)
plt.close(fig)


source_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
checkpoint_files = [
    str(path)
    for record in all_runs
    for path in (PROJECT / "saved_models" / record["run"]).glob("*.ckpt")
]
manifest = {
    "created": datetime.now(UTC).isoformat(),
    "source_commit": source_commit,
    "dataset": "sketchy_2",
    "teacher_cache": str(TEACHER_CACHE),
    "teacher_training_seed": 42,
    "teacher_pretrain_epochs": 1,
    "baseline_search_runs": len(unique_base_candidates),
    "gap_search_runs": len(gap_results),
    "gap_refinement_runs": len(refinement_results),
    "confirmation_runs": len(confirmation_rows),
    "completed_runs": sum(record["return_code"] == 0 for record in all_runs),
    "failed_runs": len(failed_records),
    "checkpoint_creation_disabled": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "fallback_checkpoint_cleanup": not TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "checkpoint_files_found": checkpoint_files,
    "weights_included": False,
    "claim_supported": claim_checks["significant"],
}
(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
if checkpoint_files:
    raise RuntimeError("Checkpoint files remain after cleanup.")

shutil.rmtree(WORK_LOGS)
archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("report") / path.relative_to(OUT))

from IPython.display import FileLink, Image, display

display(Image(filename=str(OUT / "confirmation.png")))
print("Completed runs:", manifest["completed_runs"])
print("Selected configuration:", selected_configuration)
print("Verified-main selected mAP mean delta:", primary_main["mean"])
print("Verified-main exact p:", primary_main["exact_one_sided_sign_flip_p"])
print("Verified-shuffled exact p:", primary_shuffle["exact_one_sided_sign_flip_p"])
print("Significant claim supported:", claim_checks["significant"])
print("Weights/checkpoints included: no")
print("Send this ZIP:", archive)
display(FileLink(str(archive)))
