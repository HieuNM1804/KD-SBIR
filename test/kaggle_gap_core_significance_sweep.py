"""Tune only Gap-CoRe-sensitive parameters on a fixed seed-42 main model.

Paste this entire file into one offline Kaggle GPU cell after setup. The main
model, optimizer, prompts, teacher, and student seed stay fixed. A staged
one-factor, structural-interaction, and loss-shape search selects one Gap-CoRe
candidate, then evaluates shuffled and reversed controls. No weight is kept.
"""

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
LEGACY_TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/sketchy2_gap_core_teacher1_v8.pt"
)
TEACHER_CACHE = Path(
    "/kaggle/working/teacher_cache/sketchy2_cgrd_teacher1_v9.pt"
)
STAMP = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
RUN_PREFIX = "gap_core_seed42_sensitivity_" + STAMP
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
    retained_tags = {
        "mAP",
        "precision",
        "train_loss",
        *MECHANISM_TAGS,
        *(tag + "_epoch" for tag in MECHANISM_TAGS),
    }
    return {
        tag: [
            {"step": value.step, "value": value.value, "wall_time": value.wall_time}
            for value in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags().get("scalars", [])
        if tag in retained_tags
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
        f"Reusing format-v9 cache: {TEACHER_CACHE}\n", encoding="utf-8"
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
all_metrics = {}


def launch(condition, configuration):
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
        "--momentum",
        str(configuration["momentum"]),
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
    row, metrics, version = summarize(run_name, condition, code, configuration)
    all_metrics[run_name] = metrics
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
    return row, record


FIXED_MAIN = {
    "epochs": 3,
    "seed": 42,
    "n_ctx_visual": 3,
    "prompt_depth": 12,
    "lr": 1e-2,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "lambda_domain": 3.0,
    "lambda_modality": 1.0,
}
REFERENCE_GAP = {
    "lambda_gap_core": 0.5,
    "direction": "bidirectional",
    "min_correction": 0.05,
    "huber_beta": 0.02,
    "max_weight": 0.25,
    "control": "verified",
}
SENSITIVITY_GRIDS = {
    "lambda_gap_core": (0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 1.0, 1.5, 2.0),
    "direction": ("bidirectional", "sketch_to_photo", "photo_to_sketch"),
    "min_correction": (0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10),
    "huber_beta": (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15),
    "max_weight": (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50),
}


def make_configuration(**changes):
    configuration = {
        **FIXED_MAIN,
        **REFERENCE_GAP,
        "stage": "gap_search",
    }
    configuration.update(changes)
    return configuration


def completed(row, record):
    return record["return_code"] == 0 and "selected_mAP200" in row


main_configuration = make_configuration(
    lambda_gap_core=0.0,
    direction="none",
    min_correction=0.0,
    control="none",
    stage="fixed_main",
)
main_row, main_record = launch("fixed_main", main_configuration)
if not completed(main_row, main_record):
    raise RuntimeError("The fixed main run failed; inspect its retained log.")


SEARCH_KEY_FIELDS = (
    "lambda_gap_core",
    "direction",
    "min_correction",
    "huber_beta",
    "max_weight",
    "control",
)
verified_by_key = {}


def search_key(configuration):
    return tuple(configuration[name] for name in SEARCH_KEY_FIELDS)


def annotate_against_main(row):
    if "selected_mAP200" not in row:
        return
    row["delta_selected_mAP200_vs_main"] = (
        row["selected_mAP200"] - main_row["selected_mAP200"]
    )
    row["delta_selected_P200_vs_main"] = (
        row["selected_P200"] - main_row["selected_P200"]
    )
    row["delta_final_mAP200_vs_main"] = (
        row["final_mAP200"] - main_row["final_mAP200"]
    )
    row["delta_final_P200_vs_main"] = (
        row["final_P200"] - main_row["final_P200"]
    )


def launch_verified(condition, configuration):
    configuration = {**configuration, "control": "verified"}
    key = search_key(configuration)
    if key in verified_by_key:
        return (*verified_by_key[key], False)
    row, record = launch(condition, configuration)
    annotate_against_main(row)
    verified_by_key[key] = (row, record)
    return row, record, True


def ranking_key(item):
    row = item[0]
    return (row.get("selected_mAP200", -1.0), row.get("selected_P200", -1.0))


# Stage A: isolate each Gap-CoRe parameter around the pilot reference. This
# measures sensitivity before spending runs on interactions.
reference_configuration = make_configuration(stage="sensitivity_reference")
reference_row, reference_record, _ = launch_verified(
    "gap_reference", reference_configuration
)
if not completed(reference_row, reference_record):
    raise RuntimeError("The Gap-CoRe reference run failed.")

one_factor_rows = []
one_factor_results = {}
for parameter, values in SENSITIVITY_GRIDS.items():
    parameter_results = []
    for value in values:
        configuration = make_configuration(
            **{parameter: value},
            stage="sensitivity_" + parameter,
        )
        condition = "screen_" + parameter + "_" + safe_name(value)
        row, record, _ = launch_verified(condition, configuration)
        entry = {
            **row,
            "swept_parameter": parameter,
            "swept_value": value,
        }
        one_factor_rows.append(entry)
        if completed(row, record):
            parameter_results.append((row, record, value))
    parameter_results.sort(key=lambda item: ranking_key(item[:2]), reverse=True)
    one_factor_results[parameter] = parameter_results


def top_values(parameter, count):
    values = []
    for row, record, value in one_factor_results[parameter]:
        if value not in values:
            values.append(value)
        if len(values) == count:
            break
    return values


# Stage B: the strongest structural parameters interact. Test the four best
# lambda values, every direction, and the four best correction thresholds.
top_lambdas = top_values("lambda_gap_core", 4)
top_minimums = top_values("min_correction", 4)
directions = list(SENSITIVITY_GRIDS["direction"])
structural_results = []
for lambda_value, direction, minimum in itertools.product(
    top_lambdas,
    directions,
    top_minimums,
):
    configuration = make_configuration(
        lambda_gap_core=lambda_value,
        direction=direction,
        min_correction=minimum,
        stage="structural_interaction",
    )
    condition = (
        "structure_l"
        + safe_name(lambda_value)
        + "_"
        + direction
        + "_m"
        + safe_name(minimum)
    )
    row, record, _ = launch_verified(condition, configuration)
    if completed(row, record):
        structural_results.append((row, record))
structural_results.sort(key=ranking_key, reverse=True)
if not structural_results:
    raise RuntimeError("Every structural interaction run failed.")


# Stage C: for the three strongest structures, fully cross the four best Huber
# transitions and correction-weight caps found by one-factor screening.
top_betas = top_values("huber_beta", 4)
top_weights = top_values("max_weight", 4)
top_structures = structural_results[:3]
shape_results = []
for structural_row, structural_record in top_structures:
    structural_configuration = structural_record["configuration"]
    for beta, maximum in itertools.product(top_betas, top_weights):
        configuration = make_configuration(
            lambda_gap_core=structural_configuration["lambda_gap_core"],
            direction=structural_configuration["direction"],
            min_correction=structural_configuration["min_correction"],
            huber_beta=beta,
            max_weight=maximum,
            stage="loss_shape_interaction",
        )
        condition = (
            "shape_l"
            + safe_name(configuration["lambda_gap_core"])
            + "_"
            + configuration["direction"]
            + "_m"
            + safe_name(configuration["min_correction"])
            + "_b"
            + safe_name(beta)
            + "_w"
            + safe_name(maximum)
        )
        row, record, _ = launch_verified(condition, configuration)
        if completed(row, record):
            shape_results.append((row, record))


verified_ranking = [
    item for item in verified_by_key.values() if completed(item[0], item[1])
]
verified_ranking.sort(key=ranking_key, reverse=True)
if not verified_ranking:
    raise RuntimeError("Every verified Gap-CoRe run failed.")
best_row, best_record = verified_ranking[0]
selected_configuration = dict(best_record["configuration"])


# Controls use the exact selected loss hyperparameters on the same fixed main.
control_results = {}
for control in ("shuffled", "reversed"):
    configuration = {
        **selected_configuration,
        "control": control,
        "stage": "selected_control",
    }
    row, record = launch("best_" + control, configuration)
    annotate_against_main(row)
    control_results[control] = (row, record)


sensitivity = {}
for parameter, results in one_factor_results.items():
    successful = [item for item in results if "selected_mAP200" in item[0]]
    values = [item[0]["selected_mAP200"] for item in successful]
    sensitivity[parameter] = {
        "tested_values": list(SENSITIVITY_GRIDS[parameter]),
        "completed_values": len(successful),
        "best_value": successful[0][2] if successful else None,
        "best_selected_mAP200": successful[0][0]["selected_mAP200"] if successful else None,
        "selected_mAP200_span_pp": 100 * (max(values) - min(values)) if values else None,
    }
sensitivity_order = sorted(
    sensitivity,
    key=lambda name: sensitivity[name]["selected_mAP200_span_pp"] or -1.0,
    reverse=True,
)


verified_rows = [row for row, _ in verified_ranking]
write_csv(OUT / "one_factor_sensitivity.csv", one_factor_rows)
write_csv(OUT / "structural_interactions.csv", [row for row, _ in structural_results])
write_csv(OUT / "loss_shape_interactions.csv", [row for row, _ in shape_results])
write_csv(OUT / "verified_ranking.csv", verified_rows)
write_csv(OUT / "all_runs.csv", all_rows)


best_shuffled_row = control_results["shuffled"][0]
best_reversed_row = control_results["reversed"][0]
analysis = {
    "protocol": {
        "student_seed": 42,
        "student_epochs_per_run": 3,
        "main_hyperparameters_tuned": False,
        "fixed_main": FIXED_MAIN,
        "teacher_hyperparameters_tuned": False,
        "search_stages": [
            "one_factor_sensitivity",
            "lambda_direction_threshold_interaction",
            "huber_beta_max_weight_interaction",
            "selected_shuffled_and_reversed_controls",
        ],
        "selection_rule": "maximum selected mAP@200; selected P@200 breaks ties",
        "statistical_significance_assessed": False,
    },
    "sensitivity": sensitivity,
    "sensitivity_order": sensitivity_order,
    "selected_result": best_row,
    "selected_configuration": selected_configuration,
    "controls": {
        "shuffled": best_shuffled_row,
        "reversed": best_reversed_row,
    },
    "selected_deltas": {
        "verified_minus_main_mAP200_pp": 100
        * (best_row["selected_mAP200"] - main_row["selected_mAP200"]),
        "verified_minus_shuffled_mAP200_pp": 100
        * (best_row["selected_mAP200"] - best_shuffled_row["selected_mAP200"]),
        "verified_minus_reversed_mAP200_pp": 100
        * (best_row["selected_mAP200"] - best_reversed_row["selected_mAP200"]),
    },
    "interpretation": "seed42_candidate_selected_for_later_multiseed_confirmation",
}
(OUT / "analysis.json").write_text(
    json.dumps(analysis, indent=2), encoding="utf-8"
)
(OUT / "best_configuration.json").write_text(
    json.dumps(
        {
            "fixed_main": main_row,
            "best_verified": best_row,
            "best_configuration": selected_configuration,
            "controls": {
                "shuffled": best_shuffled_row,
                "reversed": best_reversed_row,
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
(OUT / "commands.json").write_text(
    json.dumps(
        [
            {key: value for key, value in record.items() if key != "log_path"}
            for record in all_runs
        ],
        indent=2,
    ),
    encoding="utf-8",
)


# Keep detailed curves and logs only for main, selected verified, controls, and
# failed runs. Summary rows for every candidate remain in compact CSV files.
detailed_records = {
    "main": main_record,
    "best_verified": best_record,
    "best_shuffled": control_results["shuffled"][1],
    "best_reversed": control_results["reversed"][1],
}
detailed_curves = []
for condition, record in detailed_records.items():
    metrics = all_metrics.get(record["run"], {})
    for tag in sorted(metrics):
        for value in metrics[tag]:
            detailed_curves.append(
                {
                    "condition": condition,
                    "run": record["run"],
                    "tag": tag,
                    **value,
                }
            )
    shutil.copy2(record["log_path"], OUT / (condition + ".log"))
write_csv(OUT / "best_scalar_curves.csv", detailed_curves)
shutil.copy2(cache_log, OUT / "gap_cache.log")
failed_records = [record for record in all_runs if record["return_code"] != 0]
for index, record in enumerate(failed_records):
    shutil.copy2(
        record["log_path"],
        OUT / f"failed_{index:02d}_{record['condition']}.log",
    )


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
comparison = (
    ("main", main_row),
    ("verified", best_row),
    ("shuffled", best_shuffled_row),
    ("reversed", best_reversed_row),
)
axes[0].bar(
    [name for name, _ in comparison],
    [100 * row["selected_mAP200"] for _, row in comparison],
    color=("#7f7f7f", "#2ca02c", "#d62728", "#9467bd"),
)
axes[0].set_ylabel("Selected mAP@200 (%)")
axes[0].tick_params(axis="x", rotation=20)
spans = [sensitivity[name]["selected_mAP200_span_pp"] for name in sensitivity_order]
axes[1].barh(sensitivity_order[::-1], spans[::-1], color="#1f77b4")
axes[1].set_xlabel("One-factor mAP span (pp)")
axes[1].set_title("Gap-CoRe parameter sensitivity")
top_rows = verified_rows[:20][::-1]
axes[2].barh(
    [str(index + 1) for index in range(len(top_rows))],
    [100 * row["delta_selected_mAP200_vs_main"] for row in top_rows],
    color="#2ca02c",
)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_xlabel("Verified - fixed main mAP (pp)")
axes[2].set_ylabel("Top candidate rank (reversed)")
fig.tight_layout()
fig.savefig(OUT / "seed42_sensitivity.png", dpi=170)
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
    "student_seed": 42,
    "student_epochs_per_run": 3,
    "main_hyperparameters_tuned": False,
    "fixed_main": FIXED_MAIN,
    "teacher_cache": str(TEACHER_CACHE),
    "teacher_training_seed": 42,
    "teacher_pretrain_epochs": 1,
    "teacher_hyperparameters_tuned": False,
    "unique_verified_gap_runs": len(verified_by_key),
    "structural_interaction_results": len(structural_results),
    "loss_shape_interaction_results": len(shape_results),
    "control_runs": list(control_results),
    "completed_runs": sum(record["return_code"] == 0 for record in all_runs),
    "failed_runs": len(failed_records),
    "checkpoint_creation_disabled": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "fallback_checkpoint_cleanup": not TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "checkpoint_files_found": checkpoint_files,
    "weights_included": False,
    "statistical_significance_assessed": False,
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

display(Image(filename=str(OUT / "seed42_sensitivity.png")))
print("Completed runs:", manifest["completed_runs"])
print("Main hyperparameters tuned: no")
print("Most sensitive Gap-CoRe parameters:", sensitivity_order)
print("Selected Gap-CoRe configuration:", selected_configuration)
print(
    "Verified-main selected mAP delta (pp):",
    analysis["selected_deltas"]["verified_minus_main_mAP200_pp"],
)
print(
    "Verified-shuffled selected mAP delta (pp):",
    analysis["selected_deltas"]["verified_minus_shuffled_mAP200_pp"],
)
print("Statistical significance assessed: no (seed 42 tuning only)")
print("Weights/checkpoints included: no")
print("Send this ZIP:", archive)
display(FileLink(str(archive)))
