"""Run a checkpoint-free two-stage Gap-CoRe sweep and export a compact ZIP.

Paste this entire file into one offline Kaggle GPU cell after
`kaggle_gap_core_offline.py`. A compatible format-v9 teacher cache is reused.
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
RUN_PREFIX = "gap_core_sweep_sketchy2_s42_" + STAMP
OUT = Path("/kaggle/working") / ("gap_core_sweep_" + STAMP)
WORK_LOGS = OUT / "work_logs"
OUT.mkdir(parents=True, exist_ok=False)
WORK_LOGS.mkdir(parents=True, exist_ok=False)
TEACHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
TRAIN_SUPPORTS_NO_CHECKPOINTS = "--no_checkpoints" in (
    PROJECT / "src" / "train.py"
).read_text(encoding="utf-8")


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
        print("\n".join(lines[-100:]), flush=True)
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


def summarize(run_name, condition, return_code, configuration):
    version = latest_version(run_name)
    if version is None:
        return {
            "condition": condition,
            "run": run_name,
            "return_code": return_code,
            **configuration,
        }, {}, None
    metrics = read_scalars(version)
    maps = metrics.get("mAP", [])
    precisions = metrics.get("precision", [])
    row = {
        "condition": condition,
        "run": run_name,
        "version": version.name,
        "return_code": return_code,
        **configuration,
    }
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
    mechanism_tags = (
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
    for tag in mechanism_tags:
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


shared = [
    "--root",
    ROOT,
    "--dataset",
    "sketchy_2",
    "--epochs",
    "5",
    "--workers",
    "8",
    "--batch_size",
    "64",
    "--test_batch_size",
    "1024",
    "--n_ctx_visual",
    "3",
    "--prompt_depth",
    "12",
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
    "--lr",
    "1e-2",
    "--momentum",
    "0.9",
    "--weight_decay",
    "5e-4",
    "--seed",
    "42",
    "--lambda_domain",
    "3.0",
    "--lambda_modality",
    "1.0",
    "--lambda_core",
    "0.0",
    "--no_progress",
]
if TRAIN_SUPPORTS_NO_CHECKPOINTS:
    shared.append("--no_checkpoints")
else:
    print(
        "[Compatibility] source lacks --no_checkpoints; weights will be "
        "deleted immediately after every run.",
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
            *shared,
            "--lambda_gap_core",
            "0.0",
            "--teacher_cache_only",
            "--exp_name",
            RUN_PREFIX + "_cache",
        ],
    )
if cache_code != 0 or not TEACHER_CACHE.is_file():
    shutil.copy2(cache_log, OUT / "gap_cache_failed.log")
    raise RuntimeError("Gap-CoRe cache preparation failed; inspect the retained log.")


all_rows = []
all_runs = []


def launch(condition, configuration):
    index = len(all_runs)
    label = f"{index:02d}_{condition}"
    run_name = RUN_PREFIX + f"_{index:02d}_{condition}"
    arguments = [
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
        *shared,
        *arguments,
        "--exp_name",
        run_name,
    ]
    code, log_path = run_stage(label, command)
    row, metrics, version = summarize(
        run_name,
        condition,
        code,
        configuration,
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
    record = {
        "condition": condition,
        "run": run_name,
        "configuration": configuration,
        "command": command,
        "return_code": code,
        "log_path": str(log_path),
        "version": str(version) if version is not None else None,
        "checkpoint_bytes_removed": checkpoint_bytes_removed,
    }
    all_runs.append(record)
    all_rows.append(row)
    return row, metrics, record


main_configuration = {
    "lambda_gap_core": 0.0,
    "direction": "none",
    "min_correction": 0.0,
    "huber_beta": 0.05,
    "max_weight": 0.25,
    "control": "none",
    "stage": "baseline",
}
main_row, main_metrics, main_record = launch("main", main_configuration)
if main_record["return_code"] != 0 or "selected_mAP200" not in main_row:
    shutil.copy2(main_record["log_path"], OUT / "main_failed.log")
    raise RuntimeError("Matched main failed; inspect the retained log.")


# Stage 1: broad, interpretable grid over strength, direction, and target quality.
stage1_configurations = []
for lambda_value, direction, minimum in itertools.product(
    (0.5, 1.0, 2.0, 4.0),
    ("bidirectional", "sketch_to_photo"),
    (0.0, 0.02, 0.05),
):
    stage1_configurations.append(
        {
            "lambda_gap_core": lambda_value,
            "direction": direction,
            "min_correction": minimum,
            "huber_beta": 0.05,
            "max_weight": 0.25,
            "control": "verified",
            "stage": "core_grid",
        }
    )

candidate_records = []
for configuration in stage1_configurations:
    condition = (
        "verified_l"
        + safe_name(configuration["lambda_gap_core"])
        + "_"
        + configuration["direction"]
        + "_m"
        + safe_name(configuration["min_correction"])
    )
    row, metrics, record = launch(condition, configuration)
    if record["return_code"] == 0 and "selected_mAP200" in row:
        candidate_records.append((row, metrics, record))

if not candidate_records:
    raise RuntimeError("Every verified core-grid run failed.")


def rank_key(item):
    row = item[0]
    return (row["selected_mAP200"], row["selected_P200"])


top_stage1 = sorted(candidate_records, key=rank_key, reverse=True)[:2]

# Stage 2: refine loss shape and outlier clipping around the two best core runs.
seen_keys = {
    tuple(
        row[0][name]
        for name in (
            "lambda_gap_core",
            "direction",
            "min_correction",
            "huber_beta",
            "max_weight",
        )
    )
    for row in candidate_records
}
stage2_configurations = []
for top_row, _, _ in top_stage1:
    base = {
        "lambda_gap_core": top_row["lambda_gap_core"],
        "direction": top_row["direction"],
        "min_correction": top_row["min_correction"],
        "huber_beta": top_row["huber_beta"],
        "max_weight": top_row["max_weight"],
        "control": "verified",
        "stage": "loss_refinement",
    }
    variants = []
    for beta in (0.02, 0.10):
        variants.append({**base, "huber_beta": beta})
    for maximum in (0.15, 0.35):
        variants.append({**base, "max_weight": maximum})
    for configuration in variants:
        key = tuple(
            configuration[name]
            for name in (
                "lambda_gap_core",
                "direction",
                "min_correction",
                "huber_beta",
                "max_weight",
            )
        )
        if key not in seen_keys:
            seen_keys.add(key)
            stage2_configurations.append(configuration)

for configuration in stage2_configurations:
    condition = (
        "refine_l"
        + safe_name(configuration["lambda_gap_core"])
        + "_"
        + configuration["direction"]
        + "_m"
        + safe_name(configuration["min_correction"])
        + "_b"
        + safe_name(configuration["huber_beta"])
        + "_w"
        + safe_name(configuration["max_weight"])
    )
    row, metrics, record = launch(condition, configuration)
    if record["return_code"] == 0 and "selected_mAP200" in row:
        candidate_records.append((row, metrics, record))

best_row, best_metrics, best_record = max(candidate_records, key=rank_key)
best_configuration = dict(best_record["configuration"])

control_results = {}
for control in ("shuffled", "reversed"):
    configuration = {
        **best_configuration,
        "control": control,
        "stage": "best_control",
    }
    row, metrics, record = launch("best_" + control, configuration)
    control_results[control] = (row, metrics, record)


# Add matched-main deltas to every completed row.
for row in all_rows:
    if "selected_mAP200" not in row:
        continue
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

completed_candidates = [
    row
    for row in all_rows
    if row.get("control") == "verified" and "selected_mAP200" in row
]
completed_candidates.sort(
    key=lambda row: (row["selected_mAP200"], row["selected_P200"]),
    reverse=True,
)
write_csv(OUT / "sweep_summary.csv", all_rows)
write_csv(OUT / "verified_ranking.csv", completed_candidates)

best_payload = {
    "selection_rule": (
        "maximum mAP@200 at the P@200-selected checkpoint; P@200 breaks ties"
    ),
    "main": main_row,
    "best_verified": best_row,
    "best_configuration": best_configuration,
    "controls": {
        name: values[0] for name, values in control_results.items()
    },
}
(OUT / "best_configuration.json").write_text(
    json.dumps(best_payload, indent=2), encoding="utf-8"
)
(OUT / "commands.json").write_text(
    json.dumps(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"log_path", "version"}
            }
            for record in all_runs
        ],
        indent=2,
    ),
    encoding="utf-8",
)


# Retain detailed scalar curves only for main, best verified, and its controls.
selected_metric_sets = [
    ("main", main_record, main_metrics),
    ("best_verified", best_record, best_metrics),
]
for control, (_, metrics, record) in control_results.items():
    selected_metric_sets.append(("best_" + control, record, metrics))
best_curves = []
for condition, record, metrics in selected_metric_sets:
    for tag in sorted(metrics):
        for value in metrics[tag]:
            best_curves.append(
                {
                    "condition": condition,
                    "run": record["run"],
                    "tag": tag,
                    **value,
                }
            )
write_csv(OUT / "best_scalar_curves.csv", best_curves)


# Keep full logs only for runs needed to assess the final method and controls.
shutil.copy2(cache_log, OUT / "gap_cache.log")
shutil.copy2(main_record["log_path"], OUT / "main.log")
shutil.copy2(best_record["log_path"], OUT / "best_verified.log")
for control, (_, _, record) in control_results.items():
    shutil.copy2(record["log_path"], OUT / f"best_{control}.log")
failed = [record for record in all_runs if record["return_code"] != 0]
for index, record in enumerate(failed):
    shutil.copy2(
        record["log_path"],
        OUT / f"failed_{index:02d}_{record['condition']}.log",
    )


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_rows = [main_row, best_row] + [
    values[0]
    for values in control_results.values()
    if "selected_mAP200" in values[0]
]
names = [row["condition"] for row in plot_rows]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(
    range(len(plot_rows)),
    [100 * row["selected_mAP200"] for row in plot_rows],
)
axes[0].set_yticks(range(len(plot_rows)), names)
axes[0].set_xlabel("Selected mAP@200 (%)")
axes[1].barh(
    range(len(plot_rows)),
    [100 * row["selected_P200"] for row in plot_rows],
)
axes[1].set_yticks(range(len(plot_rows)), [])
axes[1].set_xlabel("Selected P@200 (%)")
fig.tight_layout()
fig.savefig(OUT / "best_comparison.png", dpi=170)
plt.close(fig)


source_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
checkpoint_files = []
for record in all_runs:
    checkpoint_files.extend(
        (PROJECT / "saved_models" / record["run"]).glob("*.ckpt")
    )
manifest = {
    "created": datetime.now(UTC).isoformat(),
    "source_commit": source_commit,
    "dataset": "sketchy_2",
    "seed": 42,
    "teacher_pretrain_epochs": 1,
    "teacher_cache": str(TEACHER_CACHE),
    "checkpoint_creation_disabled": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "native_no_checkpoint_support": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "fallback_checkpoint_cleanup": not TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "checkpoint_files_found": [str(path) for path in checkpoint_files],
    "stage1_verified_runs": len(stage1_configurations),
    "stage2_verified_runs": len(stage2_configurations),
    "control_runs": list(control_results),
    "completed_runs": sum(record["return_code"] == 0 for record in all_runs),
    "failed_runs": len(failed),
    "selection_rule": best_payload["selection_rule"],
    "zip_contents": (
        "compact all-run summaries plus detailed main/best/control curves and logs"
    ),
    "weights_included": False,
}
(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
if checkpoint_files:
    raise RuntimeError("Checkpoint creation was disabled but weight files were found.")


# Remove temporary stdout logs and TensorBoard events after compact extraction.
shutil.rmtree(WORK_LOGS)
for record in all_runs:
    tensorboard_run = PROJECT / "tb_logs" / record["run"]
    if tensorboard_run.is_dir() and record["run"].startswith(RUN_PREFIX):
        shutil.rmtree(tensorboard_run)

archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("report") / path.relative_to(OUT))

from IPython.display import FileLink, Image, display

display(Image(filename=str(OUT / "best_comparison.png")))
print("Completed runs:", manifest["completed_runs"])
print("Best configuration:", best_configuration)
print("Best selected mAP@200:", best_row["selected_mAP200"])
print("Weights/checkpoints saved: no")
print("Send this ZIP:", archive)
display(FileLink(str(archive)))
