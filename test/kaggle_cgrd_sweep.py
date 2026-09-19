"""Tune 18 CGRD configurations on fixed seed-42 main; export a compact ZIP.

Paste this entire file into one offline Kaggle GPU cell after
`kaggle_gap_core_offline.py`. A compatible format-v9 teacher cache is reused.
"""

import csv
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
RUN_PREFIX = "cgrd_sweep_sketchy2_s42_" + STAMP
OUT = Path("/kaggle/working") / ("cgrd_sweep_" + STAMP)
WORK_LOGS = OUT / "work_logs"
OUT.mkdir(parents=True, exist_ok=False)
WORK_LOGS.mkdir(parents=True, exist_ok=False)
TEACHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
TRAIN_SUPPORTS_NO_CHECKPOINTS = "--no_checkpoints" in (
    PROJECT / "src" / "train.py"
).read_text(encoding="utf-8")


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
        "cgrd_coverage",
        "cgrd_query_coverage",
        "cgrd_monotonicity",
        "cgrd_teacher_full_correction",
        "cgrd_teacher_swap_correction",
        "cgrd_student_full_correction",
        "cgrd_student_swap_correction",
        "cgrd_agreement",
        "cgrd_absolute_error",
        "cgrd_full_margin",
        "cgrd_common_margin",
        "cgrd_swapped_margin",
        "cgrd_grad_ratio",
        "cgrd_grad_cosine",
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
    "3",
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
            "--lambda_cgrd",
            "0.0",
            "--teacher_cache_only",
            "--exp_name",
            RUN_PREFIX + "_cache",
        ],
    )
if cache_code != 0 or not TEACHER_CACHE.is_file():
    shutil.copy2(cache_log, OUT / "gap_cache_failed.log")
    raise RuntimeError("CGRD cache preparation failed; inspect the retained log.")


all_rows = []
all_runs = []


def launch(condition, configuration):
    index = len(all_runs)
    label = f"{index:02d}_{condition}"
    run_name = RUN_PREFIX + f"_{index:02d}_{condition}"
    arguments = [
        "--lambda_gap_core",
        "0.0",
        "--lambda_cgrd",
        str(configuration["lambda_cgrd"]),
    ]
    if configuration["lambda_cgrd"] > 0:
        arguments.extend(
            [
                "--cgrd_control",
                configuration["control"],
                "--cgrd_direction",
                configuration["direction"],
                "--cgrd_hard_negative_topk",
                str(configuration["hard_negative_topk"]),
                "--cgrd_huber_beta",
                str(configuration["huber_beta"]),
                "--cgrd_min_full_correction",
                str(configuration["min_full_correction"]),
                "--cgrd_min_swap_correction",
                str(configuration["min_swap_correction"]),
                "--cgrd_max_weight",
                str(configuration["max_weight"]),
                "--cgrd_swapped_loss_weight",
                str(configuration["swapped_loss_weight"]),
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
    "lambda_cgrd": 0.0,
    "direction": "none",
    "hard_negative_topk": 8,
    "min_full_correction": 0.0,
    "min_swap_correction": 0.0,
    "huber_beta": 0.02,
    "max_weight": 0.25,
    "swapped_loss_weight": 1.0,
    "control": "none",
    "stage": "baseline",
}
main_row, main_metrics, main_record = launch("main", main_configuration)
if main_record["return_code"] != 0 or "selected_mAP200" not in main_row:
    shutil.copy2(main_record["log_path"], OUT / "main_failed.log")
    raise RuntimeError("Matched main failed; inspect the retained log.")


# Eighteen verified configurations isolate the parameters most likely to alter
# CGRD scale, pair selection, asymmetry, and loss curvature. Only one field (or
# the paired thresholds) changes from the reference, so the result also acts as
# a compact sensitivity analysis rather than an opaque Cartesian search.
reference = {
    "lambda_cgrd": 0.5,
    "direction": "bidirectional",
    "hard_negative_topk": 8,
    "min_full_correction": 0.0,
    "min_swap_correction": 0.0,
    "huber_beta": 0.02,
    "max_weight": 0.25,
    "swapped_loss_weight": 1.0,
    "control": "verified",
    "stage": "cgrd_tuning",
}
tuning_variants = [
    ("reference", {}),
    ("lambda_0p25", {"lambda_cgrd": 0.25}),
    ("lambda_0p75", {"lambda_cgrd": 0.75}),
    ("lambda_1p0", {"lambda_cgrd": 1.0}),
    ("topk_4", {"hard_negative_topk": 4}),
    ("topk_16", {"hard_negative_topk": 16}),
    ("direction_s2p", {"direction": "sketch_to_photo"}),
    ("direction_p2s", {"direction": "photo_to_sketch"}),
    ("min_full_0p01", {"min_full_correction": 0.01}),
    ("min_swap_0p01", {"min_swap_correction": 0.01}),
    (
        "min_both_0p01",
        {"min_full_correction": 0.01, "min_swap_correction": 0.01},
    ),
    (
        "min_both_0p02",
        {"min_full_correction": 0.02, "min_swap_correction": 0.02},
    ),
    ("swap_weight_0p5", {"swapped_loss_weight": 0.5}),
    ("swap_weight_1p5", {"swapped_loss_weight": 1.5}),
    ("huber_0p01", {"huber_beta": 0.01}),
    ("huber_0p05", {"huber_beta": 0.05}),
    ("max_weight_0p15", {"max_weight": 0.15}),
    ("max_weight_0p35", {"max_weight": 0.35}),
]
if len(tuning_variants) != 18:
    raise RuntimeError("The CGRD tuning plan must contain exactly 18 runs.")

candidate_records = []
for condition, changes in tuning_variants:
    configuration = {**reference, **changes, "variant": condition}
    row, metrics, record = launch("verified_" + condition, configuration)
    if record["return_code"] == 0 and "selected_mAP200" in row:
        candidate_records.append((row, metrics, record))

if not candidate_records:
    raise RuntimeError("Every verified CGRD tuning run failed.")
reference_result = next(
    (
        item
        for item in candidate_records
        if item[2]["configuration"]["variant"] == "reference"
    ),
    None,
)
if reference_result is None:
    raise RuntimeError("The CGRD reference run failed; sensitivity is undefined.")


def rank_key(item):
    row = item[0]
    return (row["selected_mAP200"], row["selected_P200"])


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
reference_row = reference_result[0]
tuning_effects = []
for row in completed_candidates:
    changed = [
        name
        for name in (
            "lambda_cgrd",
            "direction",
            "hard_negative_topk",
            "min_full_correction",
            "min_swap_correction",
            "huber_beta",
            "max_weight",
            "swapped_loss_weight",
        )
        if row[name] != reference[name]
    ]
    tuning_effects.append(
        {
            "variant": row["variant"],
            "changed_fields": ",".join(changed) if changed else "reference",
            "selected_mAP200": row["selected_mAP200"],
            "selected_P200": row["selected_P200"],
            "delta_selected_mAP200_vs_reference": (
                row["selected_mAP200"] - reference_row["selected_mAP200"]
            ),
            "delta_selected_P200_vs_reference": (
                row["selected_P200"] - reference_row["selected_P200"]
            ),
            "delta_selected_mAP200_vs_main": row[
                "delta_selected_mAP200_vs_main"
            ],
            "cgrd_coverage_mean": row.get("cgrd_coverage_mean"),
            "cgrd_query_coverage_mean": row.get("cgrd_query_coverage_mean"),
            "cgrd_monotonicity_mean": row.get("cgrd_monotonicity_mean"),
            "cgrd_agreement_mean": row.get("cgrd_agreement_mean"),
            "cgrd_absolute_error_mean": row.get("cgrd_absolute_error_mean"),
            "cgrd_grad_ratio_mean": row.get("cgrd_grad_ratio_mean"),
            "cgrd_grad_cosine_mean": row.get("cgrd_grad_cosine_mean"),
        }
    )
write_csv(OUT / "sweep_summary.csv", all_rows)
write_csv(OUT / "verified_ranking.csv", completed_candidates)
write_csv(OUT / "tuning_effects.csv", tuning_effects)

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
    "selected_deltas_pp": {
        "verified_minus_main_mAP200": 100
        * (best_row["selected_mAP200"] - main_row["selected_mAP200"]),
        **{
            "verified_minus_" + name + "_mAP200": 100
            * (best_row["selected_mAP200"] - values[0]["selected_mAP200"])
            for name, values in control_results.items()
            if "selected_mAP200" in values[0]
        },
    },
    "interpretation": "seed42_exploratory_tuning_not_significance_evidence",
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
fig, axes = plt.subplots(1, 3, figsize=(19, 9))
axes[0].barh(
    range(len(plot_rows)),
    [100 * row["selected_mAP200"] for row in plot_rows],
)
axes[0].set_yticks(range(len(plot_rows)), names)
axes[0].set_xlabel("Selected mAP@200 (%)")
effect_rows = sorted(
    tuning_effects,
    key=lambda row: row["delta_selected_mAP200_vs_reference"],
)
axes[1].barh(
    range(len(effect_rows)),
    [100 * row["delta_selected_mAP200_vs_reference"] for row in effect_rows],
    color="#2ca02c",
)
axes[1].set_yticks(
    range(len(effect_rows)),
    [row["variant"] for row in effect_rows],
)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Selected mAP delta vs reference (pp)")
coverage_rows = [
    row for row in effect_rows if row["cgrd_query_coverage_mean"] is not None
]
axes[2].barh(
    range(len(coverage_rows)),
    [100 * row["cgrd_query_coverage_mean"] for row in coverage_rows],
    color="#1f77b4",
)
axes[2].set_yticks(range(len(coverage_rows)), [])
axes[2].set_xlabel("Mean CGRD query coverage (%)")
fig.tight_layout()
fig.savefig(OUT / "cgrd_sweep.png", dpi=170)
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
    "student_epochs_per_run": 3,
    "teacher_cache": str(TEACHER_CACHE),
    "checkpoint_creation_disabled": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "native_no_checkpoint_support": TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "fallback_checkpoint_cleanup": not TRAIN_SUPPORTS_NO_CHECKPOINTS,
    "checkpoint_files_found": [str(path) for path in checkpoint_files],
    "main_hyperparameters_tuned": False,
    "planned_verified_tuning_runs": len(tuning_variants),
    "completed_verified_tuning_runs": len(candidate_records),
    "planned_total_student_runs": 1 + len(tuning_variants) + len(control_results),
    "control_runs": list(control_results),
    "completed_runs": sum(record["return_code"] == 0 for record in all_runs),
    "failed_runs": len(failed),
    "selection_rule": best_payload["selection_rule"],
    "zip_contents": (
        "compact all-run summaries plus detailed main/best/control curves and logs"
    ),
    "weights_included": False,
    "statistical_significance_assessed": False,
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

display(Image(filename=str(OUT / "cgrd_sweep.png")))
print("Completed runs:", manifest["completed_runs"])
print("Verified tuning runs:", len(candidate_records), "/", len(tuning_variants))
print("Best configuration:", best_configuration)
print("Best selected mAP@200:", best_row["selected_mAP200"])
print("Selected deltas (pp):", best_payload["selected_deltas_pp"])
print("Statistical significance assessed: no (seed 42 tuning only)")
print("Weights/checkpoints saved: no")
print("Send this ZIP:", archive)
display(FileLink(str(archive)))
