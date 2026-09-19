"""Run matched Sketchy2 main, CoRe-KD, and shuffled control; export one ZIP.

Paste this whole file into one offline Kaggle GPU notebook cell after running
`kaggle_core_offline.py`.
"""

import csv
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
ROOT = "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy"
TEACHER_CACHE = "/kaggle/working/teacher_cache/sketchy2_core_teacher2_v7.pt"
STAMP = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
OUT = Path("/kaggle/working") / ("core_kd_sketchy2_comparison_" + STAMP)
OUT.mkdir(parents=True, exist_ok=False)
Path(TEACHER_CACHE).parent.mkdir(parents=True, exist_ok=True)

# The first report runs the claim-critical shuffled control. Set this True only
# after the primary result is promising because it adds another full run.
RUN_REVERSED_CONTROL = False


def run_stage(label, command):
    log_path = OUT / f"{label}.log"
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
        return_code = process.wait()
    print(f"[{label}] exit code: {return_code}", flush=True)
    if return_code != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"[{label}] last log lines:", flush=True)
        print("\n".join(lines[-120:]), flush=True)
    return return_code


def latest_version(run_name):
    versions = list((PROJECT / "tb_logs" / run_name).glob("version_*"))
    if not versions:
        return None
    return max(versions, key=lambda path: int(path.name.split("_")[-1]))


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
    "2",
    "--teacher_cache_path",
    TEACHER_CACHE,
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
    "--no_progress",
]

conditions = [
    {
        "condition": "main",
        "run": "main_core_matched_sketchy2_s42_" + STAMP,
        "arguments": ["--lambda_core", "0.0"],
    },
    {
        "condition": "core_verified",
        "run": "core_verified_sketchy2_s42_" + STAMP,
        "arguments": [
            "--lambda_core",
            "0.5",
            "--core_control",
            "verified",
            "--core_direction",
            "bidirectional",
            "--core_teacher_temperature",
            "0.05",
            "--core_student_temperature",
            "0.05",
            "--core_hard_negative_topk",
            "8",
            "--core_min_teacher_correction",
            "0.0",
            "--core_max_weight",
            "0.25",
        ],
    },
    {
        "condition": "core_shuffled",
        "run": "core_shuffled_sketchy2_s42_" + STAMP,
        "arguments": [
            "--lambda_core",
            "0.5",
            "--core_control",
            "shuffled",
            "--core_direction",
            "bidirectional",
            "--core_teacher_temperature",
            "0.05",
            "--core_student_temperature",
            "0.05",
            "--core_hard_negative_topk",
            "8",
            "--core_min_teacher_correction",
            "0.0",
            "--core_max_weight",
            "0.25",
        ],
    },
]
if RUN_REVERSED_CONTROL:
    reversed_condition = dict(conditions[1])
    reversed_condition["condition"] = "core_reversed"
    reversed_condition["run"] = "core_reversed_sketchy2_s42_" + STAMP
    reversed_condition["arguments"] = [
        "reversed" if value == "verified" else value
        for value in reversed_condition["arguments"]
    ]
    conditions.append(reversed_condition)

(OUT / "commands.json").write_text(json.dumps(conditions, indent=2), encoding="utf-8")

return_codes = {}
for index, condition in enumerate(conditions):
    if index > 0 and return_codes.get("main") != 0:
        return_codes[condition["condition"]] = None
        (OUT / f"{condition['condition']}.log").write_text(
            "Not started because the matched main run failed.\n", encoding="utf-8"
        )
        continue
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        *shared,
        *condition["arguments"],
        "--exp_name",
        condition["run"],
    ]
    return_codes[condition["condition"]] = run_stage(condition["condition"], command)


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


def summarize(condition):
    version = latest_version(condition["run"])
    if version is None:
        return None, [], None, None
    metrics = read_scalars(version)
    map_values = metrics.get("mAP", [])
    precision_values = metrics.get("precision", [])
    curves = []
    for tag in sorted(metrics):
        for value in metrics[tag]:
            curves.append(
                {
                    "condition": condition["condition"],
                    "run": condition["run"],
                    "version": version.name,
                    "tag": tag,
                    **value,
                }
            )
    if not map_values or not precision_values:
        return None, curves, version, metrics
    map_by_step = {value["step"]: value["value"] for value in map_values}
    selected_index, selected_precision = max(
        enumerate(precision_values), key=lambda item: item[1]["value"]
    )
    selected_step = selected_precision["step"]
    row = {
        "condition": condition["condition"],
        "run": condition["run"],
        "version": version.name,
        "return_code": return_codes[condition["condition"]],
        "selected_validation_epoch": selected_index + 1,
        "selected_step": selected_step,
        "selected_mAP200": map_by_step[selected_step],
        "selected_P200": selected_precision["value"],
        "final_mAP200": map_values[-1]["value"],
        "final_P200": precision_values[-1]["value"],
        "best_mAP200": max(value["value"] for value in map_values),
        "best_P200": max(value["value"] for value in precision_values),
        "completed_validation_epochs": len(map_values),
    }
    return row, curves, version, metrics


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


summary_rows = []
all_curves = []
versions = {}
metric_sets = {}
for condition in conditions:
    row, curves, version, metrics = summarize(condition)
    if row is not None:
        summary_rows.append(row)
    all_curves.extend(curves)
    versions[condition["condition"]] = version
    metric_sets[condition["condition"]] = metrics

write_csv(OUT / "comparison_summary.csv", summary_rows)
write_csv(OUT / "scalar_curves.csv", all_curves)

main_row = next((row for row in summary_rows if row["condition"] == "main"), None)
delta_rows = []
if main_row is not None:
    for row in summary_rows:
        if row["condition"] == "main":
            continue
        delta_rows.append(
            {
                "condition": row["condition"],
                "run": row["run"],
                "delta_selected_mAP200": row["selected_mAP200"]
                - main_row["selected_mAP200"],
                "delta_selected_P200": row["selected_P200"] - main_row["selected_P200"],
                "delta_final_mAP200": row["final_mAP200"] - main_row["final_mAP200"],
                "delta_final_P200": row["final_P200"] - main_row["final_P200"],
            }
        )
write_csv(OUT / "comparison_deltas.csv", delta_rows)

mechanism_rows = []
mechanism_tags = (
    "core_coverage",
    "core_teacher_correction",
    "core_student_correction",
    "core_promote",
    "core_suppress",
    "core_agreement",
)
for condition in conditions:
    metrics = metric_sets.get(condition["condition"]) or {}
    if not metrics.get("core_coverage"):
        continue
    row = {"condition": condition["condition"], "run": condition["run"]}
    for tag in mechanism_tags:
        values = metrics.get(tag, [])
        row[tag + "_final"] = values[-1]["value"] if values else None
        row[tag + "_mean"] = (
            sum(value["value"] for value in values) / len(values) if values else None
        )
    mechanism_rows.append(row)
write_csv(OUT / "core_mechanisms.csv", mechanism_rows)

checkpoint_rows = []
for condition in conditions:
    for checkpoint in sorted(
        (PROJECT / "saved_models" / condition["run"]).glob("*.ckpt")
    ):
        checkpoint_rows.append(
            {
                "condition": condition["condition"],
                "run": condition["run"],
                "checkpoint": checkpoint.name,
                "size_MiB": checkpoint.stat().st_size / 1024**2,
            }
        )
write_csv(OUT / "checkpoint_inventory.csv", checkpoint_rows)

cache_path = Path(TEACHER_CACHE)
cache_inventory = {
    "path": str(cache_path),
    "exists": cache_path.is_file(),
    "size_MiB": cache_path.stat().st_size / 1024**2 if cache_path.is_file() else None,
    "included_in_zip": False,
}
(OUT / "cache_inventory.json").write_text(
    json.dumps(cache_inventory, indent=2), encoding="utf-8"
)

if summary_rows:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["condition"] for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(summary_rows) * 0.8)))
    axes[0].barh(
        range(len(summary_rows)),
        [100 * row["selected_mAP200"] for row in summary_rows],
    )
    axes[0].set_yticks(range(len(summary_rows)), names)
    axes[0].set_xlabel("mAP@200 at P@200-selected checkpoint (%)")
    axes[1].barh(
        range(len(summary_rows)),
        [100 * row["selected_P200"] for row in summary_rows],
    )
    axes[1].set_yticks(range(len(summary_rows)), [])
    axes[1].set_xlabel("Selected P@200 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_selected.png", dpi=160)
    plt.close(fig)

source_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
).strip()
manifest = {
    "created": datetime.now(UTC).isoformat(),
    "source_commit": source_commit,
    "dataset": "sketchy_2",
    "metric": "mAP@200/P@200",
    "seed": 42,
    "return_codes": return_codes,
    "runs": {item["condition"]: item["run"] for item in conditions},
    "cache": cache_inventory,
    "claim_checks": {
        "matched_main_present": main_row is not None,
        "verified_present": any(
            row["condition"] == "core_verified" for row in summary_rows
        ),
        "shuffled_control_present": any(
            row["condition"] == "core_shuffled" for row in summary_rows
        ),
    },
    "notes": [
        "All conditions share seed, main losses, optimizer, teacher cache and epochs.",
        "Only lambda_core and the declared correction control differ.",
        "Selected mAP@200 and P@200 come from the same P@200-selected step.",
        "No checkpoint, teacher cache or feature tensor is copied into the ZIP.",
    ],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("report") / path.relative_to(OUT))
    for condition in conditions:
        version = versions.get(condition["condition"])
        if version is None:
            continue
        for event in version.glob("events.out.tfevents.*"):
            bundle.write(
                event,
                Path("tb_logs") / condition["run"] / version.name / event.name,
            )

from IPython.display import FileLink, Image, display

plot = OUT / "comparison_selected.png"
if plot.is_file():
    display(Image(filename=str(plot)))
print("Return codes:", return_codes)
print("Send this ZIP:", archive)
display(FileLink(str(archive)))

failed = [name for name, code in return_codes.items() if code not in (0, None)]
if failed:
    print(
        "Failed conditions: "
        + ", ".join(failed)
        + ". The ZIP above was retained for diagnosis."
    )
