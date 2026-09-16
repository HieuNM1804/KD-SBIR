"""Collect SGCD audit/training diagnostics for analysis; no checkpoints are copied."""

import csv
import json
import re
import statistics
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path("/kaggle/working/KD-SBIR-AVKD")
OUT = Path("/kaggle/working") / (
    "pcsgcd_diagnostics_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalars(version):
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


versions = []
rows = []
curves = []
fixed_summaries = []
run_warnings = []
for pattern in ("main_baseline_*", "sgcd_*", "pcsgcd_*"):
    for run in sorted((PROJECT / "tb_logs").glob(pattern)):
        for version in sorted(run.glob("version_*")):
            diagnostic = version / "sgcd_diagnostics"
            events = list(version.glob("events.out.tfevents.*"))
            if not events and not diagnostic.is_dir():
                continue
            versions.append(version)
            metrics = scalars(version) if events else {}
            m, p = metrics.get("mAP", []), metrics.get("precision", [])
            native_m, native_p = (
                metrics.get("native_mAP", []),
                metrics.get("native_precision", []),
            )
            if m and p:
                configuration_path = diagnostic / "configuration.json"
                configuration = (
                    json.loads(configuration_path.read_text(encoding="utf-8"))
                    if configuration_path.is_file()
                    else {}
                )
                is_native = run.name.startswith("sgcd_native_")
                warnings = []
                configured_epochs = configuration.get("epochs")
                seed_match = re.search(r"(?:^|_)s(\d+)(?:_|$)", run.name)
                seed = configuration.get("seed")
                if seed is None and seed_match:
                    seed = int(seed_match.group(1))
                target_name = configuration.get("sgcd_target")
                if configured_epochs is not None and len(m) != configured_epochs:
                    warnings.append(
                        f"incomplete: {len(m)}/{configured_epochs} validation epochs"
                    )
                if is_native:
                    expected = {
                        "sgcd_student_mode": "native_prompt",
                        "lambda_sgcd_anchor": 0.0,
                        "lambda_sgcd_rank": 0.0,
                        "sgcd_beta": 0.0,
                    }
                    for name, expected_value in expected.items():
                        if configuration.get(name) != expected_value:
                            warnings.append(
                                f"{name}={configuration.get(name)!r}; "
                                f"expected {expected_value!r}"
                            )
                for warning in warnings:
                    run_warnings.append(
                        {"run": run.name, "version": version.name, "warning": warning}
                    )
                run_complete = configured_epochs is None or len(m) == configured_epochs
                native_base_eligible = is_native and not warnings and run_complete
                native_ablation_eligible = (
                    native_base_eligible and target_name == "verified"
                )
                target_control_eligible = native_base_eligible and (
                    configuration.get("lambda_sgcd_where") == 1.0
                    and configuration.get("lambda_sgcd_what") == 0.0
                    and configuration.get("lambda_sgcd_effect") == 0.25
                    and target_name in {"verified", "random", "shuffled"}
                )
                selected_index, selected_precision_point = max(
                    enumerate(p), key=lambda item: item[1]["value"]
                )
                selected_step = selected_precision_point["step"]
                selected_validation_epoch = selected_index + 1
                map_by_step = {value["step"]: value["value"] for value in m}
                native_map_by_step = {
                    value["step"]: value["value"] for value in native_m
                }
                native_precision_by_step = {
                    value["step"]: value["value"] for value in native_p
                }
                if selected_step not in map_by_step:
                    raise RuntimeError(
                        f"mAP is missing at selected precision step {selected_step}: {version}"
                    )
                rows.append(
                    {
                        "run": run.name,
                        "version": version.name,
                        "seed": seed,
                        "sgcd_target": target_name,
                        "configured_epochs": configured_epochs,
                        "run_complete": run_complete,
                        "native_ablation_eligible": native_ablation_eligible,
                        "native_target_control_eligible": target_control_eligible,
                        "configuration_warning": "; ".join(warnings),
                        "lambda_sgcd_where": configuration.get("lambda_sgcd_where"),
                        "lambda_sgcd_what": configuration.get("lambda_sgcd_what"),
                        "lambda_sgcd_effect": configuration.get("lambda_sgcd_effect"),
                        "lambda_sgcd_rank": configuration.get("lambda_sgcd_rank"),
                        "selected_step": selected_step,
                        "selected_validation_epoch": selected_validation_epoch,
                        "selected_mAP": map_by_step[selected_step],
                        "selected_precision": selected_precision_point["value"],
                        "selected_native_mAP": native_map_by_step.get(selected_step),
                        "selected_native_precision": native_precision_by_step.get(
                            selected_step
                        ),
                        "final_mAP": m[-1]["value"],
                        "best_mAP": max(x["value"] for x in m),
                        "final_precision": p[-1]["value"],
                        "best_precision": max(x["value"] for x in p),
                        "final_native_mAP": native_m[-1]["value"] if native_m else None,
                        "final_native_precision": native_p[-1]["value"]
                        if native_p
                        else None,
                        "completed_validation_epochs": len(m),
                    }
                )
                fixed_path = diagnostic / "fixed_batch.csv"
                if is_native and fixed_path.is_file():
                    with fixed_path.open(newline="", encoding="utf-8") as stream:
                        fixed = list(csv.DictReader(stream))
                    if fixed:
                        initial = next(
                            (value for value in fixed if value["stage"] == "initial"),
                            fixed[0],
                        )
                        selected_stage = f"epoch_{selected_validation_epoch}"
                        selected_fixed = next(
                            (
                                value
                                for value in fixed
                                if value["stage"] == selected_stage
                            ),
                            None,
                        )
                        if selected_fixed is None:
                            raise RuntimeError(
                                f"Missing {selected_stage} fixed diagnostics: {version}"
                            )
                        final_fixed = fixed[-1]
                        summary = {
                            "run": run.name,
                            "version": version.name,
                            "selected_step": selected_step,
                            "native_ablation_eligible": native_ablation_eligible,
                            "native_target_control_eligible": target_control_eligible,
                            "seed": seed,
                            "sgcd_target": target_name,
                            "configuration_warning": "; ".join(warnings),
                            "selected_stage": selected_fixed["stage"],
                            "final_stage": final_fixed["stage"],
                        }
                        for name in (
                            "where",
                            "what_cosine",
                            "effect_cosine",
                            "map_cosine",
                            "map_teacher_cosine_gain_over_ink",
                            "student_entropy",
                            "teacher_entropy",
                        ):
                            start = float(initial[name])
                            selected_value = float(selected_fixed[name])
                            final_value = float(final_fixed[name])
                            summary["initial_" + name] = start
                            summary["selected_" + name] = selected_value
                            summary["selected_delta_" + name] = selected_value - start
                            summary["final_" + name] = final_value
                            summary["final_delta_" + name] = final_value - start
                        fixed_summaries.append(summary)
            curve_tags = {
                "mAP",
                "precision",
                "native_mAP",
                "native_precision",
                "main_loss",
                "train_loss",
            }
            curve_tags.update(tag for tag in metrics if tag.startswith("SGCD"))
            for tag in sorted(curve_tags):
                curves.extend(
                    {"run": run.name, "version": version.name, "tag": tag, **value}
                    for value in metrics.get(tag, [])
                )
write_csv(OUT / "comparison_summary.csv", rows)
write_csv(OUT / "scalar_curves.csv", curves)
write_csv(OUT / "ablation_diagnostics.csv", fixed_summaries)
write_csv(OUT / "run_warnings.csv", run_warnings)

baseline_rows = [row for row in rows if row["run"].startswith("main_baseline_")]
baseline_by_seed = {}
for row in baseline_rows:
    seed = row["seed"]
    if seed is None:
        continue
    previous = baseline_by_seed.get(seed)
    if previous is None or (row["run"], row["version"]) > (
        previous["run"],
        previous["version"],
    ):
        baseline_by_seed[seed] = row

target_control_by_key = {}
for row in rows:
    if not row["native_target_control_eligible"]:
        continue
    key = (row["seed"], row["sgcd_target"])
    previous = target_control_by_key.get(key)
    if previous is None or (row["run"], row["version"]) > (
        previous["run"],
        previous["version"],
    ):
        target_control_by_key[key] = row
target_control_rows = [
    target_control_by_key[key]
    for key in sorted(
        target_control_by_key,
        key=lambda value: (value[0] if value[0] is not None else -1, value[1]),
    )
]
write_csv(OUT / "target_control_summary.csv", target_control_rows)
seed_42_targets = {
    row["sgcd_target"] for row in target_control_rows if row["seed"] == 42
}
missing_seed_42_targets = {"verified", "random", "shuffled"} - seed_42_targets
if missing_seed_42_targets:
    run_warnings.append(
        {
            "run": "REPORT",
            "version": "",
            "warning": "Missing complete seed-42 native target controls: "
            + ", ".join(sorted(missing_seed_42_targets)),
        }
    )

replication_rows = []
replication_deltas = []
replication_seeds = set(baseline_by_seed) | {key[0] for key in target_control_by_key}
for seed in sorted(
    replication_seeds, key=lambda value: value if value is not None else -1
):
    baseline = baseline_by_seed.get(seed)
    verified = target_control_by_key.get((seed, "verified"))
    if baseline is not None:
        replication_rows.append({"condition": "main", **baseline})
    if verified is not None:
        replication_rows.append({"condition": "native_where_effect", **verified})
    if baseline is not None and verified is not None:
        replication_deltas.append(
            {
                "seed": seed,
                "baseline_run": baseline["run"],
                "method_run": verified["run"],
                "delta_selected_mAP": verified["selected_mAP"]
                - baseline["selected_mAP"],
                "delta_selected_precision": verified["selected_precision"]
                - baseline["selected_precision"],
                "delta_final_mAP": verified["final_mAP"] - baseline["final_mAP"],
                "delta_final_precision": verified["final_precision"]
                - baseline["final_precision"],
            }
        )
write_csv(OUT / "seed_replication_summary.csv", replication_rows)
write_csv(OUT / "seed_replication_deltas.csv", replication_deltas)
if replication_deltas:
    aggregate = {"paired_seeds": len(replication_deltas)}
    for name in (
        "delta_selected_mAP",
        "delta_selected_precision",
        "delta_final_mAP",
        "delta_final_precision",
    ):
        values = [row[name] for row in replication_deltas]
        aggregate[name + "_mean"] = statistics.mean(values)
        aggregate[name + "_std"] = statistics.stdev(values) if len(values) > 1 else None
    write_csv(OUT / "seed_replication_aggregate.csv", [aggregate])

delta_rows = []
if baseline_by_seed:
    (OUT / "baseline_reference.json").write_text(
        json.dumps(baseline_by_seed, indent=2), encoding="utf-8"
    )
    for row in rows:
        baseline = baseline_by_seed.get(row["seed"])
        if baseline is None:
            continue
        delta_rows.append(
            {
                **row,
                "baseline_run": baseline["run"],
                "delta_selected_mAP": row["selected_mAP"] - baseline["selected_mAP"],
                "delta_selected_precision": row["selected_precision"]
                - baseline["selected_precision"],
                "delta_final_mAP": row["final_mAP"] - baseline["final_mAP"],
                "delta_final_precision": row["final_precision"]
                - baseline["final_precision"],
                "delta_best_mAP": row["best_mAP"] - baseline["best_mAP"],
                "delta_best_precision": row["best_precision"]
                - baseline["best_precision"],
            }
        )
write_csv(OUT / "comparison_deltas.csv", delta_rows)
if not baseline_by_seed:
    run_warnings.append(
        {
            "run": "REPORT",
            "version": "",
            "warning": "No seed-identifiable main_baseline run was found; baseline deltas are unavailable.",
        }
    )
write_csv(OUT / "run_warnings.csv", run_warnings)

checkpoint_rows = []
for run in sorted((PROJECT / "saved_models").glob("*")):
    if not (
        run.name.startswith("main_baseline_")
        or run.name.startswith("sgcd_")
        or run.name.startswith("pcsgcd_")
    ):
        continue
    for checkpoint in sorted(run.glob("*.ckpt")):
        checkpoint_rows.append(
            {
                "run": run.name,
                "checkpoint": checkpoint.name,
                "path": str(checkpoint),
                "size_MiB": checkpoint.stat().st_size / 1024**2,
            }
        )
write_csv(OUT / "checkpoint_inventory.csv", checkpoint_rows)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if rows:
    ordered = sorted(rows, key=lambda row: row["final_mAP"])
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(ordered) * 0.45)))
    names = [row["run"] for row in ordered]
    y = list(range(len(ordered)))
    axes[0].barh(y, [100 * row["final_mAP"] for row in ordered])
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_xlabel("Final full unseen mAP (%)")
    axes[1].barh(y, [100 * row["final_precision"] for row in ordered])
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Final P@100 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    ordered = sorted(rows, key=lambda row: row["selected_mAP"])
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(ordered) * 0.45)))
    names = [row["run"] for row in ordered]
    y = list(range(len(ordered)))
    axes[0].barh(y, [100 * row["selected_mAP"] for row in ordered])
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_xlabel("mAP at precision-selected checkpoint (%)")
    axes[1].barh(y, [100 * row["selected_precision"] for row in ordered])
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Selected-checkpoint P@100 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_selected.png", dpi=160)
    plt.close(fig)
if delta_rows:
    ordered = sorted(delta_rows, key=lambda row: row["delta_final_mAP"])
    fig, axis = plt.subplots(figsize=(12, max(5, len(ordered) * 0.42)))
    names = [row["run"] for row in ordered]
    values = [100 * row["delta_final_mAP"] for row in ordered]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    axis.barh(range(len(ordered)), values, color=colors)
    axis.set_yticks(range(len(ordered)), names, fontsize=8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Final mAP change from latest matched baseline (percentage points)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_deltas.png", dpi=160)
    plt.close(fig)
    ordered = sorted(delta_rows, key=lambda row: row["delta_selected_mAP"])
    fig, axis = plt.subplots(figsize=(12, max(5, len(ordered) * 0.42)))
    names = [row["run"] for row in ordered]
    values = [100 * row["delta_selected_mAP"] for row in ordered]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    axis.barh(range(len(ordered)), values, color=colors)
    axis.set_yticks(range(len(ordered)), names, fontsize=8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Selected-checkpoint mAP change from matched baseline (percentage points)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "comparison_selected_deltas.png", dpi=160)
    plt.close(fig)
if target_control_rows:
    ordered = sorted(
        target_control_rows,
        key=lambda row: (
            row["seed"] if row["seed"] is not None else -1,
            row["sgcd_target"],
        ),
    )
    names = [f"s{row['seed']} {row['sgcd_target']}" for row in ordered]
    y = list(range(len(ordered)))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(ordered) * 0.55)))
    axes[0].barh(y, [100 * row["selected_mAP"] for row in ordered])
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_xlabel("Target-control selected mAP (%)")
    axes[1].barh(y, [100 * row["selected_precision"] for row in ordered])
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Target-control selected P@100 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "target_controls.png", dpi=160)
    plt.close(fig)
if replication_deltas:
    ordered = sorted(replication_deltas, key=lambda row: row["seed"])
    names = [f"seed {row['seed']}" for row in ordered]
    selected = [100 * row["delta_selected_mAP"] for row in ordered]
    final = [100 * row["delta_final_mAP"] for row in ordered]
    y = list(range(len(ordered)))
    height = 0.36
    fig, axis = plt.subplots(figsize=(12, max(5, len(ordered) * 0.65)))
    axis.barh([value - height / 2 for value in y], selected, height, label="selected")
    axis.barh([value + height / 2 for value in y], final, height, label="final")
    axis.set_yticks(y, names)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Native W+Effect minus matched main mAP (percentage points)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "seed_replication_deltas.png", dpi=160)
    plt.close(fig)
eligible_summaries = [row for row in fixed_summaries if row["native_ablation_eligible"]]
if eligible_summaries:
    ordered = sorted(eligible_summaries, key=lambda row: row["run"])
    names = [row["run"] for row in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(17, max(8, len(ordered) * 0.7)))
    panels = (
        ("selected_delta_map_cosine", "Selected map-cosine change"),
        (
            "selected_map_teacher_cosine_gain_over_ink",
            "Selected map gain over fixed ink prior",
        ),
        ("selected_delta_what_cosine", "Selected What-cosine change"),
        ("selected_delta_effect_cosine", "Selected Effect-cosine change"),
    )
    for axis, (key, title) in zip(axes.flat, panels):
        values = [row[key] for row in ordered]
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
        axis.barh(range(len(ordered)), values, color=colors)
        axis.set_yticks(range(len(ordered)), names, fontsize=7)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_mechanisms.png", dpi=160)
    plt.close(fig)

manifest = {
    "created": datetime.now(UTC).isoformat(),
    "project": str(PROJECT),
    "included_versions": [str(path) for path in versions],
    "runs_with_metrics": len(rows),
    "notes": [
        "Pairwise teacher audit files are included even when the gate stops before training.",
        "Full unseen retrieval metrics and fixed seen-batch diagnostics have different scope.",
        "Selected metrics use mAP and precision from the same precision-selected step.",
        "Baseline deltas and replications are matched by parsed or configured seed.",
        "Native target controls differ only in verified, random or shuffled targets.",
        "No model checkpoint, teacher cache or target tensor is copied.",
    ],
    "warnings": run_warnings,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("report") / path.relative_to(OUT))
    for version in versions:
        diagnostic = version / "sgcd_diagnostics"
        if diagnostic.is_dir():
            for path in sorted(diagnostic.rglob("*")):
                if path.is_file():
                    bundle.write(
                        path,
                        Path("tb_logs")
                        / version.parent.name
                        / version.name
                        / "sgcd_diagnostics"
                        / path.relative_to(diagnostic),
                    )

from IPython.display import FileLink, Image, display

for name in (
    "comparison.png",
    "comparison_deltas.png",
    "comparison_selected.png",
    "comparison_selected_deltas.png",
    "target_controls.png",
    "seed_replication_deltas.png",
    "ablation_mechanisms.png",
):
    if (OUT / name).is_file():
        display(Image(filename=str(OUT / name)))
for version in versions:
    for name in (
        "teacher_audit.png",
        "teacher_audit_examples.png",
        "teacher_stroke_graph_examples.png",
        "training_diagnostics.png",
        "prompt_learning.png",
    ):
        image = version / "sgcd_diagnostics" / name
        if image.is_file():
            display(Image(filename=str(image)))
print("Send this diagnostics ZIP:", archive)
display(FileLink(str(archive)))
