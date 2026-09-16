"""Collect SGCD audit/training diagnostics for analysis; no checkpoints are copied."""

import csv
import json
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
                selected_index, selected_precision_point = max(
                    enumerate(p), key=lambda item: item[1]["value"]
                )
                selected_step = selected_precision_point["step"]
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
                        "selected_step": selected_step,
                        "selected_validation_epoch": selected_index + 1,
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
                if run.name.startswith("sgcd_native_") and fixed_path.is_file():
                    with fixed_path.open(newline="", encoding="utf-8") as stream:
                        fixed = list(csv.DictReader(stream))
                    if fixed:
                        initial = next(
                            (value for value in fixed if value["stage"] == "initial"),
                            fixed[0],
                        )
                        selected_fixed = next(
                            (
                                value
                                for value in fixed
                                if int(value["global_step"]) == selected_step
                            ),
                            fixed[-1],
                        )
                        final_fixed = fixed[-1]
                        summary = {
                            "run": run.name,
                            "version": version.name,
                            "selected_step": selected_step,
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

baseline_rows = [row for row in rows if row["run"].startswith("main_baseline_")]
delta_rows = []
if baseline_rows:
    baseline = max(baseline_rows, key=lambda row: (row["run"], row["version"]))
    (OUT / "baseline_reference.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    for row in rows:
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
if fixed_summaries:
    ordered = sorted(fixed_summaries, key=lambda row: row["run"])
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
        "No model checkpoint, teacher cache or target tensor is copied.",
    ],
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
