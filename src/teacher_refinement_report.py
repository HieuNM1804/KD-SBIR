"""Write and aggregate lightweight staged-teacher experiment reports."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics


REPORT_FORMAT_VERSION = 2
_SEED_FIELDS = {"seed", "teacher_prompt_seed", "teacher_text_prompt_seed"}


def report_path_for_cache(cache_path):
    cache_path = Path(cache_path)
    return cache_path.with_name(cache_path.name + ".metrics.json")


def write_teacher_refinement_report(cache_path, metadata, result, histories):
    """Atomically write metrics without loading the large feature cache."""
    report_path = report_path_for_cache(cache_path)
    temporary_path = report_path.with_name(report_path.name + ".tmp")
    payload = {
        "report_format_version": REPORT_FORMAT_VERSION,
        "cache_file": str(Path(cache_path)),
        "metadata": metadata,
        "result": result,
        "histories": histories,
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, report_path)
    return report_path


def _comparison_signature(metadata):
    signature = deepcopy(metadata)
    for field in _SEED_FIELDS:
        signature.pop(field, None)
    return signature


def summarize_teacher_refinement_reports(report_paths, minimum_runs=3):
    """Summarize text value against a matched visual-only continuation."""
    if minimum_runs < 1:
        raise ValueError("minimum_runs must be positive.")
    paths = [Path(path) for path in report_paths]
    if len(paths) < minimum_runs:
        raise ValueError(
            f"Need at least {minimum_runs} reports; received {len(paths)}."
        )

    reports = []
    signature = None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("report_format_version") != REPORT_FORMAT_VERSION:
            raise ValueError(f"Unsupported report format: {path}")
        metadata = report.get("metadata", {})
        if metadata.get("teacher_objective") != (
            "part_query_matched_control_teacher_refinement"
        ):
            raise ValueError(f"Not a staged teacher report: {path}")
        current_signature = _comparison_signature(metadata)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError(
                "Reports use incompatible non-seed teacher configurations: "
                f"{path}"
            )
        result = report.get("result", {})
        required = {
            "best_phase",
            "phase_a_acc1",
            "phase_a_acc5",
            "control_acc1",
            "control_acc5",
            "semantic_acc1",
            "semantic_acc5",
            "best_acc1",
            "best_acc5",
            "semantic_delta_vs_phase_a_acc1",
            "semantic_delta_vs_phase_a_acc5",
            "text_added_value_acc1",
            "text_added_value_acc5",
            "best_delta_vs_phase_a_acc1",
            "best_delta_vs_phase_a_acc5",
        }
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f"Report {path} is missing: {missing}")
        reports.append((path, metadata, result))

    text_value_acc1 = [
        float(result["text_added_value_acc1"])
        for _, _, result in reports
    ]
    text_value_acc5 = [
        float(result["text_added_value_acc5"])
        for _, _, result in reports
    ]
    semantic_vs_phase_a_acc1 = [
        float(result["semantic_delta_vs_phase_a_acc1"])
        for _, _, result in reports
    ]
    best_vs_phase_a_acc1 = [
        float(result["best_delta_vs_phase_a_acc1"])
        for _, _, result in reports
    ]
    runs = []
    for path, metadata, result in reports:
        runs.append(
            {
                "report": str(path),
                "seed": metadata.get("seed"),
                "best_phase": result["best_phase"],
                "phase_a_acc1": float(result["phase_a_acc1"]),
                "control_acc1": float(result["control_acc1"]),
                "semantic_acc1": float(result["semantic_acc1"]),
                "best_acc1": float(result["best_acc1"]),
                "semantic_delta_vs_phase_a_acc1": float(
                    result["semantic_delta_vs_phase_a_acc1"]
                ),
                "text_added_value_acc1": float(
                    result["text_added_value_acc1"]
                ),
                "text_added_value_acc5": float(
                    result["text_added_value_acc5"]
                ),
                "best_delta_vs_phase_a_acc1": float(
                    result["best_delta_vs_phase_a_acc1"]
                ),
            }
        )
    return {
        "run_count": len(runs),
        "positive_text_value_acc1_runs": sum(
            value > 0 for value in text_value_acc1
        ),
        "all_text_value_acc1_positive": all(
            value > 0 for value in text_value_acc1
        ),
        "mean_text_added_value_acc1": statistics.fmean(text_value_acc1),
        "stdev_text_added_value_acc1": (
            statistics.stdev(text_value_acc1)
            if len(text_value_acc1) > 1
            else 0.0
        ),
        "mean_text_added_value_acc5": statistics.fmean(text_value_acc5),
        "stdev_text_added_value_acc5": (
            statistics.stdev(text_value_acc5)
            if len(text_value_acc5) > 1
            else 0.0
        ),
        "mean_semantic_delta_vs_phase_a_acc1": statistics.fmean(
            semantic_vs_phase_a_acc1
        ),
        "mean_best_delta_vs_phase_a_acc1": statistics.fmean(
            best_vs_phase_a_acc1
        ),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate staged teacher semantic-refinement reports."
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--minimum_runs", type=int, default=3)
    parser.add_argument("--require_all_positive", action="store_true")
    args = parser.parse_args()
    summary = summarize_teacher_refinement_reports(
        args.reports,
        minimum_runs=args.minimum_runs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if (
        args.require_all_positive
        and not summary["all_text_value_acc1_positive"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
