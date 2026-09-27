#!/usr/bin/env python3
"""Validate and summarize final-environment multi-seed robustness results."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


TRAIN_SEEDS = (20260805, 20260807, 20260808)
EVAL_SEEDS = (20260825, 20260826, 20260827)
CONDITIONS = {
    "mild": {
        "prefix": "finalenv_robustness_mild",
        "iat": 12.0,
        "mtbf": 12.0,
        "repair": 1.0,
        "events": 1,
    },
    "nominal": {
        "prefix": "finalenv_nominal_holdout",
        "iat": 8.0,
        "mtbf": 8.0,
        "repair": 2.0,
        "events": 1,
    },
    "severe": {
        "prefix": "finalenv_robustness_severe",
        "iat": 4.0,
        "mtbf": 4.0,
        "repair": 3.0,
        "events": 1,
    },
    "multi_failure_mtbf5": {
        "prefix": "finalenv_robustness_multifailure_mtbf5",
        "iat": 8.0,
        "mtbf": 5.0,
        "repair": 2.0,
        "events": 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", default="save")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def finite_mean(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Empty or non-finite metric: {0}".format(key))
    return statistics.fmean(values)


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def result_directory(save_dir, condition, train_seed, eval_seed):
    prefix = CONDITIONS[condition]["prefix"]
    return save_dir / "{0}_train{1}_eval_{2}".format(
        prefix, train_seed, eval_seed
    )


def validate_cell(directory, condition, train_seed, eval_seed):
    required = (
        "arrival_scenarios.json",
        "breakdown_scenarios.json",
        "results.csv",
        "sampled_results.csv",
        "selected_results.csv",
        "summary.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} missing {1}".format(directory, ", ".join(missing)))

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        not summary.get("passed")
        or int(summary.get("instances", -1)) != 80
        or int(summary.get("offset", -1)) != 20
        or int(summary.get("sample_repeats", -1)) != 20
        or int(summary.get("sampled_rows", -1)) != 1600
        or int(summary.get("release_violations", -1)) != 0
        or int(summary.get("sampled_release_violations", -1)) != 0
        or summary.get("stability_context_mode") != "full"
        or summary.get("selection_mode") != "budget_first"
        or not close(summary.get("budget", -1), 1.7)
    ):
        raise ValueError("Invalid evaluation summary: {0}".format(directory))

    results = read_csv(directory / "results.csv")
    sampled = read_csv(directory / "sampled_results.csv")
    selected = read_csv(directory / "selected_results.csv")
    if (len(results), len(sampled), len(selected)) != (80, 1600, 80):
        raise ValueError("Unexpected CSV row counts in {0}".format(directory))
    if sum(int(row["release_violations"]) for row in results + sampled) != 0:
        raise ValueError("Release violation in {0}".format(directory))

    expected = CONDITIONS[condition]
    arrivals = json.loads(
        (directory / "arrival_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    breakdowns = json.loads(
        (directory / "breakdown_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    if len(arrivals) != 80 or len(breakdowns) != 80:
        raise ValueError("Unexpected scenario count in {0}".format(directory))
    if any(
        not close(item["mean_interarrival"], expected["iat"])
        or int(item["initial_jobs"]) != 6
        for item in arrivals
    ):
        raise ValueError("Arrival protocol mismatch in {0}".format(directory))
    if any(
        not close(item["mean_time_between_failures"], expected["mtbf"])
        or not close(item["mean_repair_time"], expected["repair"])
        or len(item["events"]) != 5 * expected["events"]
        for item in breakdowns
    ):
        raise ValueError("Breakdown protocol mismatch in {0}".format(directory))

    initial_makespan = finite_mean(selected, "initial_makespan")
    final_makespan = finite_mean(selected, "final_makespan")
    initial_cost = finite_mean(selected, "initial_stability_cost")
    final_cost = finite_mean(selected, "final_stability_cost")
    initial_budget = sum(bool_value(row["initial_budget_met"]) for row in selected)
    final_budget = sum(bool_value(row["final_budget_met"]) for row in selected)
    recomputed = {
        "selected_initial_makespan_mean": initial_makespan,
        "selected_final_makespan_mean": final_makespan,
        "selected_makespan_delta": final_makespan - initial_makespan,
        "selected_initial_stability_cost_mean": initial_cost,
        "selected_final_stability_cost_mean": final_cost,
        "selected_stability_cost_delta": final_cost - initial_cost,
        "selected_initial_budget_met": initial_budget,
        "selected_final_budget_met": final_budget,
    }
    for key, value in recomputed.items():
        if not close(summary[key], value):
            raise ValueError("Summary/CSV mismatch for {0}: {1}".format(key, directory))

    return {
        "condition": condition,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "output_dir": str(directory),
        "selected_initial_makespan_mean": initial_makespan,
        "selected_final_makespan_mean": final_makespan,
        "selected_makespan_delta": final_makespan - initial_makespan,
        "selected_initial_stability_cost_mean": initial_cost,
        "selected_final_stability_cost_mean": final_cost,
        "selected_stability_cost_delta": final_cost - initial_cost,
        "selected_initial_budget_met_rate": initial_budget / 80.0,
        "selected_final_budget_met_rate": final_budget / 80.0,
        "selected_budget_met_rate_delta": (final_budget - initial_budget) / 80.0,
        "arrival_sha256": sha256_file(directory / "arrival_scenarios.json"),
        "breakdown_sha256": sha256_file(directory / "breakdown_scenarios.json"),
        "passed": True,
    }


def aggregate(selected, aggregation, condition, train_seed):
    keys = (
        "selected_initial_makespan_mean",
        "selected_final_makespan_mean",
        "selected_makespan_delta",
        "selected_initial_stability_cost_mean",
        "selected_final_stability_cost_mean",
        "selected_stability_cost_delta",
        "selected_initial_budget_met_rate",
        "selected_final_budget_met_rate",
        "selected_budget_met_rate_delta",
    )
    row = {
        "aggregation": aggregation,
        "condition": condition,
        "train_seed": train_seed,
        "cells": len(selected),
    }
    row.update({key: statistics.fmean(item[key] for item in selected) for key in keys})
    return row


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(output_dir))

    rows = []
    for condition in CONDITIONS:
        for train_seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                rows.append(
                    validate_cell(
                        result_directory(save_dir, condition, train_seed, eval_seed),
                        condition,
                        train_seed,
                        eval_seed,
                    )
                )

    for condition in CONDITIONS:
        for eval_seed in EVAL_SEEDS:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["eval_seed"] == eval_seed
            ]
            if len({row["arrival_sha256"] for row in selected}) != 1:
                raise ValueError("Arrival scenarios differ: {0}/{1}".format(condition, eval_seed))
            if len({row["breakdown_sha256"] for row in selected}) != 1:
                raise ValueError("Breakdown scenarios differ: {0}/{1}".format(condition, eval_seed))

    aggregates = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for train_seed in TRAIN_SEEDS:
            aggregates.append(
                aggregate(
                    [row for row in condition_rows if row["train_seed"] == train_seed],
                    "condition_train_seed",
                    condition,
                    train_seed,
                )
            )
        aggregates.append(
            aggregate(condition_rows, "condition_all_train_seeds", condition, "all")
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "evaluation_cells.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    by_condition = {
        row["condition"]: row
        for row in aggregates
        if row["aggregation"] == "condition_all_train_seeds"
    }
    summary = {
        "evaluation_mode": "final_environment_multiseed_robustness",
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "conditions": list(CONDITIONS),
        "evaluation_cells": len(rows),
        "new_evaluation_cells": sum(row["condition"] != "nominal" for row in rows),
        "reused_nominal_cells": sum(row["condition"] == "nominal" for row in rows),
        "instances_per_cell": 80,
        "sample_repeats": 20,
        "scenario_hash_groups_checked": len(CONDITIONS) * len(EVAL_SEEDS),
        "scenario_hashes_match_across_train_seeds": True,
        "release_violations": 0,
        "all_cells_passed": all(row["passed"] for row in rows),
        "by_condition": by_condition,
    }
    summary["passed"] = (
        summary["evaluation_cells"] == 36
        and summary["new_evaluation_cells"] == 27
        and summary["reused_nominal_cells"] == 9
        and summary["scenario_hash_groups_checked"] == 12
        and summary["scenario_hashes_match_across_train_seeds"]
        and summary["release_violations"] == 0
        and summary["all_cells_passed"]
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("evaluation_cells={0}".format(summary["evaluation_cells"]))
    print("new_evaluation_cells={0}".format(summary["new_evaluation_cells"]))
    print("reused_nominal_cells={0}".format(summary["reused_nominal_cells"]))
    print("scenario_hashes_match_across_train_seeds={0}".format(
        summary["scenario_hashes_match_across_train_seeds"]
    ))
    for condition, row in by_condition.items():
        print(
            "{0}: makespan={1:.6f} delta={2:.6f} cost={3:.6f} "
            "cost_delta={4:.6f} budget_met={5:.6f}".format(
                condition,
                row["selected_final_makespan_mean"],
                row["selected_makespan_delta"],
                row["selected_final_stability_cost_mean"],
                row["selected_stability_cost_delta"],
                row["selected_final_budget_met_rate"],
            )
        )
    print("release_violations={0}".format(summary["release_violations"]))
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("Final-environment robustness summary failed.")


if __name__ == "__main__":
    main()
