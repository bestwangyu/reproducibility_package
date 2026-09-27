#!/usr/bin/env python3
"""Validate and summarize the final-environment 3x3 nominal holdout study."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


TRAIN_SEEDS = (20260805, 20260807, 20260808)
EVAL_SEEDS = (20260825, 20260826, 20260827)
EXPECTED_INSTANCES = 80
EXPECTED_REPEATS = 20
EXPECTED_BUDGET = 1.7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", default="save")
    parser.add_argument("--directory-prefix", default="finalenv_nominal_holdout")
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


def mean(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite or empty metric: {0}".format(key))
    return statistics.fmean(values)


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def close(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def validate_cell(directory, train_seed, eval_seed):
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
    expected_values = {
        "instances": EXPECTED_INSTANCES,
        "offset": 20,
        "sample_repeats": EXPECTED_REPEATS,
        "sampled_rows": EXPECTED_INSTANCES * EXPECTED_REPEATS,
        "release_violations": 0,
        "sampled_release_violations": 0,
    }
    for key, expected in expected_values.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError("Unexpected {0} in {1}".format(key, directory))
    if not close(summary.get("budget", -1), EXPECTED_BUDGET):
        raise ValueError("Unexpected budget in {0}".format(directory))
    if summary.get("stability_context_mode") != "full":
        raise ValueError("Unexpected context mode in {0}".format(directory))
    if summary.get("selection_mode") != "budget_first":
        raise ValueError("Unexpected selection mode in {0}".format(directory))
    if not summary.get("passed") or not summary.get("sampled_feasible"):
        raise ValueError("Evaluation did not pass in {0}".format(directory))

    results = read_csv(directory / "results.csv")
    sampled = read_csv(directory / "sampled_results.csv")
    selected = read_csv(directory / "selected_results.csv")
    if (len(results), len(sampled), len(selected)) != (80, 1600, 80):
        raise ValueError("Unexpected CSV row counts in {0}".format(directory))
    if sum(int(row["release_violations"]) for row in results + sampled) != 0:
        raise ValueError("Release violation in {0}".format(directory))

    arrivals = json.loads(
        (directory / "arrival_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    breakdowns = json.loads(
        (directory / "breakdown_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    if len(arrivals) != 80 or len(breakdowns) != 80:
        raise ValueError("Unexpected scenario count in {0}".format(directory))
    if any(
        not close(item["mean_interarrival"], 8.0) or int(item["initial_jobs"]) != 6
        for item in arrivals
    ):
        raise ValueError("Unexpected arrival protocol in {0}".format(directory))
    if any(
        not close(item["mean_time_between_failures"], 8.0)
        or not close(item["mean_repair_time"], 2.0)
        or len(item["events"]) != 5
        for item in breakdowns
    ):
        raise ValueError("Unexpected breakdown protocol in {0}".format(directory))

    selected_initial_makespan = mean(selected, "initial_makespan")
    selected_final_makespan = mean(selected, "final_makespan")
    selected_initial_cost = mean(selected, "initial_stability_cost")
    selected_final_cost = mean(selected, "final_stability_cost")
    selected_initial_budget_met = sum(
        bool_value(row["initial_budget_met"]) for row in selected
    )
    selected_final_budget_met = sum(
        bool_value(row["final_budget_met"]) for row in selected
    )
    recomputed = {
        "selected_initial_makespan_mean": selected_initial_makespan,
        "selected_final_makespan_mean": selected_final_makespan,
        "selected_makespan_delta": selected_final_makespan - selected_initial_makespan,
        "selected_initial_stability_cost_mean": selected_initial_cost,
        "selected_final_stability_cost_mean": selected_final_cost,
        "selected_stability_cost_delta": selected_final_cost - selected_initial_cost,
        "selected_initial_budget_met": selected_initial_budget_met,
        "selected_final_budget_met": selected_final_budget_met,
    }
    for key, value in recomputed.items():
        if not close(summary[key], value):
            raise ValueError("Summary/CSV mismatch for {0} in {1}".format(key, directory))

    return {
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "output_dir": str(directory),
        "selected_initial_makespan_mean": selected_initial_makespan,
        "selected_final_makespan_mean": selected_final_makespan,
        "selected_makespan_delta": selected_final_makespan - selected_initial_makespan,
        "selected_initial_stability_cost_mean": selected_initial_cost,
        "selected_final_stability_cost_mean": selected_final_cost,
        "selected_stability_cost_delta": selected_final_cost - selected_initial_cost,
        "selected_initial_budget_met_rate": selected_initial_budget_met / 80.0,
        "selected_final_budget_met_rate": selected_final_budget_met / 80.0,
        "selected_budget_met_rate_delta": (
            selected_final_budget_met - selected_initial_budget_met
        ) / 80.0,
        "arrival_sha256": sha256_file(directory / "arrival_scenarios.json"),
        "breakdown_sha256": sha256_file(directory / "breakdown_scenarios.json"),
        "passed": True,
    }


def aggregate(selected, aggregation, train_seed):
    metric_keys = (
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
        "train_seed": train_seed,
        "cells": len(selected),
    }
    row.update(
        {
            key: statistics.fmean(item[key] for item in selected)
            for key in metric_keys
        }
    )
    return row


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(output_dir))

    rows = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            directory = save_dir / "{0}_train{1}_eval_{2}".format(
                args.directory_prefix, train_seed, eval_seed
            )
            rows.append(validate_cell(directory, train_seed, eval_seed))

    for eval_seed in EVAL_SEEDS:
        same_scenario = [row for row in rows if row["eval_seed"] == eval_seed]
        if len({row["arrival_sha256"] for row in same_scenario}) != 1:
            raise ValueError("Arrival scenarios differ for eval seed {0}".format(eval_seed))
        if len({row["breakdown_sha256"] for row in same_scenario}) != 1:
            raise ValueError("Breakdown scenarios differ for eval seed {0}".format(eval_seed))

    aggregates = []
    for train_seed in TRAIN_SEEDS:
        aggregates.append(
            aggregate(
                [row for row in rows if row["train_seed"] == train_seed],
                "train_seed",
                train_seed,
            )
        )
    aggregates.append(aggregate(rows, "all_train_seeds", "all"))

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

    overall = aggregates[-1]
    summary = {
        "evaluation_mode": "final_environment_nominal_holdout_3x3",
        "directory_prefix": args.directory_prefix,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "evaluation_cells": len(rows),
        "instances_per_cell": EXPECTED_INSTANCES,
        "sample_repeats": EXPECTED_REPEATS,
        "selected_rows": EXPECTED_INSTANCES * len(rows),
        "sampled_rows": EXPECTED_INSTANCES * EXPECTED_REPEATS * len(rows),
        "scenario_hash_groups_checked": len(EVAL_SEEDS),
        "scenario_hashes_match_across_train_seeds": True,
        "release_violations": 0,
        "all_cells_passed": all(row["passed"] for row in rows),
        "by_train_seed": {
            str(row["train_seed"]): row for row in aggregates[:-1]
        },
        "overall": overall,
    }
    summary["passed"] = (
        summary["evaluation_cells"] == 9
        and summary["selected_rows"] == 720
        and summary["sampled_rows"] == 14400
        and summary["all_cells_passed"]
        and summary["release_violations"] == 0
        and summary["scenario_hashes_match_across_train_seeds"]
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("evaluation_cells={0}".format(summary["evaluation_cells"]))
    print("selected_rows={0}".format(summary["selected_rows"]))
    print("sampled_rows={0}".format(summary["sampled_rows"]))
    print("scenario_hashes_match_across_train_seeds={0}".format(
        summary["scenario_hashes_match_across_train_seeds"]
    ))
    print("selected_makespan_delta={0:.6f}".format(
        overall["selected_makespan_delta"]
    ))
    print("selected_stability_cost_delta={0:.6f}".format(
        overall["selected_stability_cost_delta"]
    ))
    print("selected_final_budget_met_rate={0:.6f}".format(
        overall["selected_final_budget_met_rate"]
    ))
    print("release_violations={0}".format(summary["release_violations"]))
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("Final-environment 3x3 holdout summary failed.")


if __name__ == "__main__":
    main()
