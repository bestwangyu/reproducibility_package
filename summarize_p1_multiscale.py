#!/usr/bin/env python3
"""Validate and summarize the P1 multiscale zero-shot study."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


SCALES = ("1510", "2005", "2010")
DIMENSIONS = {"1510": (15, 10), "2005": (20, 5), "2010": (20, 10)}
TRAIN_SEEDS = (20260805, 20260807, 20260808)
EVAL_SEEDS = (20260825, 20260826, 20260827)
INSTANCES = 80
REPEATS = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", default="save")
    parser.add_argument(
        "--directory-prefix",
        default="p1_multiscale_v2",
        help="Prefix used by evaluation cell directories",
    )
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


def bool_value(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def validate_cell(save_dir, directory_prefix, scale, train_seed, eval_seed):
    directory = save_dir / "{0}_{1}_train{2}_eval_{3}".format(
        directory_prefix,
        scale, train_seed, eval_seed
    )
    required = (
        "arrival_scenarios.json",
        "breakdown_scenarios.json",
        "sampled_results.csv",
        "selected_results.csv",
        "summary.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} missing {1}".format(directory, ", ".join(missing)))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    sampled = read_csv(directory / "sampled_results.csv")
    selected = read_csv(directory / "selected_results.csv")
    jobs, machines = DIMENSIONS[scale]
    valid = (
        summary.get("passed") is True
        and summary.get("evaluation_mode") == "P1_multiscale_zero_shot_v2"
        and summary.get("data_role") == "development_validation"
        and int(summary.get("target_jobs", -1)) == jobs
        and int(summary.get("target_machines", -1)) == machines
        and int(summary.get("instances", -1)) == INSTANCES
        and int(summary.get("offset", -1)) == 20
        and int(summary.get("sample_repeats", -1)) == REPEATS
        and int(summary.get("sampled_rows", -1)) == INSTANCES * REPEATS
        and summary.get("selection_mode") == "feasible_then_budget_first"
        and float(summary.get("stability_budget", -1)) == 1.7
        and int(summary.get("release_violations", -1)) == 0
        and int(summary.get("selected_instances", -1)) == INSTANCES
        and not summary.get("instances_without_initial_feasible_candidate")
        and not summary.get("instances_without_final_feasible_candidate")
        and int(summary.get("initial_validator_diagnostic_mismatches", -1)) == 0
        and int(summary.get("final_validator_diagnostic_mismatches", -1)) == 0
        and len(sampled) == INSTANCES * REPEATS
        and len(selected) == INSTANCES
        and sum(int(row["release_violations"]) for row in sampled) == 0
    )
    if not valid:
        raise ValueError("Invalid multiscale result: {0}".format(directory))
    return {
        "scale": scale,
        "target_jobs": jobs,
        "target_machines": machines,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "output_dir": str(directory),
        "initial_makespan_mean": mean(selected, "initial_makespan"),
        "final_makespan_mean": mean(selected, "final_makespan"),
        "makespan_delta": mean(selected, "makespan_delta"),
        "initial_stability_cost_mean": mean(selected, "initial_stability_cost"),
        "final_stability_cost_mean": mean(selected, "final_stability_cost"),
        "stability_cost_delta": mean(selected, "stability_cost_delta"),
        "initial_budget_met_rate": statistics.fmean(
            float(bool_value(row["initial_budget_met"])) for row in selected
        ),
        "final_budget_met_rate": statistics.fmean(
            float(bool_value(row["final_budget_met"])) for row in selected
        ),
        "initial_feasible_candidate_rate": float(summary["initial_feasible_candidate_rate"]),
        "final_feasible_candidate_rate": float(summary["final_feasible_candidate_rate"]),
        "arrival_sha256": sha256_file(directory / "arrival_scenarios.json"),
        "breakdown_sha256": sha256_file(directory / "breakdown_scenarios.json"),
    }


def aggregate(rows, scale):
    selected = [row for row in rows if row["scale"] == scale]
    metric_keys = (
        "initial_makespan_mean",
        "final_makespan_mean",
        "makespan_delta",
        "initial_stability_cost_mean",
        "final_stability_cost_mean",
        "stability_cost_delta",
        "initial_budget_met_rate",
        "final_budget_met_rate",
        "initial_feasible_candidate_rate",
        "final_feasible_candidate_rate",
    )
    result = {"scale": scale, "cells": len(selected)}
    result.update(
        {key: statistics.fmean(row[key] for row in selected) for key in metric_keys}
    )
    result["budget_met_rate_delta"] = (
        result["final_budget_met_rate"] - result["initial_budget_met_rate"]
    )
    result["makespan_improved_cells"] = sum(row["makespan_delta"] < 0 for row in selected)
    result["cost_improved_cells"] = sum(row["stability_cost_delta"] < 0 for row in selected)
    return result


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(output_dir))
    rows = [
        validate_cell(save_dir, args.directory_prefix, scale, train_seed, eval_seed)
        for scale in SCALES
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    ]
    for scale in SCALES:
        for eval_seed in EVAL_SEEDS:
            selected = [
                row for row in rows
                if row["scale"] == scale and row["eval_seed"] == eval_seed
            ]
            if len({row["arrival_sha256"] for row in selected}) != 1:
                raise ValueError("Arrival scenario mismatch for {0}/{1}".format(scale, eval_seed))
            if len({row["breakdown_sha256"] for row in selected}) != 1:
                raise ValueError("Breakdown scenario mismatch for {0}/{1}".format(scale, eval_seed))
    aggregates = [aggregate(rows, scale) for scale in SCALES]
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("evaluation_cells.csv", rows), ("aggregate.csv", aggregates)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "analysis": "P1_multiscale_zero_shot_v2",
        "directory_prefix": args.directory_prefix,
        "data_role": "development_validation",
        "source_training_size": "1005",
        "target_scales": list(SCALES),
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "cells": len(rows),
        "instances_per_cell": INSTANCES,
        "sample_repeats": REPEATS,
        "scenarios_match_across_training_seeds": True,
        "release_violations": 0,
        "by_scale": {row["scale"]: row for row in aggregates},
        "passed": len(rows) == 27,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("cells={0}".format(len(rows)))
    for row in aggregates:
        print(
            "{scale}: makespan_delta={makespan_delta:.6f} "
            "cost_delta={stability_cost_delta:.6f} budget_met={final_budget_met_rate:.6f}".format(**row)
        )
    print("release_violations=0")
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("P1 multiscale summary failed")


if __name__ == "__main__":
    main()
