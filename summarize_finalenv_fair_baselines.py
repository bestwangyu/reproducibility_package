#!/usr/bin/env python3
"""Validate and summarize final-environment main and fair baseline results."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


TRAIN_SEEDS = (20260805, 20260807, 20260808)
EVAL_SEEDS = (20260825, 20260826, 20260827)
RULES = ("FIFO", "SPT", "MWKR", "MOR")


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


def finite_mean(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Empty or non-finite metric: {0}".format(key))
    return statistics.fmean(values)


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} missing {1}".format(directory, ", ".join(missing)))


def scenario_hashes(directory):
    return (
        sha256_file(directory / "arrival_scenarios.json"),
        sha256_file(directory / "breakdown_scenarios.json"),
    )


def validate_main(save_dir, train_seed, eval_seed):
    directory = save_dir / "finalenv_nominal_holdout_train{0}_eval_{1}".format(
        train_seed, eval_seed
    )
    require_files(
        directory,
        (
            "arrival_scenarios.json",
            "breakdown_scenarios.json",
            "selected_results.csv",
            "summary.json",
        ),
    )
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(directory / "selected_results.csv")
    if (
        not summary.get("passed")
        or int(summary.get("instances", -1)) != 80
        or int(summary.get("sample_repeats", -1)) != 20
        or int(summary.get("sampled_rows", -1)) != 1600
        or int(summary.get("sampled_release_violations", -1)) != 0
        or summary.get("stability_context_mode") != "full"
        or summary.get("selection_mode") != "budget_first"
        or len(rows) != 80
    ):
        raise ValueError("Invalid main-method result: {0}".format(directory))
    arrival_hash, breakdown_hash = scenario_hashes(directory)
    return {
        "method": "Main_Best-of-20",
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "inference_mode": "stochastic_budget_first_best_of_20",
        "output_dir": str(directory),
        "makespan_mean": finite_mean(rows, "final_makespan"),
        "stability_cost_mean": finite_mean(rows, "final_stability_cost"),
        "budget_met_rate": sum(bool_value(row["final_budget_met"]) for row in rows)
        / 80.0,
        "release_violations": 0,
        "arrival_sha256": arrival_hash,
        "breakdown_sha256": breakdown_hash,
    }


def validate_static(save_dir, train_seed, eval_seed):
    directory = save_dir / "finalenv_static_drl_zero_shot_train{0}_eval_{1}".format(
        train_seed, eval_seed
    )
    require_files(
        directory,
        (
            "arrival_scenarios.json",
            "breakdown_scenarios.json",
            "results.csv",
            "summary.json",
        ),
    )
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(directory / "results.csv")
    if (
        not summary.get("passed")
        or summary.get("evaluation_mode") != "static_drl_dynamic_zero_shot"
        or summary.get("inference_mode") != "greedy"
        or int(summary.get("policy_seed", -1)) != train_seed
        or int(summary.get("eval_seed", -1)) != eval_seed
        or int(summary.get("instances", -1)) != 80
        or int(summary.get("release_violations", -1)) != 0
        or len(rows) != 80
        or sum(int(row["release_violations"]) for row in rows) != 0
    ):
        raise ValueError("Invalid static-DRL result: {0}".format(directory))
    arrival_hash, breakdown_hash = scenario_hashes(directory)
    return {
        "method": "Static_DRL_greedy",
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "inference_mode": "greedy_zero_shot",
        "output_dir": str(directory),
        "makespan_mean": finite_mean(rows, "dynamic_makespan"),
        "stability_cost_mean": finite_mean(rows, "stability_cost"),
        "budget_met_rate": sum(bool_value(row["budget_met"]) for row in rows) / 80.0,
        "release_violations": 0,
        "arrival_sha256": arrival_hash,
        "breakdown_sha256": breakdown_hash,
    }


def validate_rules(save_dir, eval_seed):
    directory = save_dir / "finalenv_dispatch_rules_eval_{0}".format(eval_seed)
    require_files(
        directory,
        (
            "arrival_scenarios.json",
            "breakdown_scenarios.json",
            "results.csv",
            "rule_definitions.json",
            "summary.json",
        ),
    )
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(directory / "results.csv")
    if (
        not summary.get("passed")
        or summary.get("evaluation_mode") != "dynamic_dispatching_rule_baselines"
        or summary.get("inference_mode") != "deterministic"
        or int(summary.get("eval_seed", -1)) != eval_seed
        or int(summary.get("instances_per_rule", -1)) != 80
        or int(summary.get("result_rows", -1)) != 320
        or int(summary.get("release_violations", -1)) != 0
        or len(rows) != 320
        or sum(int(row["release_violations"]) for row in rows) != 0
    ):
        raise ValueError("Invalid dispatch-rule result: {0}".format(directory))
    arrival_hash, breakdown_hash = scenario_hashes(directory)
    result = []
    for rule in RULES:
        selected = [row for row in rows if row["rule"] == rule]
        if len(selected) != 80:
            raise ValueError("Invalid {0} row count in {1}".format(rule, directory))
        result.append(
            {
                "method": rule,
                "train_seed": "none",
                "eval_seed": eval_seed,
                "inference_mode": "deterministic_rule",
                "output_dir": str(directory),
                "makespan_mean": finite_mean(selected, "dynamic_makespan"),
                "stability_cost_mean": finite_mean(selected, "stability_cost"),
                "budget_met_rate": sum(
                    bool_value(row["budget_met"]) for row in selected
                )
                / 80.0,
                "release_violations": 0,
                "arrival_sha256": arrival_hash,
                "breakdown_sha256": breakdown_hash,
            }
        )
    return result


def aggregate_method(rows, method):
    selected = [row for row in rows if row["method"] == method]
    return {
        "method": method,
        "cells": len(selected),
        "inference_mode": selected[0]["inference_mode"],
        "makespan_mean": statistics.fmean(row["makespan_mean"] for row in selected),
        "stability_cost_mean": statistics.fmean(
            row["stability_cost_mean"] for row in selected
        ),
        "budget_met_rate": statistics.fmean(
            row["budget_met_rate"] for row in selected
        ),
        "release_violations": sum(row["release_violations"] for row in selected),
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(output_dir))

    rows = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            rows.append(validate_main(save_dir, train_seed, eval_seed))
            rows.append(validate_static(save_dir, train_seed, eval_seed))
    for eval_seed in EVAL_SEEDS:
        rows.extend(validate_rules(save_dir, eval_seed))

    scenario_groups = 0
    for eval_seed in EVAL_SEEDS:
        selected = [row for row in rows if row["eval_seed"] == eval_seed]
        if len({row["arrival_sha256"] for row in selected}) != 1:
            raise ValueError("Arrival scenarios differ for eval seed {0}".format(eval_seed))
        if len({row["breakdown_sha256"] for row in selected}) != 1:
            raise ValueError("Breakdown scenarios differ for eval seed {0}".format(eval_seed))
        scenario_groups += 1

    methods = ("Main_Best-of-20", "Static_DRL_greedy") + RULES
    aggregates = [aggregate_method(rows, method) for method in methods]
    main = aggregates[0]
    comparisons = []
    for baseline in aggregates[1:]:
        comparisons.append(
            {
                "baseline": baseline["method"],
                "main_minus_baseline_makespan": (
                    main["makespan_mean"] - baseline["makespan_mean"]
                ),
                "main_minus_baseline_stability_cost": (
                    main["stability_cost_mean"] - baseline["stability_cost_mean"]
                ),
                "main_minus_baseline_budget_met_rate": (
                    main["budget_met_rate"] - baseline["budget_met_rate"]
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("evaluation_cells.csv", rows),
        ("aggregate.csv", aggregates),
        ("main_vs_baselines.csv", comparisons),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    summary = {
        "evaluation_mode": "final_environment_fair_baselines",
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "main_cells": 9,
        "static_drl_cells": 9,
        "rule_cells": 12,
        "evaluation_cells": len(rows),
        "scenario_hash_groups_checked": scenario_groups,
        "scenario_hashes_match_across_all_methods": True,
        "release_violations": sum(row["release_violations"] for row in rows),
        "inference_protocols_are_distinct": True,
        "by_method": {row["method"]: row for row in aggregates},
        "main_vs_baselines": comparisons,
    }
    summary["passed"] = (
        summary["evaluation_cells"] == 30
        and summary["scenario_hash_groups_checked"] == 3
        and summary["scenario_hashes_match_across_all_methods"]
        and summary["release_violations"] == 0
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("evaluation_cells={0}".format(summary["evaluation_cells"]))
    print("scenario_hashes_match_across_all_methods={0}".format(
        summary["scenario_hashes_match_across_all_methods"]
    ))
    for row in aggregates:
        print(
            "{0}: makespan={1:.6f} cost={2:.6f} budget_met={3:.6f}".format(
                row["method"],
                row["makespan_mean"],
                row["stability_cost_mean"],
                row["budget_met_rate"],
            )
        )
    print("release_violations={0}".format(summary["release_violations"]))
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("Final-environment baseline summary failed.")


if __name__ == "__main__":
    main()
