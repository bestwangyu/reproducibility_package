#!/usr/bin/env python3
"""Validate and summarize the frozen P0-T independent-test evaluation."""

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
EXPECTED_INSTANCES = 100
EXPECTED_REPEATS = 20
EXPECTED_BUDGET = 1.7
MAIN_PREFIX = "p0t_independent_main"
STATIC_PREFIX = "p0t_independent_static_drl"
RULE_PREFIX = "p0t_independent_rules"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", default="save")
    parser.add_argument(
        "--protocol",
        default="save/p0t_frozen_independent_test_protocol.json",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def close(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def finite_mean(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Empty or non-finite metric: {0}".format(key))
    return statistics.fmean(values)


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} missing {1}".format(directory, ", ".join(missing)))


def scenario_hashes(directory):
    return (
        sha256_file(directory / "arrival_scenarios.json"),
        sha256_file(directory / "breakdown_scenarios.json"),
    )


def validate_file_names(rows, expected_names, directory):
    actual = [row["file_name"] for row in rows]
    if sorted(actual) != expected_names:
        raise ValueError("Dataset filenames differ from frozen protocol in {0}".format(directory))


def method_row(method, train_seed, eval_seed, inference_mode, directory, rows,
               makespan_key, cost_key, budget_key):
    arrival_hash, breakdown_hash = scenario_hashes(directory)
    return {
        "method": method,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "inference_mode": inference_mode,
        "output_dir": str(directory),
        "makespan_mean": finite_mean(rows, makespan_key),
        "stability_cost_mean": finite_mean(rows, cost_key),
        "budget_met_rate": sum(bool_value(row[budget_key]) for row in rows)
        / EXPECTED_INSTANCES,
        "release_violations": 0,
        "arrival_sha256": arrival_hash,
        "breakdown_sha256": breakdown_hash,
    }


def validate_main(save_dir, train_seed, eval_seed, expected_names):
    directory = save_dir / "{0}_train{1}_eval_{2}".format(
        MAIN_PREFIX, train_seed, eval_seed
    )
    require_files(
        directory,
        (
            "arrival_scenarios.json",
            "breakdown_scenarios.json",
            "results.csv",
            "sampled_results.csv",
            "selected_results.csv",
            "summary.json",
        ),
    )
    summary = read_json(directory / "summary.json")
    results = read_csv(directory / "results.csv")
    sampled = read_csv(directory / "sampled_results.csv")
    selected = read_csv(directory / "selected_results.csv")
    valid = (
        summary.get("passed") is True
        and int(summary.get("instances", -1)) == EXPECTED_INSTANCES
        and int(summary.get("offset", -1)) == 0
        and close(summary.get("budget", -1), EXPECTED_BUDGET)
        and int(summary.get("sample_repeats", -1)) == EXPECTED_REPEATS
        and int(summary.get("sampled_rows", -1))
        == EXPECTED_INSTANCES * EXPECTED_REPEATS
        and int(summary.get("release_violations", -1)) == 0
        and int(summary.get("sampled_release_violations", -1)) == 0
        and summary.get("stability_context_mode") == "full"
        and summary.get("selection_mode") == "budget_first"
        and len(results) == EXPECTED_INSTANCES
        and len(sampled) == EXPECTED_INSTANCES * EXPECTED_REPEATS
        and len(selected) == EXPECTED_INSTANCES
        and sum(int(row["release_violations"]) for row in results + sampled) == 0
    )
    if not valid:
        raise ValueError("Invalid frozen main-method result: {0}".format(directory))
    validate_file_names(selected, expected_names, directory)
    scenarios = read_json(directory / "arrival_scenarios.json")["scenarios"]
    breakdowns = read_json(directory / "breakdown_scenarios.json")["scenarios"]
    if len(scenarios) != EXPECTED_INSTANCES or len(breakdowns) != EXPECTED_INSTANCES:
        raise ValueError("Invalid scenario count in {0}".format(directory))
    return (
        method_row(
            "Main_Best_of_20", train_seed, eval_seed,
            "stochastic_budget_first_best_of_20", directory, selected,
            "final_makespan", "final_stability_cost", "final_budget_met",
        ),
        method_row(
            "Initial_Best_of_20", train_seed, eval_seed,
            "stochastic_budget_first_best_of_20", directory, selected,
            "initial_makespan", "initial_stability_cost", "initial_budget_met",
        ),
    )


def validate_static(save_dir, train_seed, eval_seed, expected_names):
    directory = save_dir / "{0}_train{1}_eval_{2}".format(
        STATIC_PREFIX, train_seed, eval_seed
    )
    require_files(
        directory,
        ("arrival_scenarios.json", "breakdown_scenarios.json", "results.csv", "summary.json"),
    )
    summary = read_json(directory / "summary.json")
    rows = read_csv(directory / "results.csv")
    valid = (
        summary.get("passed") is True
        and summary.get("evaluation_mode") == "static_drl_dynamic_zero_shot"
        and summary.get("inference_mode") == "greedy"
        and int(summary.get("policy_seed", -1)) == train_seed
        and int(summary.get("eval_seed", -1)) == eval_seed
        and int(summary.get("instances", -1)) == EXPECTED_INSTANCES
        and int(summary.get("offset", -1)) == 0
        and int(summary.get("release_violations", -1)) == 0
        and len(rows) == EXPECTED_INSTANCES
        and sum(int(row["release_violations"]) for row in rows) == 0
    )
    if not valid:
        raise ValueError("Invalid frozen static-DRL result: {0}".format(directory))
    validate_file_names(rows, expected_names, directory)
    return method_row(
        "Static_DRL_greedy", train_seed, eval_seed, "greedy_zero_shot",
        directory, rows, "dynamic_makespan", "stability_cost", "budget_met",
    )


def validate_rules(save_dir, eval_seed, expected_names):
    directory = save_dir / "{0}_eval_{1}".format(RULE_PREFIX, eval_seed)
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
    summary = read_json(directory / "summary.json")
    rows = read_csv(directory / "results.csv")
    valid = (
        summary.get("passed") is True
        and summary.get("evaluation_mode") == "dynamic_dispatching_rule_baselines"
        and summary.get("inference_mode") == "deterministic"
        and int(summary.get("eval_seed", -1)) == eval_seed
        and int(summary.get("instances_per_rule", -1)) == EXPECTED_INSTANCES
        and int(summary.get("offset", -1)) == 0
        and int(summary.get("result_rows", -1)) == EXPECTED_INSTANCES * len(RULES)
        and int(summary.get("release_violations", -1)) == 0
        and len(rows) == EXPECTED_INSTANCES * len(RULES)
        and sum(int(row["release_violations"]) for row in rows) == 0
    )
    if not valid:
        raise ValueError("Invalid frozen dispatch-rule result: {0}".format(directory))
    result = []
    for rule in RULES:
        selected = [row for row in rows if row["rule"] == rule]
        if len(selected) != EXPECTED_INSTANCES:
            raise ValueError("Invalid {0} rows in {1}".format(rule, directory))
        validate_file_names(selected, expected_names, directory)
        result.append(
            method_row(
                rule, "none", eval_seed, "deterministic_rule", directory,
                selected, "dynamic_makespan", "stability_cost", "budget_met",
            )
        )
    return result


def aggregate_method(rows, method):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError("No rows for method {0}".format(method))
    return {
        "method": method,
        "cells": len(selected),
        "inference_mode": selected[0]["inference_mode"],
        "makespan_mean": statistics.fmean(row["makespan_mean"] for row in selected),
        "stability_cost_mean": statistics.fmean(
            row["stability_cost_mean"] for row in selected
        ),
        "budget_met_rate": statistics.fmean(row["budget_met_rate"] for row in selected),
        "release_violations": sum(row["release_violations"] for row in selected),
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    protocol_path = Path(args.protocol)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("Output directory already exists: {0}".format(output_dir))
    protocol = read_json(protocol_path)
    if protocol.get("protocol") != "P0-T_frozen_independent_test_v1":
        raise SystemExit("Unexpected P0-T protocol: {0}".format(protocol_path))
    if protocol["data"].get("content_hash_overlap_count") != 0:
        raise SystemExit("Frozen test dataset overlaps development data")
    expected_names = sorted(
        Path(item["path"]).name
        for item in protocol["data"]["independent_test"]["files"]
    )
    if len(expected_names) != EXPECTED_INSTANCES:
        raise SystemExit("Frozen protocol does not contain 100 test instances")

    rows = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            rows.extend(validate_main(save_dir, train_seed, eval_seed, expected_names))
            rows.append(validate_static(save_dir, train_seed, eval_seed, expected_names))
    for eval_seed in EVAL_SEEDS:
        rows.extend(validate_rules(save_dir, eval_seed, expected_names))

    for eval_seed in EVAL_SEEDS:
        selected = [row for row in rows if row["eval_seed"] == eval_seed]
        if len({row["arrival_sha256"] for row in selected}) != 1:
            raise ValueError("Arrival scenarios differ for eval seed {0}".format(eval_seed))
        if len({row["breakdown_sha256"] for row in selected}) != 1:
            raise ValueError("Breakdown scenarios differ for eval seed {0}".format(eval_seed))

    methods = (
        "Main_Best_of_20",
        "Initial_Best_of_20",
        "Static_DRL_greedy",
    ) + RULES
    aggregates = [aggregate_method(rows, method) for method in methods]
    main = aggregates[0]
    comparisons = [
        {
            "baseline": baseline["method"],
            "main_minus_baseline_makespan": main["makespan_mean"] - baseline["makespan_mean"],
            "main_minus_baseline_stability_cost": main["stability_cost_mean"] - baseline["stability_cost_mean"],
            "main_minus_baseline_budget_met_rate": main["budget_met_rate"] - baseline["budget_met_rate"],
        }
        for baseline in aggregates[1:]
    ]

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
        "evaluation_mode": "P0-T_frozen_independent_test_v1",
        "one_time_test": True,
        "data_dir": "data_test/1005",
        "data_manifest_sha256": protocol["data"]["independent_test"]["manifest_sha256"],
        "development_test_hash_overlap_count": 0,
        "protocol_sha256": sha256_file(protocol_path),
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "instances_per_cell": EXPECTED_INSTANCES,
        "main_output_directories": 9,
        "static_drl_output_directories": 9,
        "rule_output_directories": 3,
        "output_directories": 21,
        "method_cells": len(rows),
        "scenario_hash_groups_checked": len(EVAL_SEEDS),
        "scenario_hashes_match_across_all_methods": True,
        "release_violations": sum(row["release_violations"] for row in rows),
        "inference_protocols_are_distinct": True,
        "test_results_must_not_be_used_for_model_selection": True,
        "by_method": {row["method"]: row for row in aggregates},
        "main_vs_baselines": comparisons,
    }
    summary["passed"] = (
        summary["output_directories"] == 21
        and summary["method_cells"] == 39
        and summary["scenario_hash_groups_checked"] == 3
        and summary["scenario_hashes_match_across_all_methods"]
        and summary["release_violations"] == 0
        and all(row["release_violations"] == 0 for row in aggregates)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("output_directories={0}".format(summary["output_directories"]))
    print("method_cells={0}".format(summary["method_cells"]))
    print("scenario_hashes_match_across_all_methods=true")
    for row in aggregates:
        print(
            "{0}: makespan={1:.6f} cost={2:.6f} budget_met={3:.6f}".format(
                row["method"], row["makespan_mean"],
                row["stability_cost_mean"], row["budget_met_rate"]
            )
        )
    print("release_violations={0}".format(summary["release_violations"]))
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("P0-T frozen independent-test summary failed")


if __name__ == "__main__":
    main()
