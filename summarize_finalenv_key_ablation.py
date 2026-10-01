#!/usr/bin/env python3
"""Validate and summarize one final-environment 3x3 key ablation."""

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
    parser.add_argument("--directory-prefix", required=True)
    parser.add_argument("--expected-context-mode", choices=("full", "zero"), required=True)
    parser.add_argument(
        "--expected-selection-mode",
        choices=("budget_first", "makespan_only", "min_cost"),
        default="budget_first",
    )
    parser.add_argument("--reference-prefix", default="finalenv_nominal_holdout")
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


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def mean(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite or empty metric: {0}".format(key))
    return statistics.fmean(values)


def validate_cell(
    directory, reference, train_seed, eval_seed, context_mode, selection_mode
):
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
    if not reference.is_dir():
        raise ValueError("missing Exp-027 reference directory: {0}".format(reference))

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "instances": EXPECTED_INSTANCES,
        "offset": 20,
        "sample_repeats": EXPECTED_REPEATS,
        "sampled_rows": EXPECTED_INSTANCES * EXPECTED_REPEATS,
        "release_violations": 0,
        "sampled_release_violations": 0,
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            raise ValueError("unexpected {0} in {1}".format(key, directory))
    if not close(summary.get("budget", -1), EXPECTED_BUDGET):
        raise ValueError("unexpected budget in {0}".format(directory))
    if summary.get("stability_context_mode") != context_mode:
        raise ValueError("unexpected context mode in {0}".format(directory))
    if summary.get("selection_mode") != selection_mode:
        raise ValueError("unexpected selection mode in {0}".format(directory))
    if not summary.get("passed") or not summary.get("sampled_feasible"):
        raise ValueError("evaluation did not pass in {0}".format(directory))

    results = read_csv(directory / "results.csv")
    sampled = read_csv(directory / "sampled_results.csv")
    selected = read_csv(directory / "selected_results.csv")
    reference_selected = read_csv(reference / "selected_results.csv")
    if (len(results), len(sampled), len(selected), len(reference_selected)) != (
        80, 1600, 80, 80
    ):
        raise ValueError("unexpected CSV row counts in {0}".format(directory))
    if sum(int(row["release_violations"]) for row in results + sampled) != 0:
        raise ValueError("release violation in {0}".format(directory))

    arrival_hash = sha256_file(directory / "arrival_scenarios.json")
    breakdown_hash = sha256_file(directory / "breakdown_scenarios.json")
    if arrival_hash != sha256_file(reference / "arrival_scenarios.json"):
        raise ValueError("arrival scenarios differ from Exp-027 in {0}".format(directory))
    if breakdown_hash != sha256_file(reference / "breakdown_scenarios.json"):
        raise ValueError("breakdown scenarios differ from Exp-027 in {0}".format(directory))

    reference_by_file = {row["file_name"]: row for row in reference_selected}
    if set(reference_by_file) != {row["file_name"] for row in selected}:
        raise ValueError("instance names differ from Exp-027 in {0}".format(directory))

    paired = []
    for row in selected:
        reference_row = reference_by_file[row["file_name"]]
        paired.append({
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "file_name": row["file_name"],
            "ablation_final_makespan": float(row["final_makespan"]),
            "main_final_makespan": float(reference_row["final_makespan"]),
            "ablation_minus_main_makespan": (
                float(row["final_makespan"]) - float(reference_row["final_makespan"])
            ),
            "ablation_final_stability_cost": float(row["final_stability_cost"]),
            "main_final_stability_cost": float(reference_row["final_stability_cost"]),
            "ablation_minus_main_stability_cost": (
                float(row["final_stability_cost"])
                - float(reference_row["final_stability_cost"])
            ),
            "ablation_budget_met": bool_value(row["final_budget_met"]),
            "main_budget_met": bool_value(reference_row["final_budget_met"]),
        })

    initial_budget_met = sum(bool_value(row["initial_budget_met"]) for row in selected)
    final_budget_met = sum(bool_value(row["final_budget_met"]) for row in selected)
    return {
        "cell": {
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "output_dir": str(directory),
            "selected_initial_makespan_mean": mean(selected, "initial_makespan"),
            "selected_final_makespan_mean": mean(selected, "final_makespan"),
            "selected_makespan_delta": mean(selected, "makespan_delta"),
            "selected_initial_stability_cost_mean": mean(selected, "initial_stability_cost"),
            "selected_final_stability_cost_mean": mean(selected, "final_stability_cost"),
            "selected_stability_cost_delta": mean(selected, "stability_cost_delta"),
            "selected_initial_budget_met_rate": initial_budget_met / 80.0,
            "selected_final_budget_met_rate": final_budget_met / 80.0,
            "selected_budget_met_rate_delta": (final_budget_met - initial_budget_met) / 80.0,
            "arrival_sha256": arrival_hash,
            "breakdown_sha256": breakdown_hash,
            "scenario_hashes_match_exp027": True,
            "passed": True,
        },
        "paired": paired,
    }


def aggregate(rows, aggregation, train_seed):
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
    result = {"aggregation": aggregation, "train_seed": train_seed, "cells": len(rows)}
    result.update({key: statistics.fmean(row[key] for row in rows) for key in metric_keys})
    return result


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("output directory already exists: {0}".format(output_dir))

    cells = []
    paired = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            directory = save_dir / "{0}_train{1}_eval_{2}".format(
                args.directory_prefix, train_seed, eval_seed
            )
            reference = save_dir / "{0}_train{1}_eval_{2}".format(
                args.reference_prefix, train_seed, eval_seed
            )
            validated = validate_cell(
                directory,
                reference,
                train_seed,
                eval_seed,
                args.expected_context_mode,
                args.expected_selection_mode,
            )
            cells.append(validated["cell"])
            paired.extend(validated["paired"])

    aggregates = [
        aggregate(
            [row for row in cells if row["train_seed"] == train_seed],
            "train_seed",
            train_seed,
        )
        for train_seed in TRAIN_SEEDS
    ]
    aggregates.append(aggregate(cells, "all_train_seeds", "all"))
    overall = aggregates[-1]

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("evaluation_cells.csv", cells),
        ("aggregate.csv", aggregates),
        ("paired_with_main.csv", paired),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    paired_summary = {
        "ablation_minus_main_makespan_mean": statistics.fmean(
            row["ablation_minus_main_makespan"] for row in paired
        ),
        "ablation_minus_main_stability_cost_mean": statistics.fmean(
            row["ablation_minus_main_stability_cost"] for row in paired
        ),
        "ablation_budget_met_rate": statistics.fmean(
            row["ablation_budget_met"] for row in paired
        ),
        "main_budget_met_rate": statistics.fmean(row["main_budget_met"] for row in paired),
    }
    paired_summary["ablation_minus_main_budget_met_rate"] = (
        paired_summary["ablation_budget_met_rate"]
        - paired_summary["main_budget_met_rate"]
    )
    summary = {
        "evaluation_mode": "final_environment_key_ablation_3x3",
        "directory_prefix": args.directory_prefix,
        "reference_prefix": args.reference_prefix,
        "expected_context_mode": args.expected_context_mode,
        "expected_selection_mode": args.expected_selection_mode,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "evaluation_cells": len(cells),
        "instances_per_cell": EXPECTED_INSTANCES,
        "sample_repeats": EXPECTED_REPEATS,
        "selected_rows": len(paired),
        "sampled_rows": 9 * EXPECTED_INSTANCES * EXPECTED_REPEATS,
        "release_violations": 0,
        "scenario_hashes_match_exp027": all(
            row["scenario_hashes_match_exp027"] for row in cells
        ),
        "all_cells_passed": all(row["passed"] for row in cells),
        "overall": overall,
        "paired_with_main": paired_summary,
    }
    summary["passed"] = (
        summary["evaluation_cells"] == 9
        and summary["selected_rows"] == 720
        and summary["sampled_rows"] == 14400
        and summary["release_violations"] == 0
        and summary["scenario_hashes_match_exp027"]
        and summary["all_cells_passed"]
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("evaluation_cells={0}".format(summary["evaluation_cells"]))
    print("selected_rows={0}".format(summary["selected_rows"]))
    print("sampled_rows={0}".format(summary["sampled_rows"]))
    print("scenario_hashes_match_exp027={0}".format(
        summary["scenario_hashes_match_exp027"]
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
    print("ablation_minus_main_makespan={0:.6f}".format(
        paired_summary["ablation_minus_main_makespan_mean"]
    ))
    print("ablation_minus_main_stability_cost={0:.6f}".format(
        paired_summary["ablation_minus_main_stability_cost_mean"]
    ))
    print("ablation_minus_main_budget_met_rate={0:.6f}".format(
        paired_summary["ablation_minus_main_budget_met_rate"]
    ))
    print("release_violations=0")
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("final-environment key-ablation summary failed")


if __name__ == "__main__":
    main()
