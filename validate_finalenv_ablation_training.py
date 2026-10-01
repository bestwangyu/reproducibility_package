#!/usr/bin/env python3
"""Validate one final-environment key-ablation training directory."""

import argparse
import csv
import json
import math
from pathlib import Path


PROTOCOLS = {
    "makespan_only": {
        "reward_coeff": 1.0,
        "lambda_init": 0.0,
        "lambda_lr": 0.0,
        "lambda_max": 0.0,
        "stability_context_mode": "zero",
    },
    "reward_only": {
        "reward_coeff": 1.0,
        "lambda_init": 0.0,
        "lambda_lr": 0.0,
        "lambda_max": 0.0,
        "stability_context_mode": "full",
    },
    "zero_context": {
        "reward_coeff": 0.0,
        "lambda_init": 1.0,
        "lambda_lr": 0.0,
        "lambda_max": 1.0,
        "stability_context_mode": "zero",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--ablation", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    return parser.parse_args()


def close(left, right, tolerance=1e-12):
    return abs(float(left) - float(right)) <= tolerance


def main():
    args = parse_args()
    directory = Path(args.directory)
    required = (
        "final_model.pt",
        "conditioned_initial.pt",
        "constrained_checkpoint.pt",
        "training_results.csv",
        "metadata.json",
        "summary.json",
        "completion.json",
        "checkpoint_migration.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise SystemExit("missing files in {0}: {1}".format(directory, ", ".join(missing)))

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    protocol = PROTOCOLS[args.ablation]

    exact_metadata = {
        "iterations": 100,
        "instances": 20,
        "batch_size": 20,
        "initial_jobs": 6,
        "events_per_machine": 1,
        "train_seed": args.train_seed,
        "scenario_seed": args.train_seed + 10,
        "cost_advantage_mode": "batch_centered",
        "cost_signal": "raw",
        "stability_context_mode": protocol["stability_context_mode"],
    }
    for key, expected in exact_metadata.items():
        if metadata.get(key) != expected:
            raise SystemExit("unexpected metadata {0}={1!r}, expected {2!r}".format(
                key, metadata.get(key), expected
            ))
    numeric_metadata = {
        "mean_interarrival": 8.0,
        "mean_time_between_failures": 8.0,
        "mean_repair_time": 2.0,
        "stability_budget": 1.7,
        "lr": 0.0001,
        "reward_coeff": protocol["reward_coeff"],
        "lambda_init": protocol["lambda_init"],
        "lambda_lr": protocol["lambda_lr"],
        "lambda_max": protocol["lambda_max"],
    }
    for key, expected in numeric_metadata.items():
        if not close(metadata.get(key, float("nan")), expected):
            raise SystemExit("unexpected metadata {0}={1!r}, expected {2!r}".format(
                key, metadata.get(key), expected
            ))

    expected_summary = {
        "iterations": 100,
        "rows": 100,
        "finite_rows": 100,
        "feasible_rows": 100,
        "release_violations": 0,
        "invalid_actions": 0,
    }
    for key, expected in expected_summary.items():
        if int(summary.get(key, -1)) != expected:
            raise SystemExit("unexpected summary {0} in {1}".format(key, directory))
    if not summary.get("passed") or not completion.get("completed") or not completion.get("passed"):
        raise SystemExit("training did not pass in {0}".format(directory))
    if not close(summary.get("lambda_final", float("nan")), protocol["lambda_init"]):
        raise SystemExit("unexpected final lambda in {0}".format(directory))

    with (directory / "training_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise SystemExit("training_results.csv must contain 100 rows")
    numeric_keys = ("makespan_mean", "stability_cost_mean", "loss", "cost_value_loss")
    if not all(
        all(math.isfinite(float(row[key])) for key in numeric_keys)
        for row in rows
    ):
        raise SystemExit("non-finite training metric in {0}".format(directory))
    if any(int(float(row["release_violations"])) != 0 for row in rows):
        raise SystemExit("release violation in {0}".format(directory))

    print("ablation={0}".format(args.ablation))
    print("train_seed={0}".format(args.train_seed))
    print("rows=100")
    print("finite_rows=100")
    print("feasible_rows=100")
    print("release_violations=0")
    print("invalid_actions=0")
    print("passed=true")


if __name__ == "__main__":
    main()
