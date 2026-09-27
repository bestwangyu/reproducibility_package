#!/usr/bin/env python3
"""Compute final-environment paired statistics without candidate pseudoreplication."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats


TRAIN_SEEDS = (20260805, 20260807, 20260808)
EVAL_SEEDS = (20260825, 20260826, 20260827)
RULES = ("FIFO", "SPT", "MWKR", "MOR")
EXPECTED_INSTANCES = 80
TOLERANCE = 1e-12


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


def bool_value(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError("{0} missing {1}".format(directory, ", ".join(missing)))


def check_summary(directory, context_mode=None):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    checks = (
        summary.get("passed") is True,
        int(summary.get("instances", -1)) == EXPECTED_INSTANCES,
        int(summary.get("sample_repeats", -1)) == 20,
        int(summary.get("sampled_rows", -1)) == 1600,
        int(summary.get("sampled_release_violations", -1)) == 0,
        summary.get("selection_mode") == "budget_first",
    )
    if not all(checks):
        raise ValueError("invalid sampled evaluation summary: {0}".format(directory))
    if context_mode is not None and summary.get("stability_context_mode") != context_mode:
        raise ValueError("unexpected context mode in {0}".format(directory))


def add_observation(store, method, train_seed, eval_seed, row, prefix="final"):
    unit = (eval_seed, row["file_name"])
    store[method][unit].append(
        {
            "train_seed": train_seed,
            "makespan": float(row[prefix + "_makespan"]),
            "stability_cost": float(row[prefix + "_stability_cost"]),
            "budget_met": float(bool_value(row[prefix + "_budget_met"])),
        }
    )


def load_sampled_method(save_dir, store, method, prefix, context_mode):
    hashes = {}
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            directory = save_dir / "{0}_train{1}_eval_{2}".format(
                prefix, train_seed, eval_seed
            )
            require_files(
                directory,
                ("arrival_scenarios.json", "breakdown_scenarios.json", "selected_results.csv", "summary.json"),
            )
            check_summary(directory, context_mode)
            rows = read_csv(directory / "selected_results.csv")
            if len(rows) != EXPECTED_INSTANCES:
                raise ValueError("invalid selected row count in {0}".format(directory))
            hashes[(method, train_seed, eval_seed)] = (
                sha256_file(directory / "arrival_scenarios.json"),
                sha256_file(directory / "breakdown_scenarios.json"),
            )
            for row in rows:
                add_observation(store, method, train_seed, eval_seed, row)
                if method == "Main_Best_of_20":
                    add_observation(
                        store, "Initial_Best_of_20", train_seed, eval_seed, row, prefix="initial"
                    )
    return hashes


def load_static(save_dir, store):
    hashes = {}
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            directory = save_dir / "finalenv_static_drl_zero_shot_train{0}_eval_{1}".format(
                train_seed, eval_seed
            )
            require_files(
                directory,
                ("arrival_scenarios.json", "breakdown_scenarios.json", "results.csv", "summary.json"),
            )
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            rows = read_csv(directory / "results.csv")
            if (
                not summary.get("passed")
                or int(summary.get("instances", -1)) != EXPECTED_INSTANCES
                or int(summary.get("release_violations", -1)) != 0
                or len(rows) != EXPECTED_INSTANCES
            ):
                raise ValueError("invalid static baseline: {0}".format(directory))
            hashes[("Static_DRL_greedy", train_seed, eval_seed)] = (
                sha256_file(directory / "arrival_scenarios.json"),
                sha256_file(directory / "breakdown_scenarios.json"),
            )
            for row in rows:
                unit = (eval_seed, row["file_name"])
                store["Static_DRL_greedy"][unit].append(
                    {
                        "train_seed": train_seed,
                        "makespan": float(row["dynamic_makespan"]),
                        "stability_cost": float(row["stability_cost"]),
                        "budget_met": float(bool_value(row["budget_met"])),
                    }
                )
    return hashes


def load_rules(save_dir, store):
    hashes = {}
    for eval_seed in EVAL_SEEDS:
        directory = save_dir / "finalenv_dispatch_rules_eval_{0}".format(eval_seed)
        require_files(
            directory,
            ("arrival_scenarios.json", "breakdown_scenarios.json", "results.csv", "summary.json"),
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = read_csv(directory / "results.csv")
        if (
            not summary.get("passed")
            or int(summary.get("instances_per_rule", -1)) != EXPECTED_INSTANCES
            or int(summary.get("result_rows", -1)) != EXPECTED_INSTANCES * len(RULES)
            or int(summary.get("release_violations", -1)) != 0
            or len(rows) != EXPECTED_INSTANCES * len(RULES)
        ):
            raise ValueError("invalid rule baseline: {0}".format(directory))
        scenario_hash = (
            sha256_file(directory / "arrival_scenarios.json"),
            sha256_file(directory / "breakdown_scenarios.json"),
        )
        for rule in RULES:
            selected = [row for row in rows if row["rule"] == rule]
            if len(selected) != EXPECTED_INSTANCES:
                raise ValueError("invalid {0} rows in {1}".format(rule, directory))
            hashes[(rule, "none", eval_seed)] = scenario_hash
            for row in selected:
                unit = (eval_seed, row["file_name"])
                store[rule][unit].append(
                    {
                        "train_seed": "none",
                        "makespan": float(row["dynamic_makespan"]),
                        "stability_cost": float(row["stability_cost"]),
                        "budget_met": float(bool_value(row["budget_met"])),
                    }
                )
    return hashes


def aggregate_units(store):
    expected_units = EXPECTED_INSTANCES * len(EVAL_SEEDS)
    rows = []
    aggregated = {}
    for method, units in store.items():
        if len(units) != expected_units:
            raise ValueError("{0} has {1} units, expected {2}".format(method, len(units), expected_units))
        aggregated[method] = {}
        expected_replicates = 1 if method in RULES else len(TRAIN_SEEDS)
        for unit, replicates in sorted(units.items()):
            if len(replicates) != expected_replicates:
                raise ValueError("{0} {1} has {2} replicates".format(method, unit, len(replicates)))
            result = {
                "eval_seed": unit[0],
                "file_name": unit[1],
                "method": method,
                "training_seed_replicates": len(replicates),
                "makespan": statistics.fmean(row["makespan"] for row in replicates),
                "stability_cost": statistics.fmean(row["stability_cost"] for row in replicates),
                "budget_met_rate": statistics.fmean(row["budget_met"] for row in replicates),
            }
            if not all(math.isfinite(result[key]) for key in ("makespan", "stability_cost", "budget_met_rate")):
                raise ValueError("non-finite aggregate for {0} {1}".format(method, unit))
            aggregated[method][unit] = result
            rows.append(result)
    return aggregated, rows


def confidence_interval(differences):
    n = len(differences)
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    if sd <= TOLERANCE:
        return mean, mean, sd
    half_width = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean - half_width, mean + half_width, sd


def wilcoxon_p(differences):
    if all(abs(value) <= TOLERANCE for value in differences):
        return 1.0
    return float(
        stats.wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def compare(main, baseline, baseline_name, metric, favorable):
    units = sorted(main)
    if units != sorted(baseline):
        raise ValueError("paired units differ for {0}".format(baseline_name))
    key = "budget_met_rate" if metric == "Budget met rate" else metric.lower().replace(" ", "_")
    main_values = [main[unit][key] for unit in units]
    baseline_values = [baseline[unit][key] for unit in units]
    differences = [left - right for left, right in zip(main_values, baseline_values)]
    ci_low, ci_high, difference_sd = confidence_interval(differences)
    difference_mean = statistics.fmean(differences)
    paired_dz = difference_mean / difference_sd if difference_sd > TOLERANCE else 0.0
    if favorable == "lower":
        wins = sum(value < -TOLERANCE for value in differences)
        losses = sum(value > TOLERANCE for value in differences)
    else:
        wins = sum(value > TOLERANCE for value in differences)
        losses = sum(value < -TOLERANCE for value in differences)
    ties = len(differences) - wins - losses
    return {
        "baseline": baseline_name,
        "metric": metric,
        "paired_unit": "instance_x_evaluation_seed_after_averaging_training_seeds",
        "n": len(differences),
        "baseline_mean": statistics.fmean(baseline_values),
        "main_mean": statistics.fmean(main_values),
        "main_minus_baseline_mean": difference_mean,
        "difference_sd": difference_sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": wilcoxon_p(differences),
        "holm_adjusted_p": None,
        "paired_dz": paired_dz,
        "main_wins": wins,
        "ties": ties,
        "main_losses": losses,
    }


def holm_adjust(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["wilcoxon_p"])
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * row["wilcoxon_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("output directory already exists: {0}".format(output_dir))

    store = defaultdict(lambda: defaultdict(list))
    hashes = {}
    hashes.update(load_sampled_method(save_dir, store, "Main_Best_of_20", "finalenv_nominal_holdout", "full"))
    hashes.update(load_sampled_method(save_dir, store, "Reward_only_Best_of_20", "finalenv_reward_only_holdout", "full"))
    hashes.update(load_sampled_method(save_dir, store, "Zero_context_Best_of_20", "finalenv_zero_context_holdout", "zero"))
    hashes.update(load_static(save_dir, store))
    hashes.update(load_rules(save_dir, store))

    for eval_seed in EVAL_SEEDS:
        selected = [value for key, value in hashes.items() if key[-1] == eval_seed]
        if len({value[0] for value in selected}) != 1 or len({value[1] for value in selected}) != 1:
            raise ValueError("scenario hashes differ for evaluation seed {0}".format(eval_seed))

    aggregated, observation_rows = aggregate_units(store)
    main = aggregated["Main_Best_of_20"]
    baseline_names = (
        "Initial_Best_of_20",
        "Static_DRL_greedy",
        "FIFO",
        "SPT",
        "MWKR",
        "MOR",
        "Reward_only_Best_of_20",
        "Zero_context_Best_of_20",
    )
    comparison_rows = []
    for baseline in baseline_names:
        comparison_rows.append(compare(main, aggregated[baseline], baseline, "Makespan", "lower"))
        comparison_rows.append(compare(main, aggregated[baseline], baseline, "Stability cost", "lower"))
        comparison_rows.append(compare(main, aggregated[baseline], baseline, "Budget met rate", "higher"))
    holm_adjust(comparison_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "paired_unit_observations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observation_rows[0]))
        writer.writeheader()
        writer.writerows(observation_rows)
    with (output_dir / "paired_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "analysis": "final_environment_paired_statistics",
        "paired_unit": "instance_x_evaluation_seed_after_averaging_training_seeds",
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "paired_units_per_comparison": EXPECTED_INSTANCES * len(EVAL_SEEDS),
        "candidate_rollouts_treated_as_independent": False,
        "training_seeds_averaged_within_paired_unit": True,
        "comparisons": len(baseline_names),
        "metric_tests": len(comparison_rows),
        "continuous_ci": "two-sided 95% Student-t CI of paired mean difference",
        "test": "two-sided Wilcoxon signed-rank",
        "effect_size": "paired dz = mean paired difference / SD paired difference",
        "multiple_testing": "Holm correction across all reported metric tests",
        "scenario_hash_groups_checked": len(EVAL_SEEDS),
        "scenario_hashes_match_across_methods": True,
        "finite": True,
        "passed": len(comparison_rows) == len(baseline_names) * 3,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Final-environment paired statistics\n\n")
        handle.write(
            "Paired unit: instance x evaluation seed (n=240); the three training "
            "seeds are averaged within each unit. Candidate rollouts are not independent samples.\n\n"
        )
        handle.write("| Baseline | Metric | Main - baseline | 95% CI | Wilcoxon p | Holm p | dz | W/T/L |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in comparison_rows:
            handle.write(
                "| {baseline} | {metric} | {main_minus_baseline_mean:.6f} | "
                "[{ci95_low:.6f}, {ci95_high:.6f}] | {wilcoxon_p:.3e} | "
                "{holm_adjusted_p:.3e} | {paired_dz:.4f} | "
                "{main_wins}/{ties}/{main_losses} |\n".format(**row)
            )

    print("paired_units_per_comparison={0}".format(summary["paired_units_per_comparison"]))
    print("comparisons={0}".format(summary["comparisons"]))
    print("metric_tests={0}".format(summary["metric_tests"]))
    print("scenario_hashes_match_across_methods=True")
    print("candidate_rollouts_treated_as_independent=False")
    print("passed={0}".format(summary["passed"]))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("final-environment paired statistics failed")


if __name__ == "__main__":
    main()
