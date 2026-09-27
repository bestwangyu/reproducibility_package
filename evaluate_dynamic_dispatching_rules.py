#!/usr/bin/env python3
"""Evaluate deterministic dispatching-rule baselines under dynamic disruptions."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from dispatching_rules import RULE_DEFINITIONS, RULES, select_dispatching_actions
from dynamic.arrivals import JobArrivalGenerator, save_scenarios
from dynamic.breakdowns import (
    MachineBreakdownGenerator,
    MachineBreakdownScenario,
    save_breakdown_scenarios,
)
from evaluate_constrained_pilot import load_policy
from evaluate_static_dynamic_baseline import (
    count_release_violations,
    resolve_project_path,
    sha256_file,
)
from train_constrained_dynamic import (
    PROJECT_DIR,
    configure_device,
    make_environment,
    reference_schedules,
    select_device,
    setup_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", default="checkpoints/base/reference_save_10_5.pt")
    parser.add_argument("--data-dir", default="data_dev/1005")
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--instances", type=int, default=80)
    parser.add_argument("--initial-jobs", type=int, default=6)
    parser.add_argument("--mean-interarrival", type=float, default=8.0)
    parser.add_argument("--mean-time-between-failures", type=float, default=8.0)
    parser.add_argument("--mean-repair-time", type=float, default=2.0)
    parser.add_argument("--events-per-machine", type=int, default=1)
    parser.add_argument("--stability-budget", type=float, default=1.7)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def validate_args(args):
    if args.offset < 0 or args.instances <= 0:
        raise SystemExit("offset must be non-negative and instances must be positive.")
    if args.initial_jobs <= 0 or args.events_per_machine <= 0:
        raise SystemExit("initial-jobs and events-per-machine must be positive.")
    for name, value in (
        ("mean-interarrival", args.mean_interarrival),
        ("mean-time-between-failures", args.mean_time_between_failures),
        ("mean-repair-time", args.mean_repair_time),
    ):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit("{0} must be finite and positive.".format(name))
    if not math.isfinite(args.stability_budget) or args.stability_budget < 0:
        raise SystemExit("stability-budget must be finite and non-negative.")


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def evaluate_rule(environment, arrivals, rule, budget):
    steps = 0
    while not bool(environment.done_batch.all().item()):
        environment.step(select_dispatching_actions(environment, rule))
        steps += 1
        if steps > environment.num_opes + 1:
            raise RuntimeError(
                "Rule {0} exceeded the operation-count limit.".format(rule)
            )

    feasible = bool(environment.validate_gantt()[0])
    release_violations = count_release_violations(environment, arrivals)
    rows = []
    for index in range(len(arrivals)):
        cost = float(environment.stability_cost_mean_batch[index].item())
        rows.append(
            {
                "makespan": float(environment.makespan_batch[index].item()),
                "stability_cost": cost,
                "budget_violation": max(0.0, cost - budget),
                "budget_met": cost <= budget + 1e-6,
            }
        )
    return feasible, release_violations, rows


def main():
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    configure_device(device)
    setup_seed(args.seed)

    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    reference_checkpoint = resolve_project_path(args.reference_checkpoint)
    if not reference_checkpoint.is_file():
        raise SystemExit("Missing reference checkpoint: {0}".format(reference_checkpoint))
    data_dir = resolve_project_path(args.data_dir)
    paths = sorted(data_dir.glob("*.fjs"))[
        args.offset : args.offset + args.instances
    ]
    if len(paths) != args.instances:
        raise SystemExit("Not enough held-out instances after the requested offset.")

    arrival_generator = JobArrivalGenerator(args.seed, args.mean_interarrival)
    breakdown_generator = MachineBreakdownGenerator(
        args.seed + 1,
        args.mean_time_between_failures,
        args.mean_repair_time,
    )
    arrivals = [
        arrival_generator.generate(
            config["env_paras"]["num_jobs"],
            args.initial_jobs,
            scenario_id="heldout-arrival:{0}".format(path.stem),
            random_stream_id=path.stem,
        )
        for path in paths
    ]
    breakdowns = [
        breakdown_generator.generate(
            config["env_paras"]["num_mas"],
            args.events_per_machine,
            scenario_id="heldout-breakdown:{0}".format(path.stem),
            random_stream_id=path.stem,
        )
        for path in paths
    ]

    reference_policy = load_policy(config, device, reference_checkpoint, 0)
    reference_environment = make_environment(
        paths,
        arrivals,
        [MachineBreakdownScenario.static(item.num_machines) for item in breakdowns],
        config,
        device,
    )
    references = reference_schedules(reference_policy, reference_environment)
    reference_feasible = bool(reference_environment.validate_gantt()[0])
    reference_release_violations = count_release_violations(
        reference_environment, arrivals
    )
    reference_makespans = [
        float(value) for value in reference_environment.makespan_batch.tolist()
    ]

    result_rows = []
    rule_summaries = {}
    all_rules_feasible = True
    total_release_violations = 0
    for rule in RULES:
        environment = make_environment(
            paths,
            arrivals,
            breakdowns,
            config,
            device,
            references,
            args.stability_budget,
        )
        feasible, release_violations, rows = evaluate_rule(
            environment, arrivals, rule, args.stability_budget
        )
        all_rules_feasible = all_rules_feasible and feasible
        total_release_violations += release_violations
        for path, reference_makespan, row in zip(
            paths, reference_makespans, rows
        ):
            result_rows.append(
                {
                    "rule": rule,
                    "file_name": path.name,
                    "reference_makespan": reference_makespan,
                    "dynamic_makespan": row["makespan"],
                    "disruption_makespan_delta": row["makespan"]
                    - reference_makespan,
                    "stability_cost": row["stability_cost"],
                    "budget_violation": row["budget_violation"],
                    "budget_met": row["budget_met"],
                    "release_violations": 0,
                }
            )
        rule_result_rows = [row for row in result_rows if row["rule"] == rule]
        rule_summaries[rule] = {
            "feasible": feasible,
            "release_violations": release_violations,
            "dynamic_makespan_mean": mean(rule_result_rows, "dynamic_makespan"),
            "disruption_makespan_delta_mean": mean(
                rule_result_rows, "disruption_makespan_delta"
            ),
            "stability_cost_mean": mean(rule_result_rows, "stability_cost"),
            "budget_met": sum(bool(row["budget_met"]) for row in rule_result_rows),
            "budget_met_rate": sum(
                bool(row["budget_met"]) for row in rule_result_rows
            )
            / args.instances,
        }
        print(
            "rule={0} feasible={1} makespan={2:.6f} cost={3:.6f} "
            "budget_met_rate={4:.6f} release_violations={5}".format(
                rule,
                feasible,
                rule_summaries[rule]["dynamic_makespan_mean"],
                rule_summaries[rule]["stability_cost_mean"],
                rule_summaries[rule]["budget_met_rate"],
                release_violations,
            ),
            flush=True,
        )

    summary = {
        "evaluation_mode": "dynamic_dispatching_rule_baselines",
        "inference_mode": "deterministic",
        "rules": list(RULES),
        "eval_seed": args.seed,
        "reference_checkpoint": str(reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "reference_definition": "arrival_only_common_static_plan",
        "instances_per_rule": args.instances,
        "result_rows": len(result_rows),
        "offset": args.offset,
        "initial_jobs": args.initial_jobs,
        "mean_interarrival": args.mean_interarrival,
        "mean_time_between_failures": args.mean_time_between_failures,
        "mean_repair_time": args.mean_repair_time,
        "events_per_machine": args.events_per_machine,
        "stability_budget": args.stability_budget,
        "reference_feasible": reference_feasible,
        "reference_release_violations": reference_release_violations,
        "reference_makespan_mean": sum(reference_makespans) / len(reference_makespans),
        "all_rules_feasible": all_rules_feasible,
        "release_violations": total_release_violations,
        "by_rule": rule_summaries,
    }
    summary["passed"] = (
        reference_feasible
        and reference_release_violations == 0
        and all_rules_feasible
        and total_release_violations == 0
        and len(result_rows) == args.instances * len(RULES)
    )

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else PROJECT_DIR
        / "save"
        / "dispatch_rule_dynamic_baselines_eval_{0}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    save_scenarios(arrivals, output_dir / "arrival_scenarios.json")
    save_breakdown_scenarios(breakdowns, output_dir / "breakdown_scenarios.json")
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (output_dir / "rule_definitions.json").write_text(
        json.dumps(RULE_DEFINITIONS, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    for key, value in summary.items():
        if key != "by_rule":
            print("{0}={1}".format(key, value))
    print("results: {0}".format(output_dir))
    if not summary["passed"]:
        raise SystemExit("Dispatching-rule evaluation failed validation.")


if __name__ == "__main__":
    main()
