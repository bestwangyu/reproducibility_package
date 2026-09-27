#!/usr/bin/env python3
"""Evaluate a static DRL checkpoint zero-shot under dynamic disruptions."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from dynamic.arrivals import JobArrivalGenerator, save_scenarios
from dynamic.breakdowns import (
    MachineBreakdownGenerator,
    MachineBreakdownScenario,
    save_breakdown_scenarios,
)
from evaluate_constrained_pilot import evaluate, load_policy
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
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument(
        "--reference-checkpoint",
        default="checkpoints/base/reference_save_10_5.pt",
        help="Static checkpoint used to construct the common arrival-only plan.",
    )
    parser.add_argument("--policy-seed", type=int, required=True)
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


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def count_release_violations(environment, arrivals):
    violations = 0
    for index, scenario in enumerate(arrivals):
        for job in range(scenario.initial_jobs, scenario.num_jobs):
            operation = int(environment.num_ope_biases_batch[index, job].item())
            start = float(environment.schedules_batch[index, operation, 2].item())
            if start + 1e-5 < scenario.release_times[job]:
                violations += 1
    return violations


def build_result_rows(paths, reference_makespans, evaluated_rows):
    rows = []
    for path, reference_makespan, evaluated in zip(
        paths, reference_makespans, evaluated_rows
    ):
        dynamic_makespan = float(evaluated["makespan"])
        rows.append(
            {
                "file_name": path.name,
                "reference_makespan": float(reference_makespan),
                "dynamic_makespan": dynamic_makespan,
                "disruption_makespan_delta": (
                    dynamic_makespan - float(reference_makespan)
                ),
                "stability_cost": float(evaluated["stability_cost"]),
                "budget_violation": float(evaluated["budget_violation"]),
                "budget_met": bool(evaluated["budget_met"]),
                "release_violations": int(evaluated["release_violations"]),
            }
        )
    return rows


def build_summary(
    args,
    policy_checkpoint,
    reference_checkpoint,
    reference_feasible,
    reference_release_violations,
    policy_feasible,
    rows,
):
    release_violations = sum(row["release_violations"] for row in rows)
    summary = {
        "evaluation_mode": "static_drl_dynamic_zero_shot",
        "policy_type": "static_drl",
        "inference_mode": "greedy",
        "policy_seed": args.policy_seed,
        "eval_seed": args.seed,
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "reference_definition": "arrival_only_common_static_plan",
        "instances": args.instances,
        "offset": args.offset,
        "initial_jobs": args.initial_jobs,
        "mean_interarrival": args.mean_interarrival,
        "mean_time_between_failures": args.mean_time_between_failures,
        "mean_repair_time": args.mean_repair_time,
        "events_per_machine": args.events_per_machine,
        "stability_budget": args.stability_budget,
        "reference_feasible": reference_feasible,
        "policy_feasible": policy_feasible,
        "reference_release_violations": reference_release_violations,
        "release_violations": release_violations,
        "reference_makespan_mean": mean(rows, "reference_makespan"),
        "dynamic_makespan_mean": mean(rows, "dynamic_makespan"),
        "disruption_makespan_delta_mean": mean(
            rows, "disruption_makespan_delta"
        ),
        "stability_cost_mean": mean(rows, "stability_cost"),
        "budget_met": sum(bool(row["budget_met"]) for row in rows),
    }
    summary["budget_met_rate"] = summary["budget_met"] / args.instances
    summary["passed"] = (
        reference_feasible
        and policy_feasible
        and reference_release_violations == 0
        and release_violations == 0
    )
    return summary


def validate_args(args):
    if args.offset < 0 or args.instances <= 0:
        raise SystemExit("offset must be non-negative and instances must be positive.")
    if args.initial_jobs <= 0:
        raise SystemExit("initial-jobs must be positive.")
    if args.events_per_machine <= 0:
        raise SystemExit("events-per-machine must be positive.")
    for name, value in (
        ("mean-interarrival", args.mean_interarrival),
        ("mean-time-between-failures", args.mean_time_between_failures),
        ("mean-repair-time", args.mean_repair_time),
    ):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit("{0} must be finite and positive.".format(name))
    if not math.isfinite(args.stability_budget) or args.stability_budget < 0:
        raise SystemExit("stability-budget must be finite and non-negative.")


def main():
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    configure_device(device)
    setup_seed(args.seed)

    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    policy_checkpoint = resolve_project_path(args.policy_checkpoint)
    reference_checkpoint = resolve_project_path(args.reference_checkpoint)
    for label, path in (
        ("policy checkpoint", policy_checkpoint),
        ("reference checkpoint", reference_checkpoint),
    ):
        if not path.is_file():
            raise SystemExit("Missing {0}: {1}".format(label, path))

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

    policy = load_policy(config, device, policy_checkpoint, 0)
    policy_environment = make_environment(
        paths,
        arrivals,
        breakdowns,
        config,
        device,
        references,
        args.stability_budget,
    )
    policy_feasible, evaluated_rows = evaluate(
        policy,
        policy_environment,
        arrivals,
        args.stability_budget,
        sample=False,
    )
    result_rows = build_result_rows(paths, reference_makespans, evaluated_rows)
    summary = build_summary(
        args,
        policy_checkpoint,
        reference_checkpoint,
        reference_feasible,
        reference_release_violations,
        policy_feasible,
        result_rows,
    )

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else PROJECT_DIR
        / "save"
        / "static_drl_dynamic_zero_shot_{0}".format(
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    for key, value in summary.items():
        print("{0}={1}".format(key, value))
    print("results: {0}".format(output_dir))


if __name__ == "__main__":
    main()
