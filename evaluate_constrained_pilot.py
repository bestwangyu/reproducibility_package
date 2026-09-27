#!/usr/bin/env python3
"""Compare the initial and trained budget-conditioned policies on held-out cases."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from constrained_ppo import ConstrainedMemory
from dynamic.arrivals import JobArrivalGenerator, save_scenarios
from dynamic.breakdowns import (
    MachineBreakdownGenerator,
    MachineBreakdownScenario,
    save_breakdown_scenarios,
)
from PPO_model import HGNNScheduler
from train_constrained_dynamic import (
    PROJECT_DIR,
    configure_device,
    make_environment,
    model_parameters,
    reference_schedules,
    select_device,
    setup_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/base/reference_save_10_5.pt")
    parser.add_argument("--data-dir", default="data_dev/1005")
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--initial-jobs", type=int, default=6)
    parser.add_argument("--mean-interarrival", type=float, default=8.0)
    parser.add_argument("--mean-time-between-failures", type=float, default=8.0)
    parser.add_argument("--mean-repair-time", type=float, default=2.0)
    parser.add_argument("--events-per-machine", type=int, default=1)
    parser.add_argument("--stability-budget", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--sample-repeats", type=int, default=0)
    parser.add_argument("--budget-repair", action="store_true")
    parser.add_argument(
        "--stability-context-mode",
        choices=("full", "zero"),
        default="full",
        help="Use the learned stability context or replace all three context features with zeros.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("budget_first", "makespan_only", "min_cost"),
        default="budget_first",
        help="Select one candidate from repeated stochastic rollouts.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_policy(config, device, path, context_dim, context_mode="full"):
    policy = HGNNScheduler(
        model_parameters(config, device, context_dim, context_mode)
    ).to(device)
    policy.load_state_dict(torch.load(str(path), map_location=device))
    policy.eval()
    return policy


def budget_repaired_actions(policy, environment, sample):
    """Apply a per-remaining-operation budget mask to policy actions."""
    state = environment.state
    action_probs, operation_steps, _ = policy.get_action_prob(
        state, None, flag_sample=sample, flag_train=False
    )
    batch_indexes = state.batch_idxes
    candidate_costs = state.candidate_stability_cost_batch[
        batch_indexes
    ].transpose(1, 2).flatten(1)
    eligible = action_probs > 0
    remaining_operations = (
        state.nums_opes_batch[batch_indexes]
        - environment.stability_accounted_batch[batch_indexes].sum(dim=1)
    ).clamp_min(1)
    remaining_total = (
        environment.stability_budget_total_batch[batch_indexes]
        - environment.stability_cost_sum_batch[batch_indexes]
    ).clamp_min(0.0)
    allowance = remaining_total / remaining_operations.to(remaining_total.dtype)
    allowed = eligible & (candidate_costs <= allowance[:, None] + 1e-8)

    no_budget_action = ~allowed.any(dim=1)
    if no_budget_action.any():
        infinity = torch.full_like(candidate_costs, float("inf"))
        minimum_cost = torch.where(
            eligible, candidate_costs, infinity
        ).min(dim=1, keepdim=True)[0]
        fallback = eligible & torch.isclose(
            candidate_costs, minimum_cost, rtol=1e-6, atol=1e-8
        )
        allowed[no_budget_action] = fallback[no_budget_action]

    repaired_probs = action_probs * allowed.to(action_probs.dtype)
    repaired_probs = repaired_probs / repaired_probs.sum(dim=1, keepdim=True)
    if sample:
        action_indexes = torch.multinomial(repaired_probs, 1).squeeze(1)
    else:
        action_indexes = repaired_probs.argmax(dim=1)
    jobs = (action_indexes % environment.num_jobs).long()
    machines = (action_indexes / environment.num_jobs).long()
    operations = operation_steps[batch_indexes, jobs]
    return torch.stack((operations, machines, jobs), dim=1).t()


def evaluate(
    policy, environment, arrivals, budget, sample=False, budget_repair=False
):
    memory = ConstrainedMemory()
    steps = 0
    while not bool(environment.done_batch.all().item()):
        with torch.no_grad():
            if budget_repair:
                actions = budget_repaired_actions(policy, environment, sample)
            else:
                actions = policy.act(
                    environment.state,
                    memory,
                    environment.done_batch,
                    flag_sample=sample,
                    flag_train=False,
                )
        environment.step(actions)
        steps += 1
        if steps > environment.num_opes + 1:
            raise RuntimeError("Held-out evaluation exceeded operation-count limit.")

    feasible = bool(environment.validate_gantt()[0])
    rows = []
    for index, scenario in enumerate(arrivals):
        release_violations = 0
        for job in range(scenario.initial_jobs, scenario.num_jobs):
            operation = int(environment.num_ope_biases_batch[index, job].item())
            start = float(environment.schedules_batch[index, operation, 2].item())
            if start + 1e-5 < scenario.release_times[job]:
                release_violations += 1
        cost = float(environment.stability_cost_mean_batch[index].item())
        rows.append(
            {
                "makespan": float(environment.makespan_batch[index].item()),
                "stability_cost": cost,
                "budget_violation": max(0.0, cost - budget),
                "budget_met": cost <= budget + 1e-6,
                "release_violations": release_violations,
            }
        )
    return feasible, rows


def minimum_cost_actions(environment):
    """Choose the eligible operation-machine pair with minimum immediate cost."""
    state = environment.state
    batch_indexes = state.batch_idxes
    operation_steps = torch.where(
        state.ope_step_batch > state.end_ope_biases_batch,
        state.end_ope_biases_batch,
        state.ope_step_batch,
    )
    eligible_process = state.ope_ma_adj_batch[batch_indexes].gather(
        1,
        operation_steps[..., :, None].expand(
            -1, -1, state.ope_ma_adj_batch.size(-1)
        )[batch_indexes],
    ) == 1
    machine_blocked = state.mask_ma_procing_batch[batch_indexes]
    if state.mask_ma_failed_batch is not None:
        machine_blocked = machine_blocked | state.mask_ma_failed_batch[batch_indexes]
    job_blocked = (
        state.mask_job_procing_batch[batch_indexes]
        | state.mask_job_finish_batch[batch_indexes]
    )
    if state.mask_job_unreleased_batch is not None:
        job_blocked = job_blocked | state.mask_job_unreleased_batch[batch_indexes]
    eligible = (
        eligible_process
        & ~machine_blocked[:, None, :]
        & ~job_blocked[:, :, None]
    )
    costs = state.candidate_stability_cost_batch[batch_indexes]
    masked_costs = torch.where(
        eligible,
        costs,
        torch.full_like(costs, float("inf")),
    )
    # Flatten machine-major to match the policy action convention.
    action_indexes = masked_costs.transpose(1, 2).flatten(1).argmin(dim=1)
    jobs = (action_indexes % environment.num_jobs).long()
    machines = (action_indexes / environment.num_jobs).long()
    operations = operation_steps[batch_indexes, jobs]
    return torch.stack((operations, machines, jobs), dim=1).t()


def evaluate_minimum_cost(environment, arrivals, budget):
    steps = 0
    while not bool(environment.done_batch.all().item()):
        environment.step(minimum_cost_actions(environment))
        steps += 1
        if steps > environment.num_opes + 1:
            raise RuntimeError("Minimum-cost evaluation exceeded operation-count limit.")
    feasible = bool(environment.validate_gantt()[0])
    rows = []
    for index, scenario in enumerate(arrivals):
        release_violations = 0
        for job in range(scenario.initial_jobs, scenario.num_jobs):
            operation = int(environment.num_ope_biases_batch[index, job].item())
            start = float(environment.schedules_batch[index, operation, 2].item())
            if start + 1e-5 < scenario.release_times[job]:
                release_violations += 1
        cost = float(environment.stability_cost_mean_batch[index].item())
        rows.append(
            {
                "makespan": float(environment.makespan_batch[index].item()),
                "stability_cost": cost,
                "budget_violation": max(0.0, cost - budget),
                "budget_met": cost <= budget + 1e-6,
                "release_violations": release_violations,
            }
        )
    return feasible, rows


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def select_sample(rows, budget, selection_mode):
    if not rows:
        raise ValueError("Cannot select from an empty candidate list.")
    if selection_mode == "makespan_only":
        return min(
            rows,
            key=lambda row: (float(row["makespan"]), float(row["stability_cost"])),
        )
    if selection_mode == "min_cost":
        return min(
            rows,
            key=lambda row: (float(row["stability_cost"]), float(row["makespan"])),
        )
    if selection_mode == "budget_first":
        budget_feasible = [
            row for row in rows
            if float(row["stability_cost"]) <= budget + 1e-6
        ]
        if budget_feasible:
            return min(
                budget_feasible,
                key=lambda row: (
                    float(row["makespan"]), float(row["stability_cost"])
                ),
            )
        return min(
            rows,
            key=lambda row: (
                float(row["stability_cost"]), float(row["makespan"])
            ),
        )
    raise ValueError("Unknown selection mode: {0}".format(selection_mode))


def main():
    args = parse_args()
    if args.offset < 0 or args.instances <= 0:
        raise SystemExit("offset must be non-negative and instances must be positive.")
    if args.sample_repeats < 0:
        raise SystemExit("sample-repeats must be non-negative.")
    if not math.isfinite(args.stability_budget) or args.stability_budget < 0:
        raise SystemExit("stability-budget must be finite and non-negative.")
    device = select_device(args.device)
    configure_device(device)
    setup_seed(args.seed)
    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_DIR / source_dir
    required = ("conditioned_initial.pt", "final_model.pt", "summary.json")
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise SystemExit("Source directory is missing: {0}".format(", ".join(missing)))

    paths = sorted((PROJECT_DIR / args.data_dir).glob("*.fjs"))[
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    legacy = load_policy(config, device, checkpoint_path, 0)
    reference_env = make_environment(
        paths,
        arrivals,
        [MachineBreakdownScenario.static(item.num_machines) for item in breakdowns],
        config,
        device,
    )
    references = reference_schedules(legacy, reference_env)
    initial = load_policy(
        config,
        device,
        source_dir / "conditioned_initial.pt",
        3,
        args.stability_context_mode,
    )
    final = load_policy(
        config,
        device,
        source_dir / "final_model.pt",
        3,
        args.stability_context_mode,
    )
    initial_env = make_environment(
        paths, arrivals, breakdowns, config, device, references, args.stability_budget
    )
    final_env = make_environment(
        paths, arrivals, breakdowns, config, device, references, args.stability_budget
    )
    minimum_cost_env = make_environment(
        paths, arrivals, breakdowns, config, device, references, args.stability_budget
    )
    initial_feasible, initial_rows = evaluate(
        initial, initial_env, arrivals, args.stability_budget,
        budget_repair=args.budget_repair,
    )
    final_feasible, final_rows = evaluate(
        final, final_env, arrivals, args.stability_budget,
        budget_repair=args.budget_repair,
    )
    minimum_cost_feasible, minimum_cost_rows = evaluate_minimum_cost(
        minimum_cost_env, arrivals, args.stability_budget
    )
    sampled_initial_rows = []
    sampled_final_rows = []
    sampled_feasible = True
    for repeat in range(args.sample_repeats):
        repeat_seed = args.seed + 1000 + repeat
        setup_seed(repeat_seed)
        sampled_initial_env = make_environment(
            paths, arrivals, breakdowns, config, device, references,
            args.stability_budget,
        )
        initial_repeat_feasible, initial_repeat_rows = evaluate(
            initial,
            sampled_initial_env,
            arrivals,
            args.stability_budget,
            sample=True,
            budget_repair=args.budget_repair,
        )
        setup_seed(repeat_seed)
        sampled_final_env = make_environment(
            paths, arrivals, breakdowns, config, device, references,
            args.stability_budget,
        )
        final_repeat_feasible, final_repeat_rows = evaluate(
            final,
            sampled_final_env,
            arrivals,
            args.stability_budget,
            sample=True,
            budget_repair=args.budget_repair,
        )
        sampled_feasible = (
            sampled_feasible
            and initial_repeat_feasible
            and final_repeat_feasible
        )
        for index, (before, after) in enumerate(
            zip(initial_repeat_rows, final_repeat_rows)
        ):
            sampled_initial_rows.append(before)
            sampled_final_rows.append(after)
        print(
            "sample_repeat={0}/{1} initial_cost={2:.6f} final_cost={3:.6f}".format(
                repeat + 1,
                args.sample_repeats,
                mean(initial_repeat_rows, "stability_cost"),
                mean(final_repeat_rows, "stability_cost"),
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "save" / (
        "constrained_pilot_eval_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    save_scenarios(arrivals, output_dir / "arrival_scenarios.json")
    save_breakdown_scenarios(breakdowns, output_dir / "breakdown_scenarios.json")
    details = []
    for path, before, after, minimum_cost in zip(
        paths, initial_rows, final_rows, minimum_cost_rows
    ):
        details.append(
            {
                "file_name": path.name,
                "initial_makespan": before["makespan"],
                "final_makespan": after["makespan"],
                "makespan_delta": after["makespan"] - before["makespan"],
                "initial_stability_cost": before["stability_cost"],
                "final_stability_cost": after["stability_cost"],
                "stability_cost_delta": after["stability_cost"] - before["stability_cost"],
                "initial_budget_met": before["budget_met"],
                "final_budget_met": after["budget_met"],
                "minimum_cost_makespan": minimum_cost["makespan"],
                "minimum_stability_cost": minimum_cost["stability_cost"],
                "minimum_cost_budget_met": minimum_cost["budget_met"],
                "release_violations": after["release_violations"],
            }
        )
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    if args.sample_repeats:
        sampled_details = []
        for row_index, (before, after) in enumerate(
            zip(sampled_initial_rows, sampled_final_rows)
        ):
            repeat = row_index // args.instances
            instance = row_index % args.instances
            sampled_details.append(
                {
                    "repeat": repeat,
                    "file_name": paths[instance].name,
                    "initial_makespan": before["makespan"],
                    "final_makespan": after["makespan"],
                    "makespan_delta": after["makespan"] - before["makespan"],
                    "initial_stability_cost": before["stability_cost"],
                    "final_stability_cost": after["stability_cost"],
                    "stability_cost_delta": (
                        after["stability_cost"] - before["stability_cost"]
                    ),
                    "initial_budget_met": before["budget_met"],
                    "final_budget_met": after["budget_met"],
                    "release_violations": after["release_violations"],
                }
            )
        with (output_dir / "sampled_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(sampled_details[0])
            )
            writer.writeheader()
            writer.writerows(sampled_details)
        selected_details = []
        selected_initial_rows = []
        selected_final_rows = []
        for instance, path in enumerate(paths):
            initial_candidates = sampled_initial_rows[
                instance :: args.instances
            ]
            final_candidates = sampled_final_rows[instance :: args.instances]
            selected_initial = select_sample(
                initial_candidates,
                args.stability_budget,
                args.selection_mode,
            )
            selected_final = select_sample(
                final_candidates,
                args.stability_budget,
                args.selection_mode,
            )
            selected_initial_rows.append(selected_initial)
            selected_final_rows.append(selected_final)
            selected_details.append(
                {
                    "file_name": path.name,
                    "initial_makespan": selected_initial["makespan"],
                    "final_makespan": selected_final["makespan"],
                    "makespan_delta": (
                        selected_final["makespan"]
                        - selected_initial["makespan"]
                    ),
                    "initial_stability_cost": selected_initial[
                        "stability_cost"
                    ],
                    "final_stability_cost": selected_final["stability_cost"],
                    "stability_cost_delta": (
                        selected_final["stability_cost"]
                        - selected_initial["stability_cost"]
                    ),
                    "initial_budget_met": selected_initial["budget_met"],
                    "final_budget_met": selected_final["budget_met"],
                    "stability_context_mode": args.stability_context_mode,
                    "selection_mode": args.selection_mode,
                }
            )
        with (output_dir / "selected_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(selected_details[0])
            )
            writer.writeheader()
            writer.writerows(selected_details)

    initial_cost = mean(initial_rows, "stability_cost")
    final_cost = mean(final_rows, "stability_cost")
    initial_makespan = mean(initial_rows, "makespan")
    final_makespan = mean(final_rows, "makespan")
    minimum_stability_cost = mean(minimum_cost_rows, "stability_cost")
    minimum_cost_makespan = mean(minimum_cost_rows, "makespan")
    summary = {
        "instances": args.instances,
        "offset": args.offset,
        "budget": args.stability_budget,
        "budget_repair": args.budget_repair,
        "stability_context_mode": args.stability_context_mode,
        "selection_mode": args.selection_mode,
        "initial_feasible": initial_feasible,
        "final_feasible": final_feasible,
        "minimum_cost_feasible": minimum_cost_feasible,
        "release_violations": sum(row["release_violations"] for row in final_rows),
        "initial_makespan_mean": initial_makespan,
        "final_makespan_mean": final_makespan,
        "makespan_delta": final_makespan - initial_makespan,
        "initial_stability_cost_mean": initial_cost,
        "final_stability_cost_mean": final_cost,
        "stability_cost_delta": final_cost - initial_cost,
        "initial_budget_met": sum(row["budget_met"] for row in initial_rows),
        "final_budget_met": sum(row["budget_met"] for row in final_rows),
        "minimum_cost_makespan_mean": minimum_cost_makespan,
        "minimum_stability_cost_mean": minimum_stability_cost,
        "minimum_cost_budget_met": sum(
            row["budget_met"] for row in minimum_cost_rows
        ),
        "budget_has_constructive_evidence": (
            minimum_stability_cost <= args.stability_budget + 1e-6
        ),
        "constraint_improved": final_cost < initial_cost,
        "passed": initial_feasible and final_feasible and minimum_cost_feasible
        and not any(row["release_violations"] for row in final_rows)
        and not any(row["release_violations"] for row in minimum_cost_rows),
    }
    if args.sample_repeats:
        sampled_initial_cost = mean(sampled_initial_rows, "stability_cost")
        sampled_final_cost = mean(sampled_final_rows, "stability_cost")
        sampled_initial_makespan = mean(sampled_initial_rows, "makespan")
        sampled_final_makespan = mean(sampled_final_rows, "makespan")
        summary.update(
            {
                "sample_repeats": args.sample_repeats,
                "sampled_rows": len(sampled_initial_rows),
                "sampled_feasible": sampled_feasible,
                "sampled_release_violations": sum(
                    row["release_violations"] for row in sampled_final_rows
                ),
                "sampled_initial_makespan_mean": sampled_initial_makespan,
                "sampled_final_makespan_mean": sampled_final_makespan,
                "sampled_makespan_delta": (
                    sampled_final_makespan - sampled_initial_makespan
                ),
                "sampled_initial_stability_cost_mean": sampled_initial_cost,
                "sampled_final_stability_cost_mean": sampled_final_cost,
                "sampled_stability_cost_delta": (
                    sampled_final_cost - sampled_initial_cost
                ),
                "sampled_initial_budget_met": sum(
                    row["budget_met"] for row in sampled_initial_rows
                ),
                "sampled_final_budget_met": sum(
                    row["budget_met"] for row in sampled_final_rows
                ),
                "sampled_constraint_improved": (
                    sampled_final_cost < sampled_initial_cost
                ),
                "selected_initial_makespan_mean": mean(
                    selected_initial_rows, "makespan"
                ),
                "selected_final_makespan_mean": mean(
                    selected_final_rows, "makespan"
                ),
                "selected_makespan_delta": (
                    mean(selected_final_rows, "makespan")
                    - mean(selected_initial_rows, "makespan")
                ),
                "selected_initial_stability_cost_mean": mean(
                    selected_initial_rows, "stability_cost"
                ),
                "selected_final_stability_cost_mean": mean(
                    selected_final_rows, "stability_cost"
                ),
                "selected_stability_cost_delta": (
                    mean(selected_final_rows, "stability_cost")
                    - mean(selected_initial_rows, "stability_cost")
                ),
                "selected_initial_budget_met": sum(
                    row["budget_met"] for row in selected_initial_rows
                ),
                "selected_final_budget_met": sum(
                    row["budget_met"] for row in selected_final_rows
                ),
                "selected_constraint_satisfied": (
                    mean(selected_final_rows, "stability_cost")
                    <= args.stability_budget + 1e-6
                ),
            }
        )
        summary["passed"] = (
            summary["passed"]
            and sampled_feasible
            and summary["sampled_release_violations"] == 0
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in summary.items():
        print("{0}={1}".format(key, value))
    print("results: {0}".format(output_dir))


if __name__ == "__main__":
    main()
