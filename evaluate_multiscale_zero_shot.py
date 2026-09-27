#!/usr/bin/env python3
"""Zero-shot evaluation of the frozen main policy on a different FJSP size."""

import argparse
import csv
import json
import math
from pathlib import Path

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    from tests.gym_compat import install_gym_stub_if_needed

    install_gym_stub_if_needed()

from dynamic.arrivals import JobArrivalGenerator, save_scenarios
from dynamic.breakdowns import MachineBreakdownGenerator, MachineBreakdownScenario, save_breakdown_scenarios
from env.load_data import nums_detec
from evaluate_constrained_pilot import evaluate, load_policy, select_sample
from train_constrained_dynamic import PROJECT_DIR, configure_device, make_environment, reference_schedules, select_device, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/base/reference_save_10_5.pt")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--instances", type=int, default=80)
    parser.add_argument("--initial-job-ratio", type=float, default=0.6)
    parser.add_argument("--mean-interarrival", type=float, default=8.0)
    parser.add_argument("--mean-time-between-failures", type=float, default=8.0)
    parser.add_argument("--mean-repair-time", type=float, default=2.0)
    parser.add_argument("--events-per-machine", type=int, default=1)
    parser.add_argument("--stability-budget", type=float, default=1.7)
    parser.add_argument("--sample-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def dimensions(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    jobs, machines, _ = nums_detec(lines)
    return int(jobs), int(machines)


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


FAILURE_KEYS = (
    "unscheduled",
    "processing_time_mismatch",
    "machine_overlap",
    "job_precedence_overlap",
    "start_during_repair",
)
SELECTED_FIELDS = (
    "file_name",
    "initial_repeat",
    "final_repeat",
    "initial_makespan",
    "final_makespan",
    "makespan_delta",
    "initial_stability_cost",
    "final_stability_cost",
    "stability_cost_delta",
    "initial_budget_met",
    "final_budget_met",
)


def classify_candidate_feasibility(environment, tolerance=1e-5):
    """Return validator-compatible failure counts for every batch instance."""
    results = []
    for batch_index in range(environment.batch_size):
        operation_count = int(environment.nums_opes[batch_index])
        schedules = environment.schedules_batch[batch_index]
        failures = {key: 0 for key in FAILURE_KEYS}
        machine_rows = [[] for _ in range(environment.num_mas)]

        for operation_index in range(operation_count):
            schedule = schedules[operation_index]
            if int(schedule[0].item()) != 1:
                failures["unscheduled"] += 1
                continue
            machine_index = int(schedule[1].item())
            start_time = float(schedule[2].item())
            end_time = float(schedule[3].item())
            expected_duration = environment.expected_operation_duration(
                batch_index, operation_index, machine_index
            )
            if abs((end_time - start_time) - expected_duration) > tolerance:
                failures["processing_time_mismatch"] += 1
            machine_rows[machine_index].append((start_time, end_time))

            scenario = environment.breakdown_scenarios[batch_index]
            for event in scenario.events:
                if event.machine_id != machine_index:
                    continue
                if (
                    start_time + tolerance >= event.start_time
                    and start_time < event.repair_end_time - tolerance
                ):
                    failures["start_during_repair"] += 1

        for rows in machine_rows:
            rows.sort(key=lambda row: row[0])
            for left, right in zip(rows, rows[1:]):
                if left[1] > right[0] + tolerance:
                    failures["machine_overlap"] += 1

        operation_counts = environment.nums_ope_batch[batch_index]
        operation_biases = environment.num_ope_biases_batch[batch_index]
        for job_index in range(environment.num_jobs):
            first_operation = int(operation_biases[job_index].item())
            for relative_index in range(int(operation_counts[job_index].item()) - 1):
                left = schedules[first_operation + relative_index]
                right = schedules[first_operation + relative_index + 1]
                if int(left[0].item()) != 1 or int(right[0].item()) != 1:
                    continue
                if float(left[3].item()) > float(right[2].item()) + tolerance:
                    failures["job_precedence_overlap"] += 1

        failures["feasible"] = not any(failures[key] for key in FAILURE_KEYS)
        results.append(failures)
    return results


def feasibility_totals(rows, prefix):
    return {
        key: sum(int(row["{0}_{1}".format(prefix, key)]) for row in rows)
        for key in FAILURE_KEYS
    }


def main():
    args = parse_args()
    if args.instances <= 0 or args.offset < 0 or args.sample_repeats <= 0:
        raise SystemExit("Invalid offset, instances, or sample repeats")
    if not 0.0 < args.initial_job_ratio < 1.0:
        raise SystemExit("initial-job-ratio must be between zero and one")
    device = select_device(args.device); configure_device(device); setup_seed(args.seed)
    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    source_dir = Path(args.source_dir); source_dir = source_dir if source_dir.is_absolute() else PROJECT_DIR / source_dir
    checkpoint = Path(args.checkpoint); checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_DIR / checkpoint
    for path in (source_dir / "conditioned_initial.pt", source_dir / "final_model.pt", checkpoint):
        if not path.is_file(): raise SystemExit("Missing model: {0}".format(path))
    data_dir = Path(args.data_dir); data_dir = data_dir if data_dir.is_absolute() else PROJECT_DIR / data_dir
    paths = sorted(data_dir.glob("*.fjs"))[args.offset:args.offset + args.instances]
    if len(paths) != args.instances: raise SystemExit("Not enough evaluation instances")
    sizes = {dimensions(path) for path in paths}
    if len(sizes) != 1: raise SystemExit("All evaluation instances must have one common size")
    num_jobs, num_machines = next(iter(sizes))
    initial_jobs = max(1, min(num_jobs - 1, int(round(num_jobs * args.initial_job_ratio))))

    arrival_generator = JobArrivalGenerator(args.seed, args.mean_interarrival)
    breakdown_generator = MachineBreakdownGenerator(args.seed + 1, args.mean_time_between_failures, args.mean_repair_time)
    arrivals = [arrival_generator.generate(num_jobs, initial_jobs, scenario_id="multiscale-arrival:{0}".format(path.stem), random_stream_id=path.stem) for path in paths]
    breakdowns = [breakdown_generator.generate(num_machines, args.events_per_machine, scenario_id="multiscale-breakdown:{0}".format(path.stem), random_stream_id=path.stem) for path in paths]
    legacy = load_policy(config, device, checkpoint, 0)
    reference_env = make_environment(paths, arrivals, [MachineBreakdownScenario.static(num_machines) for _ in paths], config, device)
    references = reference_schedules(legacy, reference_env)
    initial = load_policy(config, device, source_dir / "conditioned_initial.pt", 3, "full")
    final = load_policy(config, device, source_dir / "final_model.pt", 3, "full")

    sampled = []
    initial_validator_failed_repeats = 0
    final_validator_failed_repeats = 0
    initial_validator_diagnostic_mismatches = 0
    final_validator_diagnostic_mismatches = 0
    for repeat in range(args.sample_repeats):
        repeat_seed = args.seed + 1000 + repeat
        setup_seed(repeat_seed)
        initial_env = make_environment(paths, arrivals, breakdowns, config, device, references, args.stability_budget)
        initial_feasible, before = evaluate(initial, initial_env, arrivals, args.stability_budget, sample=True)
        initial_diagnostics = classify_candidate_feasibility(initial_env)
        setup_seed(repeat_seed)
        final_env = make_environment(paths, arrivals, breakdowns, config, device, references, args.stability_budget)
        final_feasible, after = evaluate(final, final_env, arrivals, args.stability_budget, sample=True)
        final_diagnostics = classify_candidate_feasibility(final_env)
        initial_validator_failed_repeats += int(not initial_feasible)
        final_validator_failed_repeats += int(not final_feasible)
        initial_validator_diagnostic_mismatches += int(
            initial_feasible != all(row["feasible"] for row in initial_diagnostics)
        )
        final_validator_diagnostic_mismatches += int(
            final_feasible != all(row["feasible"] for row in final_diagnostics)
        )
        for path, left, right, left_diag, right_diag in zip(
            paths, before, after, initial_diagnostics, final_diagnostics
        ):
            row = {"repeat": repeat, "file_name": path.name, "initial_makespan": left["makespan"], "final_makespan": right["makespan"], "initial_stability_cost": left["stability_cost"], "final_stability_cost": right["stability_cost"], "initial_budget_met": left["budget_met"], "final_budget_met": right["budget_met"], "initial_release_violations": left["release_violations"], "final_release_violations": right["release_violations"], "release_violations": right["release_violations"]}
            for key in FAILURE_KEYS + ("feasible",):
                row["initial_{0}".format(key)] = left_diag[key]
                row["final_{0}".format(key)] = right_diag[key]
            sampled.append(row)
        print(
            "sample_repeat={0}/{1} initial_cost={2:.6f} final_cost={3:.6f} "
            "initial_feasible={4}/{5} final_feasible={6}/{5}".format(
                repeat + 1,
                args.sample_repeats,
                mean(before, "stability_cost"),
                mean(after, "stability_cost"),
                sum(row["feasible"] for row in initial_diagnostics),
                args.instances,
                sum(row["feasible"] for row in final_diagnostics),
            ),
            flush=True,
        )

    selected = []
    instances_without_initial_feasible_candidate = []
    instances_without_final_feasible_candidate = []
    for index, path in enumerate(paths):
        candidates = sampled[index::args.instances]
        initial_candidates = [
            {"repeat": row["repeat"], "makespan": row["initial_makespan"], "stability_cost": row["initial_stability_cost"], "budget_met": row["initial_budget_met"]}
            for row in candidates if row["initial_feasible"]
        ]
        final_candidates = [
            {"repeat": row["repeat"], "makespan": row["final_makespan"], "stability_cost": row["final_stability_cost"], "budget_met": row["final_budget_met"]}
            for row in candidates if row["final_feasible"]
        ]
        if not initial_candidates:
            instances_without_initial_feasible_candidate.append(path.name)
        if not final_candidates:
            instances_without_final_feasible_candidate.append(path.name)
        if not initial_candidates or not final_candidates:
            continue
        initial_choice = select_sample(initial_candidates, args.stability_budget, "budget_first")
        final_choice = select_sample(final_candidates, args.stability_budget, "budget_first")
        selected.append({"file_name": path.name, "initial_repeat": initial_choice["repeat"], "final_repeat": final_choice["repeat"], "initial_makespan": initial_choice["makespan"], "final_makespan": final_choice["makespan"], "makespan_delta": final_choice["makespan"] - initial_choice["makespan"], "initial_stability_cost": initial_choice["stability_cost"], "final_stability_cost": final_choice["stability_cost"], "stability_cost_delta": final_choice["stability_cost"] - initial_choice["stability_cost"], "initial_budget_met": initial_choice["budget_met"], "final_budget_met": final_choice["budget_met"]})

    output = Path(args.output_dir); output = output if output.is_absolute() else PROJECT_DIR / output
    output.mkdir(parents=True, exist_ok=False)
    save_scenarios(arrivals, output / "arrival_scenarios.json"); save_breakdown_scenarios(breakdowns, output / "breakdown_scenarios.json")
    for name, data, fieldnames in (
        ("sampled_results.csv", sampled, list(sampled[0])),
        ("selected_results.csv", selected, SELECTED_FIELDS),
    ):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(data)
    release_violations = sum(int(row["release_violations"]) for row in sampled)
    initial_feasible_candidates = sum(bool(row["initial_feasible"]) for row in sampled)
    final_feasible_candidates = sum(bool(row["final_feasible"]) for row in sampled)
    summary = {"evaluation_mode": "P1_multiscale_zero_shot_v2", "data_role": "development_validation", "source_size": "1005", "target_jobs": num_jobs, "target_machines": num_machines, "initial_jobs": initial_jobs, "initial_job_ratio": args.initial_job_ratio, "instances": args.instances, "offset": args.offset, "sample_repeats": args.sample_repeats, "sampled_rows": len(sampled), "selected_instances": len(selected), "selection_mode": "feasible_then_budget_first", "stability_budget": args.stability_budget, "initial_feasible_candidates": initial_feasible_candidates, "final_feasible_candidates": final_feasible_candidates, "initial_feasible_candidate_rate": initial_feasible_candidates / len(sampled), "final_feasible_candidate_rate": final_feasible_candidates / len(sampled), "initial_validator_failed_repeats": initial_validator_failed_repeats, "final_validator_failed_repeats": final_validator_failed_repeats, "initial_validator_diagnostic_mismatches": initial_validator_diagnostic_mismatches, "final_validator_diagnostic_mismatches": final_validator_diagnostic_mismatches, "initial_failure_totals": feasibility_totals(sampled, "initial"), "final_failure_totals": feasibility_totals(sampled, "final"), "instances_without_initial_feasible_candidate": instances_without_initial_feasible_candidate, "instances_without_final_feasible_candidate": instances_without_final_feasible_candidate, "release_violations": release_violations}
    if selected:
        summary.update({"selected_initial_makespan_mean": mean(selected, "initial_makespan"), "selected_final_makespan_mean": mean(selected, "final_makespan"), "selected_makespan_delta": mean(selected, "makespan_delta"), "selected_initial_stability_cost_mean": mean(selected, "initial_stability_cost"), "selected_final_stability_cost_mean": mean(selected, "final_stability_cost"), "selected_stability_cost_delta": mean(selected, "stability_cost_delta"), "selected_initial_budget_met_rate": sum(bool(row["initial_budget_met"]) for row in selected) / len(selected), "selected_final_budget_met_rate": sum(bool(row["final_budget_met"]) for row in selected) / len(selected)})
    summary["passed"] = release_violations == 0 and len(sampled) == args.instances * args.sample_repeats and len(selected) == args.instances and initial_validator_diagnostic_mismatches == 0 and final_validator_diagnostic_mismatches == 0
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary.items(): print("{0}={1}".format(key, value))
    print("results: {0}".format(output))
    if not summary["passed"]: raise SystemExit("Multiscale evaluation failed")


if __name__ == "__main__": main()
