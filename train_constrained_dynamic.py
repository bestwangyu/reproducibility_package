#!/usr/bin/env python3
"""Smoke/pilot training for dynamic FJSP with a stability-cost budget.

The entry point deliberately lives beside the legacy trainers.  It constructs
reference schedules with the legacy policy, then trains the context-conditioned
policy in the same arrival/failure scenarios using Lagrangian PPO.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fjsp-drl-matplotlib")
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    from tests.gym_compat import install_gym_stub_if_needed

    install_gym_stub_if_needed()

import numpy as np
import torch

from constrained_ppo import ConstrainedMemory, ConstrainedPPO
from dynamic.arrivals import JobArrivalGenerator, save_scenarios
from dynamic.breakdowns import MachineBreakdownGenerator, save_breakdown_scenarios
from env.composite_dynamic_fjsp_env import CompositeDynamicFJSPEnv
from env.load_data import nums_detec
from PPO_model import HGNNScheduler, load_compatible_policy_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/base/reference_save_10_5.pt")
    parser.add_argument("--data-dir", default="data_dev/1005")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--initial-jobs", type=int, default=6)
    parser.add_argument("--mean-interarrival", type=float, default=8.0)
    parser.add_argument("--mean-time-between-failures", type=float, default=8.0)
    parser.add_argument("--mean-repair-time", type=float, default=2.0)
    parser.add_argument("--events-per-machine", type=int, default=1)
    parser.add_argument("--stability-budget", type=float, default=1.0)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--reward-coeff", type=float, default=1.0)
    parser.add_argument(
        "--cost-advantage-mode",
        choices=("critic", "batch_centered"),
        default="critic",
    )
    parser.add_argument(
        "--cost-signal",
        choices=("raw", "violation"),
        default="raw",
    )
    parser.add_argument(
        "--stability-context-mode",
        choices=("full", "zero"),
        default="full",
        help="Use the measured three-dimensional stability context or replace it with zeros.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=20260802)
    parser.add_argument("--scenario-seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Resume from a constrained_checkpoint.pt in the output directory.")
    parser.add_argument("--smoke-check", action="store_true")
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def configure_device(device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def model_parameters(config, device, context_dim, context_mode="full"):
    parameters = copy.deepcopy(config["model_paras"])
    parameters["device"] = device
    parameters["actor_in_dim"] = parameters["out_size_ma"] * 2 + parameters["out_size_ope"] * 2
    parameters["critic_in_dim"] = parameters["out_size_ma"] + parameters["out_size_ope"]
    parameters["stability_context_dim"] = context_dim
    parameters["stability_context_mode"] = context_mode
    return parameters


def make_environment(paths, arrivals, breakdowns, config, device, references=None, budget=None):
    first_lines = paths[0].read_text(encoding="utf-8").splitlines(keepends=True)
    num_jobs, num_machines, _ = nums_detec(first_lines)
    parameters = copy.deepcopy(config["env_paras"])
    parameters.update({
        "num_jobs": num_jobs,
        "num_mas": num_machines,
        "batch_size": len(paths),
        "device": device,
        "dynamic_reward_mode": "raw",
    })
    return CompositeDynamicFJSPEnv(
        case=[str(path) for path in paths],
        env_paras=parameters,
        data_source="file",
        arrival_scenario=arrivals,
        breakdown_scenario=breakdowns,
        reference_schedules=references,
        stability_budget=budget,
    )


def build_scenarios(paths, config, args, iteration):
    arrival_generator = JobArrivalGenerator(
        seed=args.scenario_seed + iteration,
        mean_interarrival=args.mean_interarrival,
    )
    breakdown_generator = MachineBreakdownGenerator(
        seed=args.scenario_seed + 1000 + iteration,
        mean_time_between_failures=args.mean_time_between_failures,
        mean_repair_time=args.mean_repair_time,
    )
    arrivals = [
        arrival_generator.generate(
            num_jobs=config["env_paras"]["num_jobs"],
            initial_jobs=args.initial_jobs,
            scenario_id="arrival:iter_{0}:{1}".format(iteration, path.stem),
            random_stream_id=path.stem,
        )
        for path in paths
    ]
    breakdowns = [
        breakdown_generator.generate(
            num_machines=config["env_paras"]["num_mas"],
            events_per_machine=args.events_per_machine,
            scenario_id="breakdown:iter_{0}:{1}".format(iteration, path.stem),
            random_stream_id=path.stem,
        )
        for path in paths
    ]
    return arrivals, breakdowns


def reference_schedules(policy, environment):
    memory = ConstrainedMemory()
    steps = 0
    while not bool(environment.done_batch.all().item()):
        with torch.no_grad():
            action = policy.act(
                environment.state, memory, environment.done_batch,
                flag_sample=False, flag_train=False,
            )
        environment.step(action)
        steps += 1
        if steps > environment.num_opes + 1:
            raise RuntimeError("Reference episode exceeded operation-count limit.")
    return environment.schedules_batch.detach().clone()


def run_episode(model, environment, cost_signal="raw"):
    if cost_signal not in ("raw", "violation"):
        raise ValueError("cost_signal must be raw or violation")
    memory = ConstrainedMemory()
    steps = 0
    if cost_signal == "violation":
        previous_cost = environment.stability_violation_batch.detach().clone()
    else:
        previous_cost = environment.stability_cost_sum_batch.detach().clone()
    invalid_actions = 0
    while not bool(environment.done_batch.all().item()):
        with torch.no_grad():
            actions = model.policy_old.act(
                environment.state, memory, environment.done_batch,
                flag_sample=True, flag_train=True,
            )
        try:
            _, rewards, dones = environment.step(actions)
        except (IndexError, ValueError, RuntimeError):
            invalid_actions += 1
            raise
        if cost_signal == "violation":
            current_cost = environment.stability_violation_batch.detach().clone()
            cost = current_cost - previous_cost
        else:
            current_cost = environment.stability_cost_sum_batch.detach().clone()
            cost = (current_cost - previous_cost) / environment.nums_opes.to(
                dtype=current_cost.dtype
            ).clamp_min(1.0)
        cost = torch.clamp(cost, min=0.0)
        previous_cost = current_cost
        memory.rewards.append(rewards.detach().clone())
        memory.costs.append(cost.detach().clone())
        memory.is_terminals.append(dones.detach().clone())
        steps += 1
        if steps > environment.num_opes + 1:
            raise RuntimeError("Constrained episode exceeded operation-count limit.")
    feasible = bool(environment.validate_gantt()[0])
    release_violations = 0
    for index, scenario in enumerate(environment.arrival_scenarios):
        for job in range(scenario.initial_jobs, scenario.num_jobs):
            operation = int(environment.num_ope_biases_batch[index, job].item())
            start = float(environment.schedules_batch[index, operation, 2].item())
            if start + 1e-5 < scenario.release_times[job]:
                release_violations += 1
    return memory, {
        "steps": steps,
        "feasible": feasible,
        "release_violations": release_violations,
        "invalid_actions": invalid_actions,
        "makespan_mean": float(environment.makespan_batch.mean().item()),
        "stability_cost_mean": float(environment.stability_cost_mean_batch.mean().item()),
        "stability_violation_mean": float(environment.stability_violation_batch.mean().item()),
    }


def write_row(path, row):
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.instances <= 0 or args.iterations <= 0:
        raise SystemExit("instances and iterations must be positive")
    if args.batch_size is None:
        args.batch_size = args.instances
    if args.batch_size != args.instances:
        raise SystemExit("For this pilot entry point, batch-size must equal instances.")
    if args.resume and not args.output_dir:
        raise SystemExit("--output-dir is required when --resume is used.")
    if args.stability_budget < 0 or not math.isfinite(args.stability_budget):
        raise SystemExit("stability-budget must be finite and non-negative")
    for name, value in (
        ("lambda-init", args.lambda_init),
        ("lambda-lr", args.lambda_lr),
        ("lambda-max", args.lambda_max),
    ):
        if value < 0 or not math.isfinite(value):
            raise SystemExit("{0} must be finite and non-negative".format(name))
    if args.lambda_init > args.lambda_max:
        raise SystemExit("lambda-init must not exceed lambda-max")
    if args.reward_coeff < 0 or not math.isfinite(args.reward_coeff):
        raise SystemExit("reward-coeff must be finite and non-negative")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise SystemExit("lr must be finite and positive")
    device = select_device(args.device)
    configure_device(device)
    setup_seed(args.train_seed)
    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    num_jobs = int(config["env_paras"]["num_jobs"])
    if not 1 <= args.initial_jobs < num_jobs:
        raise SystemExit("initial-jobs must be between 1 and num_jobs-1")
    paths = sorted((PROJECT_DIR / args.data_dir).glob("*.fjs"))[: args.instances]
    if len(paths) != args.instances:
        raise SystemExit("Requested {0} instances, found {1}.".format(args.instances, len(paths)))

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "save" / (
        "constrained_training_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    metadata = vars(args).copy()
    metadata.update({"device": str(device), "checkpoint": args.checkpoint})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    legacy = HGNNScheduler(model_parameters(config, device, 0)).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    legacy.load_state_dict(checkpoint)
    legacy.eval()

    train_parameters = copy.deepcopy(config["train_paras"])
    train_parameters.update({
        "minibatch_size": min(int(config["train_paras"].get("minibatch_size", 512)), 64),
        "lr": args.lr,
        "stability_budget": args.stability_budget,
        "lambda_init": args.lambda_init,
        "lambda_lr": args.lambda_lr,
        "lambda_max": args.lambda_max,
        "A_coeff": args.reward_coeff,
        "cost_advantage_mode": args.cost_advantage_mode,
    })
    constrained = ConstrainedPPO(
        model_parameters(
            config, device, 3, args.stability_context_mode
        ),
        train_parameters,
        args.instances,
    )
    migration = load_compatible_policy_checkpoint(constrained.policy, checkpoint)
    constrained.policy_old.load_state_dict(constrained.policy.state_dict())
    start_iteration = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_DIR / resume_path
        if not resume_path.is_file():
            raise SystemExit("Resume checkpoint not found: {0}".format(resume_path))
        resume = torch.load(str(resume_path), map_location=device)
        if not isinstance(resume, dict) or "policy" not in resume or "cost_critic" not in resume:
            raise SystemExit("Resume checkpoint must contain policy and cost_critic.")
        constrained.policy.load_state_dict(resume["policy"])
        constrained.policy_old.load_state_dict(resume["policy"])
        constrained.cost_critic.load_state_dict(resume["cost_critic"])
        constrained.lambda_value = float(resume.get("lambda", 0.0))
        completed = int(resume.get("iteration", 0))
        start_iteration = completed + 1
        if start_iteration > args.iterations:
            raise SystemExit("Resume iteration is already beyond --iterations.")
        migration = {"resumed": True, "resume_iteration": completed}
    torch.save(constrained.policy_old.state_dict(), output_dir / "conditioned_initial.pt")
    (output_dir / "checkpoint_migration.json").write_text(
        json.dumps(migration, indent=2, default=list) + "\n", encoding="utf-8"
    )

    results_path = output_dir / "training_results.csv"
    rows = []
    if args.resume and (output_dir / "training_results.csv").is_file():
        with (output_dir / "training_results.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    passed = True
    for iteration in range(start_iteration, args.iterations + 1):
        iteration_start = time.perf_counter()
        paths_for_iteration = paths
        arrivals, breakdowns = build_scenarios(paths_for_iteration, config, args, iteration)
        save_scenarios(arrivals, output_dir / "arrival_scenarios_{0:04d}.json".format(iteration))
        save_breakdown_scenarios(
            breakdowns, output_dir / "breakdown_scenarios_{0:04d}.json".format(iteration)
        )
        reference_environment = make_environment(
            paths_for_iteration, arrivals,
            [item.static(item.num_machines, seed=item.seed, scenario_id="static") for item in breakdowns],
            config, device,
        )
        references = reference_schedules(legacy, reference_environment)
        environment = make_environment(
            paths_for_iteration, arrivals, breakdowns, config, device,
            references=references, budget=args.stability_budget,
        )
        memory, diagnostics = run_episode(
            constrained, environment, cost_signal=args.cost_signal
        )
        update = constrained.update(memory, {"device": device}, train_parameters)
        lambda_value = constrained.update_lambda(diagnostics["stability_cost_mean"])
        row = {
            "iteration": iteration,
            **diagnostics,
            **update,
            "lambda": lambda_value,
            "budget": args.stability_budget,
            "finite": all(math.isfinite(float(value)) for value in (
                diagnostics["makespan_mean"], diagnostics["stability_cost_mean"],
                diagnostics["stability_violation_mean"], update["loss"],
                update["cost_value_loss"], lambda_value,
            )),
            "runtime_seconds": time.perf_counter() - iteration_start,
        }
        rows.append(row)
        write_row(results_path, row)
        torch.save(
            {
                "policy": constrained.policy_old.state_dict(),
                "cost_critic": constrained.cost_critic.state_dict(),
                "lambda": constrained.lambda_value,
                "budget": args.stability_budget,
                "iteration": iteration,
            },
            output_dir / "checkpoint_latest.pt",
        )
        memory.clear_memory()
        passed = passed and row["finite"] and diagnostics["feasible"] and diagnostics["release_violations"] == 0
        print(
            "iteration={0}/{1} makespan={2:.3f} cost={3:.6f} budget={4:.6f} "
            "lambda={5:.6f} loss={6:.6f} feasible={7} release_violations={8}".format(
                iteration, args.iterations, diagnostics["makespan_mean"],
                diagnostics["stability_cost_mean"], args.stability_budget,
                lambda_value, update["loss"], diagnostics["feasible"],
                diagnostics["release_violations"],
            )
        )

    final_model = output_dir / "final_model.pt"
    torch.save(constrained.policy_old.state_dict(), final_model)
    torch.save(
        {
            "policy": constrained.policy_old.state_dict(),
            "cost_critic": constrained.cost_critic.state_dict(),
            "lambda": constrained.lambda_value,
                "budget": args.stability_budget,
                "iteration": args.iterations,
            },
        output_dir / "constrained_checkpoint.pt",
    )
    def row_bool(row, key):
        value = row.get(key, False)
        return value if isinstance(value, bool) else str(value).lower() == "true"

    def row_int(row, key):
        return int(float(row.get(key, 0)))

    summary = {
        "iterations": args.iterations,
        "rows": len(rows),
        "finite_rows": sum(row_bool(row, "finite") for row in rows),
        "feasible_rows": sum(row_bool(row, "feasible") for row in rows),
        "release_violations": sum(row_int(row, "release_violations") for row in rows),
        "invalid_actions": sum(row_int(row, "invalid_actions") for row in rows),
        "budget": args.stability_budget,
        "lambda_final": constrained.lambda_value,
        "passed": passed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "completion.json").write_text(
        json.dumps({"completed": True, "iterations": args.iterations, "passed": passed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rows={0}".format(summary["rows"]))
    print("finite_rows={0}".format(summary["finite_rows"]))
    print("feasible_rows={0}".format(summary["feasible_rows"]))
    print("release_violations={0}".format(summary["release_violations"]))
    print("invalid_actions={0}".format(summary["invalid_actions"]))
    print("lambda_final={0:.6f}".format(summary["lambda_final"]))
    print("passed={0}".format(summary["passed"]))
    if args.smoke_check and not passed:
        raise SystemExit("FAIL constrained training smoke")
    print("results: {0}".format(output_dir))


if __name__ == "__main__":
    main()
