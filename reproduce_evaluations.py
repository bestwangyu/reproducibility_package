#!/usr/bin/env python3
"""Portable evaluation entry points; no training or model selection on test data."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from verify_reproduction_inputs import ROOT, TRAIN_SEEDS, verify_inputs

EVAL_SEEDS = (20260825, 20260826, 20260827)
MAIN_PREFIX = "finalenv_nominal_holdout"
BASE = "checkpoints/base/reference_save_10_5.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "ablations", "independent"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", "outputs"))
    parser.add_argument("--models", choices=("frozen", "retrained"), default=os.getenv("MODELS", "frozen"))
    parser.add_argument("--smoke", action="store_true", default=os.getenv("SMOKE") == "1")
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN") == "1")
    args = parser.parse_args()
    if args.stage == "independent" and args.models != "frozen":
        parser.error("The frozen independent-test reproduction must use --models frozen.")
    return args


def command(script, *args):
    return [sys.executable, "-u", script] + [str(x) for x in args]


def build_plan(args):
    root = Path(args.output_root).resolve()
    if root == ROOT or root == ROOT / "results" or ROOT / "results" in root.parents:
        raise ValueError("Use a generated-output directory, never the frozen results/ directory.")
    if args.smoke:
        root = root / "smoke"
    train_seeds = TRAIN_SEEDS[:1] if args.smoke else TRAIN_SEEDS
    eval_seeds = EVAL_SEEDS[:1] if args.smoke else EVAL_SEEDS
    independent = args.stage == "independent"
    data = "data_test/1005" if independent else "data_dev/1005"
    instances = 4 if args.smoke else (100 if independent else 80)
    offset = 0 if independent else 20
    repeats = 2 if args.smoke else 20
    common = ["--device", args.device, "--data-dir", data,
              "--offset", offset, "--instances", instances,
              "--initial-jobs", 6, "--mean-interarrival", 8,
              "--mean-time-between-failures", 8, "--mean-repair-time", 2,
              "--events-per-machine", 1, "--stability-budget", 1.7]
    jobs = []

    def add(script, path, extra, fields, files):
        cmd = command(script, *common, *extra, "--output-dir", path)
        jobs.append({"command": cmd, "directory": path,
                     "fields": dict(fields, passed=True),
                     "files": ["summary.json", "arrival_scenarios.json", "breakdown_scenarios.json"] + files})

    modes = ("reward_only", "zero_context") if args.stage == "ablations" else ("main",)
    for mode in modes:
        context = "zero" if mode == "zero_context" else "full"
        prefix = ("p0t_independent_main" if independent else MAIN_PREFIX) if mode == "main" else "finalenv_{0}_holdout".format(mode)
        for seed in train_seeds:
            if args.models == "frozen":
                source = ["--policy-checkpoint", "checkpoints/{0}/{0}_seed_{1}.pt".format(mode, seed),
                          "--initial-checkpoint", "checkpoints/main/main_seed_{0}_conditioned_initial.pt".format(seed)]
            else:
                source = ["--source-dir", root / "{0}_seed_{1}".format(mode, seed)]
            for eval_seed in eval_seeds:
                path = root / "{0}_train{1}_eval_{2}".format(prefix, seed, eval_seed)
                add("evaluate_constrained_pilot.py", path,
                    source + ["--checkpoint", BASE, "--seed", eval_seed,
                              "--sample-repeats", repeats, "--stability-context-mode", context,
                              "--selection-mode", "budget_first"],
                    {"instances": instances, "offset": offset, "sample_repeats": repeats,
                     "sampled_rows": instances * repeats, "stability_context_mode": context,
                     "selection_mode": "budget_first", "release_violations": 0,
                     "sampled_release_violations": 0},
                    ["results.csv", "sampled_results.csv", "selected_results.csv"])

    if independent:
        for seed in train_seeds:
            for eval_seed in eval_seeds:
                path = root / "p0t_independent_static_drl_train{0}_eval_{1}".format(seed, eval_seed)
                add("evaluate_static_dynamic_baseline.py", path,
                    ["--policy-checkpoint", "checkpoints/static_drl/static_drl_seed_{0}.pt".format(seed),
                     "--reference-checkpoint", BASE, "--policy-seed", seed, "--seed", eval_seed],
                    {"instances": instances, "offset": 0, "policy_seed": seed,
                     "eval_seed": eval_seed, "release_violations": 0}, ["results.csv"])
        for eval_seed in eval_seeds:
            path = root / "p0t_independent_rules_eval_{0}".format(eval_seed)
            add("evaluate_dynamic_dispatching_rules.py", path,
                ["--reference-checkpoint", BASE, "--seed", eval_seed],
                {"instances_per_rule": instances, "result_rows": instances * 4,
                 "offset": 0, "eval_seed": eval_seed, "release_violations": 0},
                ["results.csv", "rule_definitions.json"])

    summaries = []
    if not args.smoke:
        if args.stage == "development":
            summaries.append(command("summarize_finalenv_main_holdout.py", "--save-dir", root,
                                     "--directory-prefix", MAIN_PREFIX, "--output-dir", root / "nominal_summary"))
        elif args.stage == "ablations":
            for mode in modes:
                prefix = "finalenv_{0}_holdout".format(mode)
                summaries.append(command("summarize_finalenv_key_ablation.py", "--save-dir", root,
                    "--directory-prefix", prefix, "--reference-prefix", MAIN_PREFIX,
                    "--expected-context-mode", "zero" if mode == "zero_context" else "full",
                    "--output-dir", root / (prefix + "_3x3_summary")))
        else:
            summaries.append(command("summarize_p0t_frozen_test.py", "--save-dir", root,
                "--protocol", "protocol/independent_test_v1.json",
                "--output-dir", root / "p0t_frozen_independent_test_summary"))
    return root, jobs, summaries


def check_output(job):
    directory = job["directory"]
    missing = [name for name in job["files"] if not (directory / name).is_file()]
    if missing:
        raise ValueError("Incomplete evaluation: {0}: {1}".format(directory, missing))
    summary = json.loads((directory / "summary.json").read_text())
    for key, expected in job["fields"].items():
        if summary.get(key) != expected:
            raise ValueError("Unexpected {0} in {1}: {2!r} (expected {3!r})".format(key, directory, summary.get(key), expected))


def run_logged(cmd, log):
    print(shlex.join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise RuntimeError("Command failed with exit code {0}; see {1}".format(code, log))


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    verify_inputs()
    root, jobs, summaries = build_plan(args)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        for cmd in summaries:
            print(shlex.join(cmd))
        print("DRY RUN: stage={0}, evaluation_cells={1}, summary_commands={2}; no output written".format(args.stage, len(jobs), len(summaries)))
        return

    # Validate dependencies before launching any expensive evaluation.
    if args.stage == "ablations" and not args.smoke:
        for seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                directory = root / "{0}_train{1}_eval_{2}".format(MAIN_PREFIX, seed, eval_seed)
                for name in ("selected_results.csv", "arrival_scenarios.json", "breakdown_scenarios.json", "summary.json"):
                    if not (directory / name).is_file():
                        raise ValueError("Run development evaluation first; missing " + str(directory / name))
    for job in jobs:
        cmd = job["command"]
        for flag in ("--policy-checkpoint", "--initial-checkpoint", "--checkpoint", "--reference-checkpoint"):
            if flag in cmd and not (ROOT / cmd[cmd.index(flag) + 1]).is_file():
                raise ValueError("Missing checkpoint for " + flag)
        if "--source-dir" in cmd:
            source = Path(cmd[cmd.index("--source-dir") + 1])
            for name in ("summary.json", "conditioned_initial.pt", "final_model.pt"):
                if not (source / name).is_file():
                    raise ValueError("Missing retrained input: " + str(source / name))
        if job["directory"].exists():
            check_output(job)
            receipt = job["directory"] / "reproduction_request.json"
            if not receipt.is_file() or json.loads(receipt.read_text())["command"] != cmd:
                raise ValueError("Existing result has no matching reproduction request; use a fresh OUTPUT_ROOT: " + str(job["directory"]))
    for cmd in summaries:
        dest = Path(cmd[cmd.index("--output-dir") + 1])
        if dest.exists():
            raise ValueError("Summary already exists; use a fresh OUTPUT_ROOT: " + str(dest))
    for job in jobs:
        if job["directory"].exists():
            print("skip verified evaluation: " + str(job["directory"]), flush=True)
            continue
        run_logged(job["command"], root / "logs" / (job["directory"].name + ".log"))
        check_output(job)
        receipt = {"command": job["command"], "stage": args.stage,
                   "smoke": args.smoke, "models": args.models}
        (job["directory"] / "reproduction_request.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for cmd in summaries:
        dest = Path(cmd[cmd.index("--output-dir") + 1])
        run_logged(cmd, root / "logs" / (dest.name + ".log"))
    print("PASS {0} {1}: evaluation_cells={2}".format(args.stage, "smoke (not manuscript results)" if args.smoke else "reproduction", len(jobs)))


if __name__ == "__main__":
    main()
