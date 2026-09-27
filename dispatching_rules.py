"""Deterministic dispatching rules for dynamic FJSP evaluation."""

import torch


RULES = ("FIFO", "SPT", "MWKR", "MOR")

RULE_DEFINITIONS = {
    "FIFO": (
        "Select the eligible job with the earliest release time, breaking ties "
        "by job id; assign its current operation to the eligible machine with "
        "the shortest processing time."
    ),
    "SPT": (
        "Select the eligible current operation-machine pair with the shortest "
        "processing time, breaking ties by job id and machine id."
    ),
    "MWKR": (
        "Select the eligible job with the largest remaining work. Remaining "
        "work is the sum, from the current operation through the final operation, "
        "of each operation's shortest positive technological processing time. "
        "Break ties by release time and job id; assign the current operation to "
        "its shortest-time eligible machine."
    ),
    "MOR": (
        "Select the eligible job with the most remaining operations, breaking "
        "ties by remaining work, release time, and job id; assign its current "
        "operation to the eligible machine with the shortest processing time."
    ),
}


def _is_blocked(environment, batch_index, job_index):
    if bool(environment.mask_job_procing_batch[batch_index, job_index].item()):
        return True
    if bool(environment.mask_job_finish_batch[batch_index, job_index].item()):
        return True
    unreleased = getattr(environment, "mask_job_unreleased_batch", None)
    return bool(
        unreleased is not None
        and unreleased[batch_index, job_index].item()
    )


def _machine_is_blocked(environment, batch_index, machine_index):
    if bool(environment.mask_ma_procing_batch[batch_index, machine_index].item()):
        return True
    failed = getattr(environment, "mask_ma_failed_batch", None)
    return bool(
        failed is not None and failed[batch_index, machine_index].item()
    )


def _eligible_machines(environment, batch_index, operation_index):
    candidates = []
    for machine_index in range(environment.num_mas):
        if _machine_is_blocked(environment, batch_index, machine_index):
            continue
        adjacent = int(
            environment.ope_ma_adj_batch[
                batch_index, operation_index, machine_index
            ].item()
        )
        processing_time = float(
            environment.proc_times_batch[
                batch_index, operation_index, machine_index
            ].item()
        )
        if adjacent == 1 and processing_time > 0:
            candidates.append((processing_time, machine_index))
    return candidates


def _remaining_work(environment, batch_index, job_index, operation_index):
    last_operation = int(
        environment.end_ope_biases_batch[batch_index, job_index].item()
    )
    total = 0.0
    for remaining_operation in range(operation_index, last_operation + 1):
        processing_times = []
        for machine_index in range(environment.num_mas):
            adjacent = int(
                environment.ope_ma_adj_batch[
                    batch_index, remaining_operation, machine_index
                ].item()
            )
            processing_time = float(
                environment.proc_times_batch[
                    batch_index, remaining_operation, machine_index
                ].item()
            )
            if adjacent == 1 and processing_time > 0:
                processing_times.append(processing_time)
        if not processing_times:
            raise RuntimeError(
                "Operation {0} of job {1} has no technological machine option.".format(
                    remaining_operation, job_index
                )
            )
        total += min(processing_times)
    return total


def _job_candidates(environment, batch_index):
    candidates = []
    release_times = getattr(environment, "job_release_times_batch", None)
    for job_index in range(environment.num_jobs):
        if _is_blocked(environment, batch_index, job_index):
            continue
        operation_index = int(
            environment.ope_step_batch[batch_index, job_index].item()
        )
        machines = _eligible_machines(
            environment, batch_index, operation_index
        )
        if not machines:
            continue
        machines.sort()
        release_time = (
            float(release_times[batch_index, job_index].item())
            if release_times is not None
            else 0.0
        )
        last_operation = int(
            environment.end_ope_biases_batch[batch_index, job_index].item()
        )
        candidates.append(
            {
                "job": job_index,
                "operation": operation_index,
                "machine": machines[0][1],
                "processing_time": machines[0][0],
                "release_time": release_time,
                "remaining_operations": last_operation - operation_index + 1,
                "remaining_work": _remaining_work(
                    environment, batch_index, job_index, operation_index
                ),
            }
        )
    return candidates


def select_dispatching_actions(environment, rule):
    """Return one deterministic action for every active batch instance."""
    normalized_rule = rule.upper()
    if normalized_rule not in RULES:
        raise ValueError(
            "Unknown dispatching rule {0!r}; expected one of {1}.".format(
                rule, ", ".join(RULES)
            )
        )

    actions = []
    for batch_index in environment.batch_idxes.tolist():
        candidates = _job_candidates(environment, batch_index)
        if not candidates:
            raise RuntimeError(
                "No eligible action for active batch index {0}.".format(
                    batch_index
                )
            )

        if normalized_rule == "FIFO":
            selected = min(
                candidates,
                key=lambda item: (item["release_time"], item["job"]),
            )
        elif normalized_rule == "SPT":
            selected = min(
                candidates,
                key=lambda item: (
                    item["processing_time"],
                    item["job"],
                    item["machine"],
                ),
            )
        elif normalized_rule == "MWKR":
            selected = min(
                candidates,
                key=lambda item: (
                    -item["remaining_work"],
                    item["release_time"],
                    item["job"],
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda item: (
                    -item["remaining_operations"],
                    -item["remaining_work"],
                    item["release_time"],
                    item["job"],
                ),
            )
        actions.append(
            (selected["operation"], selected["machine"], selected["job"])
        )

    return torch.tensor(
        actions, dtype=torch.long, device=environment.time.device
    ).t().contiguous()
