import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StabilityMetricConfig:
    """Weights for normalized start-time and machine-assignment changes."""

    start_time_weight: float = 1.0
    machine_change_weight: float = 1.0
    duration_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        for name, value in (
            ("start_time_weight", self.start_time_weight),
            ("machine_change_weight", self.machine_change_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{0} must be finite and non-negative.".format(name))
        if not math.isfinite(self.duration_epsilon) or self.duration_epsilon <= 0:
            raise ValueError("duration_epsilon must be positive and finite.")
        if self.start_time_weight == 0 and self.machine_change_weight == 0:
            raise ValueError("At least one stability weight must be positive.")


@dataclass(frozen=True)
class OperationStability:
    operation_id: int
    reference_machine: int
    disrupted_machine: int
    reference_start: float
    disrupted_start: float
    reference_duration: float
    signed_start_deviation: float
    absolute_start_deviation: float
    normalized_start_deviation: float
    machine_changed: bool
    stability_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "reference_machine": self.reference_machine,
            "disrupted_machine": self.disrupted_machine,
            "reference_start": self.reference_start,
            "disrupted_start": self.disrupted_start,
            "reference_duration": self.reference_duration,
            "signed_start_deviation": self.signed_start_deviation,
            "absolute_start_deviation": self.absolute_start_deviation,
            "normalized_start_deviation": self.normalized_start_deviation,
            "machine_changed": self.machine_changed,
            "stability_cost": self.stability_cost,
        }


@dataclass(frozen=True)
class ScheduleStability:
    operations: Tuple[OperationStability, ...]
    start_deviation_sum: float
    start_deviation_mean: float
    normalized_start_deviation_sum: float
    normalized_start_deviation_mean: float
    machine_changes: int
    machine_change_rate: float
    stability_cost_sum: float
    stability_cost_mean: float

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_count": self.operation_count,
            "start_deviation_sum": self.start_deviation_sum,
            "start_deviation_mean": self.start_deviation_mean,
            "normalized_start_deviation_sum": self.normalized_start_deviation_sum,
            "normalized_start_deviation_mean": self.normalized_start_deviation_mean,
            "machine_changes": self.machine_changes,
            "machine_change_rate": self.machine_change_rate,
            "stability_cost_sum": self.stability_cost_sum,
            "stability_cost_mean": self.stability_cost_mean,
        }


@dataclass(frozen=True)
class StabilityBudgetStatus:
    budget: float
    consumed: float
    remaining: float
    violation: float
    accounting_error: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "stability_budget": self.budget,
            "stability_consumed": self.consumed,
            "stability_remaining": self.remaining,
            "stability_violation": self.violation,
            "budget_accounting_error": self.accounting_error,
        }


class StabilityBudgetTracker:
    """Accumulate non-negative stability costs against an explicit budget."""

    def __init__(self, budget: float) -> None:
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and non-negative.")
        self.budget = float(budget)
        self.consumed = 0.0

    def consume(self, cost: float) -> StabilityBudgetStatus:
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("stability cost must be finite and non-negative.")
        self.consumed += float(cost)
        return self.status()

    def status(self) -> StabilityBudgetStatus:
        remaining = max(0.0, self.budget - self.consumed)
        violation = max(0.0, self.consumed - self.budget)
        accounting_error = abs(
            self.consumed + remaining - self.budget - violation
        )
        return StabilityBudgetStatus(
            budget=self.budget,
            consumed=self.consumed,
            remaining=remaining,
            violation=violation,
            accounting_error=accounting_error,
        )


def _schedule_rows(schedule: Any) -> List[List[float]]:
    if hasattr(schedule, "detach"):
        schedule = schedule.detach().cpu().tolist()
    elif hasattr(schedule, "tolist"):
        schedule = schedule.tolist()
    rows = [list(row) for row in schedule]
    if any(len(row) < 4 for row in rows):
        raise ValueError("Every schedule row must contain status, machine, start, end.")
    return rows


def calculate_schedule_stability(
    reference_schedule: Any,
    disrupted_schedule: Any,
    num_operations: Optional[int] = None,
    reference_durations: Optional[Sequence[float]] = None,
    config: Optional[StabilityMetricConfig] = None,
) -> ScheduleStability:
    """Compare two complete schedules operation by operation.

    Start-time deviation is normalized by each reference operation duration so
    the mean cost remains comparable across instance sizes and processing scales.
    """

    metric = config or StabilityMetricConfig()
    reference_rows = _schedule_rows(reference_schedule)
    disrupted_rows = _schedule_rows(disrupted_schedule)
    if len(reference_rows) != len(disrupted_rows):
        raise ValueError("Reference and disrupted schedules must have equal length.")
    count = len(reference_rows) if num_operations is None else int(num_operations)
    if count <= 0 or count > len(reference_rows):
        raise ValueError("num_operations must select at least one available row.")
    if reference_durations is not None and len(reference_durations) < count:
        raise ValueError("reference_durations is shorter than num_operations.")

    operations: List[OperationStability] = []
    for operation_id in range(count):
        reference = reference_rows[operation_id]
        disrupted = disrupted_rows[operation_id]
        if int(round(float(reference[0]))) != 1:
            raise ValueError("Reference schedule contains an unscheduled operation.")
        if int(round(float(disrupted[0]))) != 1:
            raise ValueError("Disrupted schedule contains an unscheduled operation.")

        reference_machine = int(round(float(reference[1])))
        disrupted_machine = int(round(float(disrupted[1])))
        reference_start = float(reference[2])
        disrupted_start = float(disrupted[2])
        if reference_durations is None:
            reference_duration = float(reference[3]) - reference_start
        else:
            reference_duration = float(reference_durations[operation_id])
        values = (reference_start, disrupted_start, reference_duration)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Schedule times and durations must be finite.")
        if reference_duration <= 0:
            raise ValueError("Reference operation durations must be positive.")

        signed_deviation = disrupted_start - reference_start
        absolute_deviation = abs(signed_deviation)
        normalized_deviation = absolute_deviation / max(
            reference_duration, metric.duration_epsilon
        )
        machine_changed = reference_machine != disrupted_machine
        stability_cost = (
            metric.start_time_weight * normalized_deviation
            + metric.machine_change_weight * float(machine_changed)
        )
        operations.append(
            OperationStability(
                operation_id=operation_id,
                reference_machine=reference_machine,
                disrupted_machine=disrupted_machine,
                reference_start=reference_start,
                disrupted_start=disrupted_start,
                reference_duration=reference_duration,
                signed_start_deviation=signed_deviation,
                absolute_start_deviation=absolute_deviation,
                normalized_start_deviation=normalized_deviation,
                machine_changed=machine_changed,
                stability_cost=stability_cost,
            )
        )

    start_sum = sum(item.absolute_start_deviation for item in operations)
    normalized_sum = sum(item.normalized_start_deviation for item in operations)
    machine_changes = sum(int(item.machine_changed) for item in operations)
    cost_sum = sum(item.stability_cost for item in operations)
    return ScheduleStability(
        operations=tuple(operations),
        start_deviation_sum=start_sum,
        start_deviation_mean=start_sum / count,
        normalized_start_deviation_sum=normalized_sum,
        normalized_start_deviation_mean=normalized_sum / count,
        machine_changes=machine_changes,
        machine_change_rate=machine_changes / count,
        stability_cost_sum=cost_sum,
        stability_cost_mean=cost_sum / count,
    )
