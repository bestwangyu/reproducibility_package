import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from dynamic.breakdowns import (
    MachineBreakdownScenario,
    load_breakdown_scenarios,
)
from dynamic.stability import StabilityMetricConfig
from env.dynamic_fjsp_env import DynamicFJSPEnv


BreakdownScenarioInput = Optional[
    Union[
        MachineBreakdownScenario,
        Dict,
        str,
        Path,
        Sequence[MachineBreakdownScenario],
    ]
]


class CompositeDynamicFJSPEnv(DynamicFJSPEnv):
    """Dynamic FJSP with job arrivals and interrupt-resume machine failures."""

    EVENT_TIME_TOLERANCE = 1e-5

    def __init__(
        self,
        case,
        env_paras,
        data_source='case',
        arrival_scenario=None,
        breakdown_scenario: BreakdownScenarioInput = None,
        reference_schedules: Optional[Any] = None,
        stability_budget: Optional[Any] = None,
        stability_metric_config: Optional[Any] = None,
    ):
        super().__init__(
            case=case,
            env_paras=env_paras,
            data_source=data_source,
            arrival_scenario=arrival_scenario,
        )
        scenario_input = breakdown_scenario
        if scenario_input is None:
            scenario_input = env_paras.get("breakdown_scenario")
        self.breakdown_scenarios = self._resolve_breakdown_scenarios(
            scenario_input
        )
        self.breakdowns_enabled = any(
            scenario.breakdowns_enabled
            for scenario in self.breakdown_scenarios
        )
        self.mask_ma_failed_batch = torch.zeros(
            (self.batch_size, self.num_mas),
            dtype=torch.bool,
            device=self.time.device,
        )
        self.operation_interruption_batch = torch.zeros(
            (self.batch_size, self.num_opes),
            dtype=self.time.dtype,
            device=self.time.device,
        )
        self.breakdown_history: List[List[Dict]] = [
            [] for _ in range(self.batch_size)
        ]
        self._build_breakdown_tensors()
        self._initialize_online_stability(
            reference_schedules,
            stability_budget,
            stability_metric_config,
        )
        self._refresh_state_references()

    def _initialize_online_stability(
        self,
        reference_schedules: Optional[Any],
        stability_budget: Optional[Any],
        stability_metric_config: Optional[Any],
    ) -> None:
        if stability_metric_config is None:
            metric_config = StabilityMetricConfig()
        elif isinstance(stability_metric_config, StabilityMetricConfig):
            metric_config = stability_metric_config
        elif isinstance(stability_metric_config, dict):
            metric_config = StabilityMetricConfig(**stability_metric_config)
        else:
            raise TypeError(
                "stability_metric_config must be a StabilityMetricConfig or dict."
            )
        self.stability_metric_config = metric_config
        self.online_stability_enabled = reference_schedules is not None
        dtype = self.time.dtype
        device = self.time.device

        self.reference_schedules_batch = None
        self.reference_operation_durations_batch = None
        self.stability_budget_batch = torch.zeros(
            self.batch_size, dtype=dtype, device=device
        )
        self.stability_budget_total_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_cost_sum_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_cost_mean_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_remaining_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_violation_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_remaining_total_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_violation_total_batch = torch.zeros_like(
            self.stability_budget_batch
        )
        self.stability_signed_start_deviation_batch = torch.zeros(
            (self.batch_size, self.num_opes), dtype=dtype, device=device
        )
        self.stability_normalized_start_deviation_batch = torch.zeros_like(
            self.stability_signed_start_deviation_batch
        )
        self.stability_machine_change_batch = torch.zeros(
            (self.batch_size, self.num_opes), dtype=torch.bool, device=device
        )
        self.stability_cost_operation_batch = torch.zeros_like(
            self.stability_signed_start_deviation_batch
        )
        self.stability_accounted_batch = torch.zeros(
            (self.batch_size, self.num_opes), dtype=torch.bool, device=device
        )
        self.candidate_stability_cost_batch = torch.zeros(
            (self.batch_size, self.num_jobs, self.num_mas),
            dtype=dtype,
            device=device,
        )
        if not self.online_stability_enabled:
            if stability_budget is not None:
                raise ValueError(
                    "reference_schedules are required when stability_budget is set."
                )
            return
        if stability_budget is None:
            raise ValueError(
                "stability_budget is required when reference_schedules are provided."
            )

        references = torch.as_tensor(
            reference_schedules, dtype=dtype, device=device
        )
        if references.dim() == 2:
            references = references.unsqueeze(0)
        if references.dim() != 3 or references.size(2) < 4:
            raise ValueError(
                "reference_schedules must have shape [batch, operations, 4]."
            )
        if references.size(0) == 1 and self.batch_size > 1:
            references = references.expand(self.batch_size, -1, -1).clone()
        if references.size(0) != self.batch_size:
            raise ValueError(
                "Expected 1 or {0} reference schedules, got {1}.".format(
                    self.batch_size, references.size(0)
                )
            )
        if references.size(1) < self.num_opes:
            raise ValueError(
                "Reference schedules contain fewer operations than the environment."
            )
        references = references[:, :self.num_opes, :4].clone()
        operation_ids = torch.arange(self.num_opes, device=device)[None, :]
        valid_operations = operation_ids < self.nums_opes[:, None]
        scheduled = torch.isclose(
            references[:, :, 0], torch.ones_like(references[:, :, 0])
        )
        if ((~scheduled) & valid_operations).any():
            raise ValueError("Reference schedules contain unscheduled operations.")
        durations = references[:, :, 3] - references[:, :, 2]
        if ((durations <= 0) & valid_operations).any():
            raise ValueError("Reference operation durations must be positive.")
        self.reference_schedules_batch = references
        self.reference_operation_durations_batch = durations

        budgets = torch.as_tensor(stability_budget, dtype=dtype, device=device)
        if budgets.dim() == 0:
            budgets = budgets.repeat(self.batch_size)
        else:
            budgets = budgets.flatten()
            if budgets.numel() == 1 and self.batch_size > 1:
                budgets = budgets.repeat(self.batch_size)
        if budgets.numel() != self.batch_size:
            raise ValueError(
                "Expected 1 or {0} stability budgets, got {1}.".format(
                    self.batch_size, budgets.numel()
                )
            )
        if not bool(torch.isfinite(budgets).all().item()) or bool(
            (budgets < 0).any().item()
        ):
            raise ValueError("stability_budget must be finite and non-negative.")
        self.stability_budget_batch.copy_(budgets)
        operation_counts = self.nums_opes.to(dtype=dtype)
        self.stability_budget_total_batch.copy_(budgets * operation_counts)
        self.stability_remaining_batch.copy_(budgets)
        self.stability_remaining_total_batch.copy_(
            self.stability_budget_total_batch
        )

    def _update_candidate_stability_costs(self) -> None:
        if not self.online_stability_enabled:
            self.candidate_stability_cost_batch.zero_()
            return
        operation_indexes = torch.where(
            self.ope_step_batch > self.end_ope_biases_batch,
            self.end_ope_biases_batch,
            self.ope_step_batch,
        )
        reference_rows = self.reference_schedules_batch.gather(
            1,
            operation_indexes[:, :, None].expand(-1, -1, 4),
        )
        reference_machines = reference_rows[:, :, 1].long()
        reference_starts = reference_rows[:, :, 2]
        reference_durations = self.reference_operation_durations_batch.gather(
            1, operation_indexes
        )
        normalized_deviation = (
            self.time[:, None] - reference_starts
        ).abs() / reference_durations.clamp_min(
            self.stability_metric_config.duration_epsilon
        )
        machine_indexes = torch.arange(
            self.num_mas, device=self.time.device
        )[None, None, :]
        machine_changes = machine_indexes != reference_machines[:, :, None]
        costs = (
            self.stability_metric_config.start_time_weight
            * normalized_deviation[:, :, None]
            + self.stability_metric_config.machine_change_weight
            * machine_changes.to(dtype=self.time.dtype)
        )
        self.candidate_stability_cost_batch.copy_(costs)

    def _record_online_stability(
        self,
        batch_indexes: torch.Tensor,
        operation_indexes: torch.Tensor,
        machine_indexes: torch.Tensor,
        start_times: torch.Tensor,
    ) -> None:
        if not self.online_stability_enabled:
            return
        batch_indexes = batch_indexes.long()
        operation_indexes = operation_indexes.long()
        machine_indexes = machine_indexes.long()
        if self.stability_accounted_batch[
            batch_indexes, operation_indexes
        ].any():
            raise RuntimeError("An operation stability cost was accounted twice.")

        references = self.reference_schedules_batch[
            batch_indexes, operation_indexes
        ]
        reference_machines = references[:, 1].long()
        reference_starts = references[:, 2]
        reference_durations = self.reference_operation_durations_batch[
            batch_indexes, operation_indexes
        ]
        signed_deviation = start_times - reference_starts
        normalized_deviation = signed_deviation.abs() / reference_durations.clamp_min(
            self.stability_metric_config.duration_epsilon
        )
        machine_changes = machine_indexes != reference_machines
        costs = (
            self.stability_metric_config.start_time_weight * normalized_deviation
            + self.stability_metric_config.machine_change_weight
            * machine_changes.to(dtype=self.time.dtype)
        )
        if not bool(torch.isfinite(costs).all().item()):
            raise RuntimeError("Online stability produced a non-finite cost.")

        self.stability_signed_start_deviation_batch[
            batch_indexes, operation_indexes
        ] = signed_deviation
        self.stability_normalized_start_deviation_batch[
            batch_indexes, operation_indexes
        ] = normalized_deviation
        self.stability_machine_change_batch[
            batch_indexes, operation_indexes
        ] = machine_changes
        self.stability_cost_operation_batch[
            batch_indexes, operation_indexes
        ] = costs
        self.stability_accounted_batch[
            batch_indexes, operation_indexes
        ] = True
        self.stability_cost_sum_batch[batch_indexes] += costs

        operation_counts = self.nums_opes[batch_indexes].to(dtype=self.time.dtype)
        mean_costs = self.stability_cost_sum_batch[batch_indexes] / operation_counts
        budgets = self.stability_budget_batch[batch_indexes]
        remaining = torch.clamp(budgets - mean_costs, min=0.0)
        violation = torch.clamp(mean_costs - budgets, min=0.0)
        self.stability_cost_mean_batch[batch_indexes] = mean_costs
        self.stability_remaining_batch[batch_indexes] = remaining
        self.stability_violation_batch[batch_indexes] = violation
        self.stability_remaining_total_batch[batch_indexes] = (
            remaining * operation_counts
        )
        self.stability_violation_total_batch[batch_indexes] = (
            violation * operation_counts
        )

    def _resolve_breakdown_scenarios(
        self, scenario_input: BreakdownScenarioInput
    ) -> List[MachineBreakdownScenario]:
        if scenario_input is None:
            scenarios = [MachineBreakdownScenario.static(self.num_mas)]
        elif isinstance(scenario_input, MachineBreakdownScenario):
            scenarios = [scenario_input]
        elif isinstance(scenario_input, (str, Path)):
            scenarios = load_breakdown_scenarios(scenario_input)
        elif isinstance(scenario_input, dict):
            scenarios = [MachineBreakdownScenario.from_dict(scenario_input)]
        else:
            scenarios = list(scenario_input)
        if len(scenarios) == 1 and self.batch_size > 1:
            scenarios = scenarios * self.batch_size
        if len(scenarios) != self.batch_size:
            raise ValueError(
                "Expected 1 or {0} breakdown scenarios, got {1}.".format(
                    self.batch_size, len(scenarios)
                )
            )
        if not all(
            isinstance(scenario, MachineBreakdownScenario)
            for scenario in scenarios
        ):
            raise TypeError(
                "Every breakdown scenario must be a MachineBreakdownScenario."
            )
        for scenario in scenarios:
            if scenario.num_machines != self.num_mas:
                raise ValueError(
                    "Scenario {0} has {1} machines; expected {2}.".format(
                        scenario.scenario_id,
                        scenario.num_machines,
                        self.num_mas,
                    )
                )
        return scenarios

    def _build_breakdown_tensors(self) -> None:
        max_events = max(
            1, max(len(scenario.events) for scenario in self.breakdown_scenarios)
        )
        starts = torch.full(
            (self.batch_size, max_events),
            float('inf'),
            dtype=self.time.dtype,
            device=self.time.device,
        )
        repairs = torch.full_like(starts, float('inf'))
        machines = torch.full(
            (self.batch_size, max_events),
            -1,
            dtype=torch.long,
            device=self.time.device,
        )
        valid = torch.zeros(
            (self.batch_size, max_events),
            dtype=torch.bool,
            device=self.time.device,
        )
        for batch_index, scenario in enumerate(self.breakdown_scenarios):
            for event_index, event in enumerate(scenario.events):
                starts[batch_index, event_index] = event.start_time
                repairs[batch_index, event_index] = event.repair_end_time
                machines[batch_index, event_index] = event.machine_id
                valid[batch_index, event_index] = True
        self.breakdown_start_times_batch = starts
        self.breakdown_repair_end_times_batch = repairs
        self.breakdown_machine_batch = machines
        self.breakdown_event_valid_batch = valid
        self.breakdown_event_started_batch = torch.zeros_like(valid)
        self.breakdown_event_repaired_batch = torch.zeros_like(valid)

    def _refresh_state_references(self) -> None:
        super()._refresh_state_references()
        if hasattr(self, "mask_ma_failed_batch"):
            self.state.mask_ma_failed_batch = self.mask_ma_failed_batch
        if hasattr(self, "stability_budget_batch"):
            self._update_candidate_stability_costs()
            self.state.stability_budget_batch = self.stability_budget_batch
            self.state.stability_cost_mean_batch = self.stability_cost_mean_batch
            self.state.stability_remaining_batch = self.stability_remaining_batch
            self.state.stability_violation_batch = self.stability_violation_batch
            self.state.candidate_stability_cost_batch = (
                self.candidate_stability_cost_batch
            )

    def if_no_eligible(self):
        if not getattr(self, "breakdowns_enabled", False):
            return super().if_no_eligible()
        ope_step_batch = torch.where(
            self.ope_step_batch > self.end_ope_biases_batch,
            self.end_ope_biases_batch,
            self.ope_step_batch,
        )
        op_proc_time = self.proc_times_batch.gather(
            1,
            ope_step_batch.unsqueeze(-1).expand(
                -1, -1, self.proc_times_batch.size(2)
            ),
        )
        ma_eligible = ~(
            self.mask_ma_procing_batch | self.mask_ma_failed_batch
        ).unsqueeze(1).expand_as(op_proc_time)
        job_blocked = (
            self.mask_job_procing_batch
            | self.mask_job_finish_batch
            | self.mask_job_unreleased_batch
        )
        job_eligible = ~job_blocked[:, :, None].expand_as(op_proc_time)
        return torch.sum(
            torch.where(
                ma_eligible & job_eligible,
                op_proc_time.double(),
                0.0,
            ).transpose(1, 2),
            dim=[1, 2],
        )

    def _apply_interruption(
        self,
        batch_index: int,
        machine_index: int,
        repair_duration: float,
    ) -> Dict:
        job_index = int(
            self.machines_batch[batch_index, machine_index, 3].item()
        )
        operation_index = int(
            self.ope_step_batch[batch_index, job_index].item() - 1
        )
        duration = torch.tensor(
            repair_duration,
            dtype=self.time.dtype,
            device=self.time.device,
        )
        self.machines_batch[batch_index, machine_index, 1] += duration
        self.feat_mas_batch[batch_index, 1, machine_index] += duration
        self.operation_interruption_batch[batch_index, operation_index] += duration
        self.feat_opes_batch[batch_index, 2, operation_index] += duration

        first_operation = int(
            self.num_ope_biases_batch[batch_index, job_index].item()
        )
        last_operation = int(
            self.end_ope_biases_batch[batch_index, job_index].item()
        )
        self.feat_opes_batch[
            batch_index, 4, first_operation:last_operation + 1
        ] += duration
        if operation_index < last_operation:
            self.feat_opes_batch[
                batch_index, 5, operation_index + 1:last_operation + 1
            ] += duration
        self.schedules_batch[batch_index, operation_index, 3] += duration
        if operation_index < last_operation:
            self.schedules_batch[
                batch_index,
                operation_index + 1:last_operation + 1,
                2:4,
            ] += duration
        return {
            "job": job_index,
            "operation": operation_index,
        }

    def _record_breakdown(
        self,
        batch_index: int,
        event_index: int,
        interrupted: bool,
        interruption: Optional[Dict] = None,
        finalized: bool = False,
    ) -> None:
        self.breakdown_history[batch_index].append(
            {
                "event_index": event_index,
                "machine": int(
                    self.breakdown_machine_batch[
                        batch_index, event_index
                    ].item()
                ),
                "start_time": float(
                    self.breakdown_start_times_batch[
                        batch_index, event_index
                    ].item()
                ),
                "repair_end_time": float(
                    self.breakdown_repair_end_times_batch[
                        batch_index, event_index
                    ].item()
                ),
                "interrupted": interrupted,
                "job": None if interruption is None else interruption["job"],
                "operation": (
                    None if interruption is None else interruption["operation"]
                ),
                "finalized_after_last_dispatch": finalized,
            }
        )

    def _process_repairs(self, event_time, flag_need_trans) -> None:
        repairs = (
            self.breakdown_event_valid_batch
            & self.breakdown_event_started_batch
            & ~self.breakdown_event_repaired_batch
            & torch.isclose(
                self.breakdown_repair_end_times_batch,
                event_time[:, None],
                rtol=0.0,
                atol=self.EVENT_TIME_TOLERANCE,
            )
            & flag_need_trans[:, None]
        )
        for batch_index, event_index in torch.nonzero(
            repairs, as_tuple=False
        ).tolist():
            machine_index = int(
                self.breakdown_machine_batch[batch_index, event_index].item()
            )
            self.mask_ma_failed_batch[batch_index, machine_index] = False
            self.breakdown_event_repaired_batch[batch_index, event_index] = True
            if not self.mask_ma_procing_batch[batch_index, machine_index]:
                self.feat_mas_batch[batch_index, 1, machine_index] = event_time[
                    batch_index
                ]

    def _process_breakdown_starts(self, event_time, flag_need_trans) -> None:
        starts = (
            self.breakdown_event_valid_batch
            & ~self.breakdown_event_started_batch
            & torch.isclose(
                self.breakdown_start_times_batch,
                event_time[:, None],
                rtol=0.0,
                atol=self.EVENT_TIME_TOLERANCE,
            )
            & flag_need_trans[:, None]
        )
        for batch_index, event_index in torch.nonzero(
            starts, as_tuple=False
        ).tolist():
            machine_index = int(
                self.breakdown_machine_batch[batch_index, event_index].item()
            )
            repair_duration = float(
                (
                    self.breakdown_repair_end_times_batch[
                        batch_index, event_index
                    ]
                    - self.breakdown_start_times_batch[batch_index, event_index]
                ).item()
            )
            interrupted = bool(
                self.mask_ma_procing_batch[batch_index, machine_index].item()
                and self.machines_batch[batch_index, machine_index, 1]
                > event_time[batch_index]
            )
            interruption = None
            if interrupted:
                interruption = self._apply_interruption(
                    batch_index, machine_index, repair_duration
                )
            else:
                self.feat_mas_batch[batch_index, 1, machine_index] = (
                    self.breakdown_repair_end_times_batch[
                        batch_index, event_index
                    ]
                )
            self.mask_ma_failed_batch[batch_index, machine_index] = True
            self.breakdown_event_started_batch[batch_index, event_index] = True
            self._record_breakdown(
                batch_index,
                event_index,
                interrupted,
                interruption=interruption,
            )

    def next_time(self, flag_trans_2_next_time):
        if not getattr(self, "breakdowns_enabled", False):
            return super().next_time(flag_trans_2_next_time)

        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
        infinity = torch.full_like(self.time, float('inf'))
        machine_available = self.machines_batch[:, :, 1]
        time_tolerance = self.EVENT_TIME_TOLERANCE
        future_machine_times = torch.where(
            self.mask_ma_procing_batch
            & (machine_available > self.time[:, None]),
            machine_available,
            infinity[:, None],
        )
        next_machine_time = torch.min(future_machine_times, dim=1)[0]

        future_release_times = torch.where(
            self.mask_job_unreleased_batch
            & (self.job_release_times_batch > self.time[:, None]),
            self.job_release_times_batch,
            infinity[:, None],
        )
        next_release_time = torch.min(future_release_times, dim=1)[0]
        future_breakdown_times = torch.where(
            self.breakdown_event_valid_batch
            & ~self.breakdown_event_started_batch
            & (self.breakdown_start_times_batch > self.time[:, None]),
            self.breakdown_start_times_batch,
            infinity[:, None],
        )
        next_breakdown_time = torch.min(future_breakdown_times, dim=1)[0]
        future_repair_times = torch.where(
            self.breakdown_event_valid_batch
            & self.breakdown_event_started_batch
            & ~self.breakdown_event_repaired_batch
            & (self.breakdown_repair_end_times_batch > self.time[:, None]),
            self.breakdown_repair_end_times_batch,
            infinity[:, None],
        )
        next_repair_time = torch.min(future_repair_times, dim=1)[0]
        next_event_time = torch.minimum(
            torch.minimum(next_machine_time, next_release_time),
            torch.minimum(next_breakdown_time, next_repair_time),
        )
        if torch.isinf(next_event_time[flag_need_trans]).any():
            raise RuntimeError(
                "No completion, arrival, breakdown, or repair event is available."
            )

        event_time = torch.where(flag_need_trans, next_event_time, self.time)
        completed_machines = (
            self.mask_ma_procing_batch
            & torch.isclose(
                machine_available,
                event_time[:, None],
                rtol=0.0,
                atol=time_tolerance,
            )
            & flag_need_trans[:, None]
        )
        self.time = event_time
        self.machines_batch[:, :, 0][completed_machines] = 1
        completed_indexes = torch.nonzero(completed_machines, as_tuple=False)
        if completed_indexes.numel() > 0:
            batch_indexes = completed_indexes[:, 0]
            machine_indexes = completed_indexes[:, 1]
            job_indexes = self.machines_batch[
                batch_indexes, machine_indexes, 3
            ].long()
            self.mask_job_procing_batch[batch_indexes, job_indexes] = False
            self.mask_ma_procing_batch[completed_machines] = False

        self._process_repairs(event_time, flag_need_trans)
        self._process_breakdown_starts(event_time, flag_need_trans)
        self._release_arrived_jobs()

        utilization = self.machines_batch[:, :, 2]
        current_time = self.time[:, None].expand_as(utilization)
        utilization = torch.minimum(utilization, current_time)
        self.feat_mas_batch[:, 2, :] = utilization.div(
            self.time[:, None] + 1e-5
        )
        self.mask_job_finish_batch = torch.where(
            self.ope_step_batch == self.end_ope_biases_batch + 1,
            True,
            self.mask_job_finish_batch,
        )
        self._refresh_state_references()

    def _finalize_inflight_breakdowns(self) -> None:
        tolerance = 1e-6
        done_indexes = torch.nonzero(self.done_batch, as_tuple=False).flatten()
        for batch_index in done_indexes.tolist():
            for event_index in range(
                self.breakdown_event_valid_batch.size(1)
            ):
                if not self.breakdown_event_valid_batch[
                    batch_index, event_index
                ]:
                    continue
                if self.breakdown_event_started_batch[batch_index, event_index]:
                    continue
                machine_index = int(
                    self.breakdown_machine_batch[
                        batch_index, event_index
                    ].item()
                )
                if not self.mask_ma_procing_batch[
                    batch_index, machine_index
                ]:
                    continue
                start_time = float(
                    self.breakdown_start_times_batch[
                        batch_index, event_index
                    ].item()
                )
                machine_completion = float(
                    self.machines_batch[batch_index, machine_index, 1].item()
                )
                if start_time < float(self.time[batch_index].item()) - tolerance:
                    continue
                if start_time >= machine_completion - tolerance:
                    continue
                repair_duration = float(
                    (
                        self.breakdown_repair_end_times_batch[
                            batch_index, event_index
                        ]
                        - self.breakdown_start_times_batch[
                            batch_index, event_index
                        ]
                    ).item()
                )
                interruption = self._apply_interruption(
                    batch_index, machine_index, repair_duration
                )
                self.breakdown_event_started_batch[
                    batch_index, event_index
                ] = True
                self.breakdown_event_repaired_batch[
                    batch_index, event_index
                ] = True
                self._record_breakdown(
                    batch_index,
                    event_index,
                    True,
                    interruption=interruption,
                    finalized=True,
                )

    def step(self, actions):
        if self.mask_ma_failed_batch[
            self.batch_idxes, actions[1, :]
        ].any():
            raise ValueError("An action selected a machine during repair.")
        stability_batch_indexes = self.batch_idxes.clone()
        stability_operation_indexes = actions[0, :].clone()
        stability_machine_indexes = actions[1, :].clone()
        stability_start_times = self.time[stability_batch_indexes].clone()
        state, rewards, done_batch = super().step(actions)
        self._record_online_stability(
            stability_batch_indexes,
            stability_operation_indexes,
            stability_machine_indexes,
            stability_start_times,
        )
        makespan_before_finalization = self.makespan_batch.clone()
        self._finalize_inflight_breakdowns()
        current_makespan = torch.max(
            self.feat_opes_batch[:, 4, :], dim=1
        )[0]
        finalization_penalty = current_makespan - makespan_before_finalization
        if finalization_penalty.any():
            self.raw_reward_batch -= finalization_penalty
            self.compensated_reward_batch -= finalization_penalty
            if self.dynamic_reward_mode == "arrival_compensated":
                self.reward_batch = self.compensated_reward_batch
            else:
                self.reward_batch = self.raw_reward_batch
            rewards = self.reward_batch
        self.makespan_batch = current_makespan
        self._refresh_state_references()
        return state, rewards, done_batch

    def expected_operation_duration(self, batch_id, operation_id, machine_id):
        base_duration = super().expected_operation_duration(
            batch_id, operation_id, machine_id
        )
        interruption = self.operation_interruption_batch[
            batch_id, operation_id
        ].item()
        return base_duration + interruption

    def validate_gantt(self):
        feasible, schedules = super().validate_gantt()
        tolerance = 1e-5
        starts_during_repair = 0
        for batch_index in range(self.batch_size):
            scenario = self.breakdown_scenarios[batch_index]
            for operation_index in range(int(self.nums_opes[batch_index])):
                schedule = schedules[batch_index, operation_index]
                if int(schedule[0].item()) != 1:
                    continue
                machine_index = int(schedule[1].item())
                start_time = float(schedule[2].item())
                for event in scenario.events:
                    if event.machine_id != machine_index:
                        continue
                    if (
                        start_time + tolerance >= event.start_time
                        and start_time < event.repair_end_time - tolerance
                    ):
                        starts_during_repair += 1
        return feasible and starts_during_repair == 0, schedules

    def reset(self):
        state = super().reset()
        self.mask_ma_failed_batch.zero_()
        self.operation_interruption_batch.zero_()
        self.breakdown_event_started_batch.zero_()
        self.breakdown_event_repaired_batch.zero_()
        self.breakdown_history = [[] for _ in range(self.batch_size)]
        self.stability_signed_start_deviation_batch.zero_()
        self.stability_normalized_start_deviation_batch.zero_()
        self.stability_machine_change_batch.zero_()
        self.stability_cost_operation_batch.zero_()
        self.stability_accounted_batch.zero_()
        self.stability_cost_sum_batch.zero_()
        self.stability_cost_mean_batch.zero_()
        self.stability_violation_batch.zero_()
        self.stability_violation_total_batch.zero_()
        if self.online_stability_enabled:
            self.stability_remaining_batch.copy_(self.stability_budget_batch)
            self.stability_remaining_total_batch.copy_(
                self.stability_budget_total_batch
            )
        else:
            self.stability_remaining_batch.zero_()
            self.stability_remaining_total_batch.zero_()
        self._refresh_state_references()
        return state
