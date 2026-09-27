import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from dynamic.arrivals import JobArrivalScenario, load_scenarios
from env.fjsp_env import FJSPEnv


ScenarioInput = Optional[
    Union[
        JobArrivalScenario,
        Dict,
        str,
        Path,
        Sequence[JobArrivalScenario],
    ]
]


class DynamicFJSPEnv(FJSPEnv):
    """FJSP environment with non-anticipatory, event-driven job releases."""

    REWARD_MODES = ("raw", "arrival_compensated")

    def __init__(
        self,
        case,
        env_paras,
        data_source='case',
        arrival_scenario: ScenarioInput = None,
    ):
        super().__init__(case=case, env_paras=env_paras, data_source=data_source)

        self.dynamic_reward_mode = env_paras.get("dynamic_reward_mode", "raw")
        if self.dynamic_reward_mode not in self.REWARD_MODES:
            raise ValueError(
                "dynamic_reward_mode must be one of {0}; got {1!r}.".format(
                    ", ".join(self.REWARD_MODES), self.dynamic_reward_mode
                )
            )
        self.raw_reward_batch = torch.zeros_like(self.makespan_batch)
        self.compensated_reward_batch = torch.zeros_like(self.makespan_batch)
        self.release_potential_jump_batch = torch.zeros_like(self.makespan_batch)
        self.release_potential_jump_total_batch = torch.zeros_like(
            self.makespan_batch
        )

        scenario_input = arrival_scenario
        if scenario_input is None:
            scenario_input = env_paras.get("arrival_scenario")
        self.arrival_scenarios = self._resolve_scenarios(scenario_input)
        release_matrix = [scenario.release_times for scenario in self.arrival_scenarios]
        self.job_release_times_batch = torch.tensor(
            release_matrix,
            dtype=self.time.dtype,
            device=self.time.device,
        )
        self.dynamic_enabled = bool((self.job_release_times_batch > 0).any().item())
        self.mask_job_unreleased_batch = self.job_release_times_batch > self.time[:, None]
        self.arrival_history: List[List[Dict]] = [[] for _ in range(self.batch_size)]

        self.state.mask_job_unreleased_batch = self.mask_job_unreleased_batch
        if not self.dynamic_enabled:
            self.visible_nums_opes = copy.deepcopy(self.nums_opes)
            self.old_state = copy.deepcopy(self.state)
            return

        self._capture_full_instance()
        self._apply_initial_visibility()
        self._save_dynamic_initial_state()

    def _resolve_scenarios(self, scenario_input: ScenarioInput) -> List[JobArrivalScenario]:
        if scenario_input is None:
            scenarios = [JobArrivalScenario.static(self.num_jobs)]
        elif isinstance(scenario_input, JobArrivalScenario):
            scenarios = [scenario_input]
        elif isinstance(scenario_input, (str, Path)):
            scenarios = load_scenarios(scenario_input)
        elif isinstance(scenario_input, dict):
            scenarios = [JobArrivalScenario.from_dict(scenario_input)]
        else:
            scenarios = list(scenario_input)

        if len(scenarios) == 1 and self.batch_size > 1:
            scenarios = scenarios * self.batch_size
        if len(scenarios) != self.batch_size:
            raise ValueError(
                f"Expected 1 or {self.batch_size} arrival scenarios, got {len(scenarios)}."
            )
        if not all(isinstance(item, JobArrivalScenario) for item in scenarios):
            raise TypeError("Every arrival scenario must be a JobArrivalScenario.")
        for scenario in scenarios:
            if scenario.num_jobs != self.num_jobs:
                raise ValueError(
                    f"Scenario {scenario.scenario_id} has {scenario.num_jobs} jobs; "
                    f"the environment expects {self.num_jobs}."
                )
        return scenarios

    def _capture_full_instance(self) -> None:
        self._full_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self._full_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self._full_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
        self._full_ope_sub_adj_batch = copy.deepcopy(self.ope_sub_adj_batch)
        self._full_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self._full_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)

        valid_opes = (
            torch.arange(self.num_opes, device=self.nums_opes.device)[None, :]
            < self.nums_opes[:, None]
        )
        release_by_ope = self.job_release_times_batch.gather(
            1, self.opes_appertain_batch
        )
        release_by_ope = release_by_ope * valid_opes.to(release_by_ope.dtype)
        self._full_feat_opes_batch[:, 4, :] += release_by_ope
        self._full_feat_opes_batch[:, 5, :] += release_by_ope

    def _valid_ope_mask(self) -> torch.Tensor:
        return (
            torch.arange(self.num_opes, device=self.nums_opes.device)[None, :]
            < self.nums_opes[:, None]
        )

    def _visible_ope_mask(self) -> torch.Tensor:
        released_jobs = ~self.mask_job_unreleased_batch
        return (
            released_jobs.gather(1, self.opes_appertain_batch)
            & self._valid_ope_mask()
        )

    def _apply_initial_visibility(self) -> None:
        visible_opes = self._visible_ope_mask()
        visible_edges = visible_opes[:, :, None] & visible_opes[:, None, :]

        self.proc_times_batch = torch.where(
            visible_opes[:, :, None],
            self._full_proc_times_batch,
            torch.zeros_like(self._full_proc_times_batch),
        )
        self.ope_ma_adj_batch = torch.where(
            visible_opes[:, :, None],
            self._full_ope_ma_adj_batch,
            torch.zeros_like(self._full_ope_ma_adj_batch),
        )
        self.ope_pre_adj_batch = self._full_ope_pre_adj_batch & visible_edges
        self.ope_sub_adj_batch = self._full_ope_sub_adj_batch & visible_edges
        self.cal_cumul_adj_batch = self._full_cal_cumul_adj_batch * visible_edges
        self.feat_opes_batch = torch.where(
            visible_opes[:, None, :],
            self._full_feat_opes_batch,
            torch.zeros_like(self._full_feat_opes_batch),
        )
        self.feat_mas_batch[:, 0, :] = torch.count_nonzero(
            self.ope_ma_adj_batch, dim=1
        ).float()
        self.visible_nums_opes = visible_opes.sum(dim=1)

        self.schedules_batch.zero_()
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = (
            self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        )
        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self._refresh_state_references()

    def _save_dynamic_initial_state(self) -> None:
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self._old_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
        self._old_ope_sub_adj_batch = copy.deepcopy(self.ope_sub_adj_batch)
        self._old_mask_job_unreleased_batch = copy.deepcopy(
            self.mask_job_unreleased_batch
        )
        self._old_visible_nums_opes = copy.deepcopy(self.visible_nums_opes)
        self.old_state = copy.deepcopy(self.state)

    def _refresh_state_references(self) -> None:
        self.state.ope_pre_adj_batch = self.ope_pre_adj_batch
        self.state.ope_sub_adj_batch = self.ope_sub_adj_batch
        self.state.proc_times_batch = self.proc_times_batch
        self.state.ope_ma_adj_batch = self.ope_ma_adj_batch
        self.state.feat_opes_batch = self.feat_opes_batch
        self.state.feat_mas_batch = self.feat_mas_batch
        self.state.mask_job_unreleased_batch = self.mask_job_unreleased_batch
        self.state.nums_opes_batch = self.visible_nums_opes
        self.state.time_batch = self.time

    def _release_arrived_jobs(self) -> None:
        newly_released_jobs = self.mask_job_unreleased_batch & (
            self.job_release_times_batch <= self.time[:, None]
        )
        if not newly_released_jobs.any():
            return

        potential_before_release = torch.max(
            self.feat_opes_batch[:, 4, :], dim=1
        )[0]
        previous_visible = self._visible_ope_mask()
        self.mask_job_unreleased_batch = (
            self.mask_job_unreleased_batch & ~newly_released_jobs
        )
        visible_opes = self._visible_ope_mask()
        newly_visible_opes = visible_opes & ~previous_visible
        visible_edges = visible_opes[:, :, None] & visible_opes[:, None, :]

        self.proc_times_batch = torch.where(
            newly_visible_opes[:, :, None],
            self._full_proc_times_batch,
            self.proc_times_batch,
        )
        self.ope_ma_adj_batch = torch.where(
            newly_visible_opes[:, :, None],
            self._full_ope_ma_adj_batch,
            self.ope_ma_adj_batch,
        )
        self.ope_pre_adj_batch = self._full_ope_pre_adj_batch & visible_edges
        self.ope_sub_adj_batch = self._full_ope_sub_adj_batch & visible_edges
        newly_visible_edges = visible_edges & (
            newly_visible_opes[:, :, None] | newly_visible_opes[:, None, :]
        )
        self.cal_cumul_adj_batch = torch.where(
            newly_visible_edges,
            self._full_cal_cumul_adj_batch,
            self.cal_cumul_adj_batch,
        )
        self.feat_opes_batch = torch.where(
            newly_visible_opes[:, None, :],
            self._full_feat_opes_batch,
            self.feat_opes_batch,
        )
        self.feat_mas_batch[:, 0, :] = torch.count_nonzero(
            self.ope_ma_adj_batch, dim=1
        ).float()
        self.visible_nums_opes = visible_opes.sum(dim=1)
        self.schedules_batch[:, :, 2] = torch.where(
            newly_visible_opes,
            self.feat_opes_batch[:, 5, :],
            self.schedules_batch[:, :, 2],
        )
        self.schedules_batch[:, :, 3] = torch.where(
            newly_visible_opes,
            self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :],
            self.schedules_batch[:, :, 3],
        )

        potential_after_release = torch.max(
            self.feat_opes_batch[:, 4, :], dim=1
        )[0]
        release_potential_jump = (
            potential_after_release - potential_before_release
        ).clamp_min(0)
        self.release_potential_jump_batch += release_potential_jump
        self.release_potential_jump_total_batch += release_potential_jump

        for batch_index in range(self.batch_size):
            jobs = torch.nonzero(
                newly_released_jobs[batch_index], as_tuple=False
            ).flatten()
            if jobs.numel() > 0:
                self.arrival_history[batch_index].append(
                    {
                        "time": float(self.time[batch_index].item()),
                        "jobs": tuple(int(job.item()) for job in jobs),
                        "potential_jump": float(
                            release_potential_jump[batch_index].item()
                        ),
                    }
                )
        self._refresh_state_references()

    def if_no_eligible(self):
        if not self.dynamic_enabled:
            return super().if_no_eligible()

        ope_step_batch = torch.where(
            self.ope_step_batch > self.end_ope_biases_batch,
            self.end_ope_biases_batch,
            self.ope_step_batch,
        )
        op_proc_time = self.proc_times_batch.gather(
            1,
            ope_step_batch.unsqueeze(-1).expand(-1, -1, self.proc_times_batch.size(2)),
        )
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_blocked = (
            self.mask_job_procing_batch
            | self.mask_job_finish_batch
            | self.mask_job_unreleased_batch
        )
        job_eligible = ~job_blocked[:, :, None].expand_as(op_proc_time)
        return torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0)
            .transpose(1, 2),
            dim=[1, 2],
        )

    def next_time(self, flag_trans_2_next_time):
        if not self.dynamic_enabled:
            return super().next_time(flag_trans_2_next_time)

        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
        infinity = torch.full_like(self.time, float('inf'))

        machine_available = self.machines_batch[:, :, 1]
        future_machine_times = torch.where(
            machine_available > self.time[:, None],
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
        next_event_time = torch.minimum(next_machine_time, next_release_time)
        if torch.isinf(next_event_time[flag_need_trans]).any():
            raise RuntimeError("No machine-completion or job-arrival event is available.")

        event_time = torch.where(flag_need_trans, next_event_time, self.time)
        completed_machines = (
            (machine_available == event_time[:, None])
            & (self.machines_batch[:, :, 0] == 0)
            & flag_need_trans[:, None]
        )
        self.time = event_time

        self.machines_batch[:, :, 0][completed_machines] = 1

        utilization = self.machines_batch[:, :, 2]
        current_time = self.time[:, None].expand_as(utilization)
        utilization = torch.minimum(utilization, current_time)
        self.feat_mas_batch[:, 2, :] = utilization.div(self.time[:, None] + 1e-5)

        completed_indexes = torch.nonzero(completed_machines, as_tuple=False)
        if completed_indexes.numel() > 0:
            batch_indexes = completed_indexes[:, 0]
            machine_indexes = completed_indexes[:, 1]
            job_indexes = self.machines_batch[
                batch_indexes, machine_indexes, 3
            ].long()
            self.mask_job_procing_batch[batch_indexes, job_indexes] = False
            self.mask_ma_procing_batch[completed_machines] = False

        self.mask_job_finish_batch = torch.where(
            self.ope_step_batch == self.end_ope_biases_batch + 1,
            True,
            self.mask_job_finish_batch,
        )
        self._release_arrived_jobs()

    def step(self, actions):
        if not self.dynamic_enabled:
            state, rewards, done_batch = super().step(actions)
            self.raw_reward_batch = rewards.clone()
            self.compensated_reward_batch = rewards.clone()
            self.release_potential_jump_batch.zero_()
            return state, rewards, done_batch

        jobs = actions[2, :]
        if self.mask_job_unreleased_batch[self.batch_idxes, jobs].any():
            raise ValueError("An action selected a job before its release time.")

        self.release_potential_jump_batch.zero_()
        previous_makespan = self.makespan_batch.clone()
        state, _, done_batch = super().step(actions)
        current_makespan = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.raw_reward_batch = previous_makespan - current_makespan
        self.compensated_reward_batch = (
            self.raw_reward_batch + self.release_potential_jump_batch
        )
        if self.dynamic_reward_mode == "arrival_compensated":
            self.reward_batch = self.compensated_reward_batch
        else:
            self.reward_batch = self.raw_reward_batch
        self.makespan_batch = current_makespan
        self._refresh_state_references()
        return state, self.reward_batch, done_batch

    def reset(self):
        if not self.dynamic_enabled:
            state = super().reset()
            self.mask_job_unreleased_batch = self.job_release_times_batch > self.time[:, None]
            state.mask_job_unreleased_batch = self.mask_job_unreleased_batch
            self.raw_reward_batch.zero_()
            self.compensated_reward_batch.zero_()
            self.release_potential_jump_batch.zero_()
            self.release_potential_jump_total_batch.zero_()
            return state

        self.ope_pre_adj_batch = copy.deepcopy(self._old_ope_pre_adj_batch)
        self.ope_sub_adj_batch = copy.deepcopy(self._old_ope_sub_adj_batch)
        state = super().reset()
        self.mask_job_unreleased_batch = copy.deepcopy(
            self._old_mask_job_unreleased_batch
        )
        self.visible_nums_opes = copy.deepcopy(self._old_visible_nums_opes)
        self.arrival_history = [[] for _ in range(self.batch_size)]
        self.raw_reward_batch.zero_()
        self.compensated_reward_batch.zero_()
        self.release_potential_jump_batch.zero_()
        self.release_potential_jump_total_batch.zero_()
        self.done = False
        self._refresh_state_references()
        return state
