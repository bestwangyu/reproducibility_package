import copy
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fjsp-drl-matplotlib")

from tests.gym_compat import install_gym_stub_if_needed

install_gym_stub_if_needed()

import torch

from dynamic.arrivals import JobArrivalScenario
from dynamic.breakdowns import (
    MachineBreakdownEvent,
    MachineBreakdownScenario,
)
from dynamic.stability import calculate_schedule_stability
from env.composite_dynamic_fjsp_env import CompositeDynamicFJSPEnv
from tests.test_composite_dynamic_fjsp_env import run_to_completion
from tests.test_dynamic_fjsp_env import (
    INSTANCE,
    environment_parameters,
    shortest_processing_time_action,
)


def reference_schedule(arrival):
    environment = CompositeDynamicFJSPEnv(
        case=[str(INSTANCE)],
        env_paras=environment_parameters(),
        data_source="file",
        arrival_scenario=arrival,
        breakdown_scenario=MachineBreakdownScenario.static(5),
    )
    run_to_completion(environment)
    return environment.schedules_batch[0].detach().clone()


class OnlineStabilityTest(unittest.TestCase):
    def setUp(self):
        self.arrival = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260731,
            scenario_id="online_stability",
        )
        self.reference = reference_schedule(self.arrival)

    def make_environment(self, breakdown):
        return CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=copy.deepcopy(environment_parameters()),
            data_source="file",
            arrival_scenario=self.arrival,
            breakdown_scenario=breakdown,
            reference_schedules=self.reference,
            stability_budget=1.0,
        )

    def test_online_cost_matches_offline_schedule_comparison(self):
        breakdown = MachineBreakdownScenario(
            num_machines=5,
            events=tuple(
                MachineBreakdownEvent(machine, 1 + machine, 2)
                for machine in range(5)
            ),
            seed=20260731,
            scenario_id="online_stability",
        )
        environment = self.make_environment(breakdown)
        candidate_errors = []
        while not bool(environment.done_batch.all().item()):
            action = shortest_processing_time_action(environment)
            batch_index = int(environment.batch_idxes[0].item())
            operation = int(action[0, 0].item())
            machine = int(action[1, 0].item())
            job = int(action[2, 0].item())
            predicted_cost = float(
                environment.candidate_stability_cost_batch[
                    batch_index, job, machine
                ].item()
            )
            environment.step(action)
            charged_cost = float(
                environment.stability_cost_operation_batch[
                    batch_index, operation
                ].item()
            )
            candidate_errors.append(abs(predicted_cost - charged_cost))
        self.assertLess(max(candidate_errors), 1e-6)
        num_operations = int(environment.nums_opes[0].item())
        offline = calculate_schedule_stability(
            self.reference,
            environment.schedules_batch[0],
            num_operations=num_operations,
        )

        self.assertEqual(
            int(environment.stability_accounted_batch[0].sum().item()),
            num_operations,
        )
        self.assertTrue(
            torch.allclose(
                environment.stability_cost_operation_batch[0, :num_operations],
                torch.tensor(
                    [item.stability_cost for item in offline.operations],
                    dtype=environment.time.dtype,
                    device=environment.time.device,
                ),
                atol=1e-5,
                rtol=0,
            )
        )
        self.assertAlmostEqual(
            environment.stability_cost_sum_batch[0].item(),
            offline.stability_cost_sum,
            places=4,
        )
        self.assertAlmostEqual(
            environment.stability_cost_mean_batch[0].item(),
            offline.stability_cost_mean,
            places=5,
        )
        accounting_error = abs(
            environment.stability_cost_mean_batch[0]
            + environment.stability_remaining_batch[0]
            - environment.stability_budget_batch[0]
            - environment.stability_violation_batch[0]
        )
        self.assertLess(accounting_error.item(), 1e-6)

        environment.reset()
        self.assertEqual(environment.stability_accounted_batch.sum().item(), 0)
        self.assertEqual(environment.stability_cost_sum_batch.sum().item(), 0)
        self.assertTrue(
            torch.equal(
                environment.stability_remaining_batch,
                environment.stability_budget_batch,
            )
        )

    def test_reference_trajectory_has_zero_online_cost(self):
        environment = self.make_environment(MachineBreakdownScenario.static(5))
        run_to_completion(environment)
        self.assertEqual(environment.stability_cost_sum_batch[0].item(), 0)
        self.assertEqual(environment.stability_violation_batch[0].item(), 0)
        self.assertEqual(environment.stability_remaining_batch[0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
