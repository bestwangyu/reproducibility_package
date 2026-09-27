import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fjsp-drl-matplotlib")

from tests.gym_compat import install_gym_stub_if_needed

install_gym_stub_if_needed()

import torch

from dynamic.arrivals import JobArrivalScenario
from dynamic.breakdowns import (
    MachineBreakdownEvent,
    MachineBreakdownScenario,
)
from env.composite_dynamic_fjsp_env import CompositeDynamicFJSPEnv
from env.dynamic_fjsp_env import DynamicFJSPEnv
from PPO_model import HGNNScheduler, Memory
from tests.test_dynamic_fjsp_env import (
    INSTANCE,
    environment_parameters,
    shortest_processing_time_action,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def run_to_completion(environment, max_steps=200):
    steps = 0
    while not bool(environment.done_batch.all().item()):
        environment.step(shortest_processing_time_action(environment))
        steps += 1
        if steps > max_steps:
            raise AssertionError("Environment did not finish within the step limit.")
    return steps


class CompositeDynamicFJSPEnvironmentTest(unittest.TestCase):
    def test_disabled_breakdowns_match_arrival_environment_step_by_step(self):
        arrival = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260730,
            scenario_id="arrival_only",
        )
        parameters = environment_parameters()
        arrival_environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=parameters,
            data_source="file",
            arrival_scenario=arrival,
        )
        composite_environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=copy.deepcopy(parameters),
            data_source="file",
            arrival_scenario=arrival,
            breakdown_scenario=MachineBreakdownScenario.static(5),
        )

        while not bool(arrival_environment.done_batch.all().item()):
            arrival_action = shortest_processing_time_action(arrival_environment)
            composite_action = shortest_processing_time_action(
                composite_environment
            )
            self.assertTrue(torch.equal(arrival_action, composite_action))
            _, arrival_reward, _ = arrival_environment.step(arrival_action)
            _, composite_reward, _ = composite_environment.step(
                composite_action
            )
            self.assertTrue(torch.equal(arrival_reward, composite_reward))
            self.assertTrue(
                torch.equal(
                    arrival_environment.time,
                    composite_environment.time,
                )
            )
            self.assertTrue(
                torch.equal(
                    arrival_environment.schedules_batch,
                    composite_environment.schedules_batch,
                )
            )

        self.assertTrue(arrival_environment.validate_gantt()[0])
        self.assertTrue(composite_environment.validate_gantt()[0])
        self.assertEqual(composite_environment.breakdown_history, [[]])

    def test_last_dispatched_operation_includes_future_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            instance_path = Path(directory) / "1j_1m.fjs"
            instance_path.write_text(
                "1 1 1\n1 1 1 10\n",
                encoding="utf-8",
            )
            parameters = {
                "num_jobs": 1,
                "num_mas": 1,
                "batch_size": 1,
                "ope_feat_dim": 6,
                "ma_feat_dim": 3,
                "show_mode": "print",
                "device": torch.device("cpu"),
            }
            breakdown = MachineBreakdownScenario(
                num_machines=1,
                events=(MachineBreakdownEvent(0, 2, 3),),
                seed=20260730,
                scenario_id="tail_interruption",
            )
            environment = CompositeDynamicFJSPEnv(
                case=[str(instance_path)],
                env_paras=parameters,
                data_source="file",
                arrival_scenario=JobArrivalScenario.static(1),
                breakdown_scenario=breakdown,
            )
            environment.step(torch.tensor([[0], [0], [0]]))

            self.assertTrue(environment.done_batch.all().item())
            self.assertAlmostEqual(environment.makespan_batch.item(), 13.0)
            self.assertAlmostEqual(
                environment.operation_interruption_batch[0, 0].item(), 3.0
            )
            self.assertAlmostEqual(
                environment.schedules_batch[0, 0, 3].item(), 13.0
            )
            self.assertTrue(environment.validate_gantt()[0])
            self.assertTrue(
                environment.breakdown_history[0][0][
                    "finalized_after_last_dispatch"
                ]
            )

    def test_breakdown_at_machine_completion_is_processed_before_next_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            instance_path = Path(directory) / "2j_1m.fjs"
            instance_path.write_text(
                "2 1 1\n1 1 1 4\n1 1 1 1\n",
                encoding="utf-8",
            )
            parameters = {
                "num_jobs": 2,
                "num_mas": 1,
                "batch_size": 1,
                "ope_feat_dim": 6,
                "ma_feat_dim": 3,
                "show_mode": "print",
                "device": torch.device("cpu"),
            }
            breakdown = MachineBreakdownScenario(
                num_machines=1,
                events=(MachineBreakdownEvent(0, 4.0000004, 1),),
                seed=20260730,
                scenario_id="completion_boundary",
            )
            environment = CompositeDynamicFJSPEnv(
                case=[str(instance_path)],
                env_paras=parameters,
                data_source="file",
                arrival_scenario=JobArrivalScenario.static(2),
                breakdown_scenario=breakdown,
            )

            environment.step(torch.tensor([[0], [0], [0]]))

            self.assertAlmostEqual(environment.time.item(), 5.0000004, places=5)
            self.assertFalse(environment.mask_ma_failed_batch[0, 0].item())
            self.assertFalse(environment.mask_ma_procing_batch[0, 0].item())
            self.assertTrue(environment.breakdown_history[0][0]["interrupted"] is False)

            environment.step(torch.tensor([[1], [0], [1]]))
            self.assertGreaterEqual(
                environment.schedules_batch[0, 1, 2].item(),
                breakdown.events[0].repair_end_time - 1e-5,
            )
            self.assertTrue(environment.validate_gantt()[0])

    def test_policy_cannot_select_failed_machine(self):
        breakdown = MachineBreakdownScenario(
            num_machines=5,
            events=(MachineBreakdownEvent(0, 1, 10),),
            seed=20260730,
            scenario_id="mask_test",
        )
        environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=JobArrivalScenario.static(10),
            breakdown_scenario=breakdown,
        )
        for _ in range(environment.num_opes):
            if environment.mask_ma_failed_batch[0, 0]:
                break
            environment.step(shortest_processing_time_action(environment))
        self.assertTrue(environment.mask_ma_failed_batch[0, 0].item())

        config = json.loads((PROJECT_DIR / "config.json").read_text())
        model_parameters = copy.deepcopy(config["model_paras"])
        model_parameters["device"] = torch.device("cpu")
        model_parameters["actor_in_dim"] = (
            model_parameters["out_size_ma"] * 2
            + model_parameters["out_size_ope"] * 2
        )
        model_parameters["critic_in_dim"] = (
            model_parameters["out_size_ma"] + model_parameters["out_size_ope"]
        )
        policy = HGNNScheduler(model_parameters)
        action = policy.act(
            environment.state,
            Memory(),
            environment.done_batch,
            flag_sample=False,
            flag_train=False,
        )
        self.assertNotEqual(int(action[1, 0].item()), 0)
        run_to_completion(environment)
        self.assertTrue(environment.validate_gantt()[0])

    def test_composite_events_are_feasible_and_resettable(self):
        arrival = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260730,
            scenario_id="composite",
        )
        breakdown = MachineBreakdownScenario(
            num_machines=5,
            events=tuple(
                MachineBreakdownEvent(machine, 1 + machine, 2)
                for machine in range(5)
            ),
            seed=20260730,
            scenario_id="composite",
        )
        environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=arrival,
            breakdown_scenario=breakdown,
        )
        steps = run_to_completion(environment)
        first_makespan = environment.makespan_batch.clone()
        self.assertGreater(steps, 0)
        self.assertTrue(environment.validate_gantt()[0])
        self.assertEqual(
            sum(len(event["jobs"]) for event in environment.arrival_history[0]),
            4,
        )
        self.assertGreater(len(environment.breakdown_history[0]), 0)
        self.assertGreater(
            environment.operation_interruption_batch.sum().item(), 0
        )

        environment.reset()
        self.assertEqual(environment.arrival_history, [[]])
        self.assertEqual(environment.breakdown_history, [[]])
        self.assertEqual(environment.mask_ma_failed_batch.sum().item(), 0)
        run_to_completion(environment)
        self.assertTrue(torch.equal(first_makespan, environment.makespan_batch))
        self.assertTrue(environment.validate_gantt()[0])


if __name__ == "__main__":
    unittest.main()
