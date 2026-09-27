import os
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fjsp-drl-matplotlib")

from tests.gym_compat import install_gym_stub_if_needed

install_gym_stub_if_needed()

import torch

from dispatching_rules import RULES, select_dispatching_actions
from dynamic.arrivals import JobArrivalScenario
from dynamic.breakdowns import MachineBreakdownEvent, MachineBreakdownScenario
from env.composite_dynamic_fjsp_env import CompositeDynamicFJSPEnv


PROJECT_DIR = Path(__file__).resolve().parents[1]
INSTANCE = PROJECT_DIR / "data_dev" / "1005" / "10j_5m_001.fjs"


class FakeEnvironment:
    def __init__(self):
        self.num_jobs = 3
        self.num_mas = 2
        self.batch_idxes = torch.tensor([0])
        self.time = torch.tensor([0.0])
        self.ope_step_batch = torch.tensor([[0, 2, 4]])
        self.end_ope_biases_batch = torch.tensor([[1, 3, 6]])
        self.mask_job_procing_batch = torch.tensor([[False, False, False]])
        self.mask_job_finish_batch = torch.tensor([[False, False, False]])
        self.mask_job_unreleased_batch = torch.tensor([[False, False, False]])
        self.mask_ma_procing_batch = torch.tensor([[False, False]])
        self.mask_ma_failed_batch = torch.tensor([[False, False]])
        self.job_release_times_batch = torch.tensor([[0.0, 1.0, 2.0]])
        self.ope_ma_adj_batch = torch.ones((1, 7, 2), dtype=torch.long)
        self.proc_times_batch = torch.tensor(
            [
                [
                    [4.0, 5.0],
                    [2.0, 3.0],
                    [1.0, 6.0],
                    [1.0, 1.0],
                    [3.0, 2.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                ]
            ]
        )


class DispatchingRuleSelectionTest(unittest.TestCase):
    def action(self, environment, rule):
        return tuple(
            int(value) for value in select_dispatching_actions(environment, rule)[:, 0]
        )

    def test_fifo_uses_earliest_release_then_shortest_machine(self):
        self.assertEqual(self.action(FakeEnvironment(), "FIFO"), (0, 0, 0))

    def test_spt_selects_global_shortest_current_pair(self):
        self.assertEqual(self.action(FakeEnvironment(), "SPT"), (2, 0, 1))

    def test_mwkr_selects_largest_remaining_work(self):
        self.assertEqual(self.action(FakeEnvironment(), "MWKR"), (4, 1, 2))

    def test_mor_selects_most_remaining_operations(self):
        self.assertEqual(self.action(FakeEnvironment(), "MOR"), (4, 1, 2))

    def test_unreleased_job_and_failed_machine_are_never_selected(self):
        environment = FakeEnvironment()
        environment.mask_job_unreleased_batch[0, 1] = True
        environment.mask_ma_failed_batch[0, 0] = True
        self.assertEqual(self.action(environment, "SPT"), (4, 1, 2))

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            select_dispatching_actions(FakeEnvironment(), "UNKNOWN")


class DispatchingRuleIntegrationTest(unittest.TestCase):
    def make_environment(self):
        parameters = {
            "num_jobs": 10,
            "num_mas": 5,
            "batch_size": 1,
            "ope_feat_dim": 6,
            "ma_feat_dim": 3,
            "show_mode": "print",
            "device": torch.device("cpu"),
            "dynamic_reward_mode": "raw",
        }
        arrivals = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260825,
            scenario_id="dispatch-rule-test-arrival",
        )
        breakdowns = MachineBreakdownScenario(
            num_machines=5,
            events=(MachineBreakdownEvent(0, 5.0, 2.0),),
            seed=20260826,
            scenario_id="dispatch-rule-test-breakdown",
        )
        return CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=parameters,
            data_source="file",
            arrival_scenario=arrivals,
            breakdown_scenario=breakdowns,
        )

    def test_all_rules_finish_with_feasible_schedules(self):
        for rule in RULES:
            environment = self.make_environment()
            steps = 0
            while not bool(environment.done_batch.all().item()):
                environment.step(select_dispatching_actions(environment, rule))
                steps += 1
                self.assertLessEqual(steps, environment.num_opes + 1)
            self.assertTrue(environment.validate_gantt()[0], msg=rule)
            for job in range(6, 10):
                operation = int(environment.num_ope_biases_batch[0, job].item())
                start = float(environment.schedules_batch[0, operation, 2].item())
                self.assertGreaterEqual(start + 1e-5, (job - 5) * 2, msg=rule)


if __name__ == "__main__":
    unittest.main()
