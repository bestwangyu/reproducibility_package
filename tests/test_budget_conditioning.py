import copy
import json
import os
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
from PPO_model import (
    HGNNScheduler,
    Memory,
    load_compatible_policy_checkpoint,
)
from tests.test_composite_dynamic_fjsp_env import run_to_completion
from tests.test_dynamic_fjsp_env import INSTANCE, environment_parameters


PROJECT_DIR = Path(__file__).resolve().parents[1]


def model_parameters(context_dim):
    config = json.loads((PROJECT_DIR / "config.json").read_text())
    parameters = copy.deepcopy(config["model_paras"])
    parameters["device"] = torch.device("cpu")
    parameters["actor_in_dim"] = (
        parameters["out_size_ma"] * 2 + parameters["out_size_ope"] * 2
    )
    parameters["critic_in_dim"] = (
        parameters["out_size_ma"] + parameters["out_size_ope"]
    )
    parameters["stability_context_dim"] = context_dim
    return parameters


class BudgetConditioningTest(unittest.TestCase):
    def test_zero_context_mode_records_only_zero_context(self):
        arrival = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260808,
            scenario_id="zero_context",
        )
        environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=arrival,
            breakdown_scenario=MachineBreakdownScenario.static(5),
        )
        policy = HGNNScheduler(model_parameters(3))
        policy.set_stability_context_mode("zero")
        memory = Memory()
        probabilities, _, _ = policy.get_action_prob(
            environment.state,
            memory,
            flag_sample=False,
            flag_train=True,
        )
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertEqual(len(memory.stability_action_context), 1)
        self.assertEqual(len(memory.stability_global_context), 1)
        self.assertEqual(
            torch.count_nonzero(memory.stability_action_context[0]), 0
        )
        self.assertEqual(
            torch.count_nonzero(memory.stability_global_context[0]), 0
        )

    def test_invalid_context_mode_is_rejected(self):
        policy = HGNNScheduler(model_parameters(3))
        with self.assertRaises(ValueError):
            policy.set_stability_context_mode("invalid")

    def test_legacy_checkpoint_migration_preserves_action_probabilities(self):
        arrival = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260731,
            scenario_id="budget_conditioning",
        )
        reference_environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=arrival,
            breakdown_scenario=MachineBreakdownScenario.static(5),
        )
        run_to_completion(reference_environment)
        reference = reference_environment.schedules_batch[0].detach().clone()
        breakdown = MachineBreakdownScenario(
            num_machines=5,
            events=tuple(
                MachineBreakdownEvent(machine, 1 + machine, 2)
                for machine in range(5)
            ),
            seed=20260731,
            scenario_id="budget_conditioning",
        )
        environment = CompositeDynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=copy.deepcopy(environment_parameters()),
            data_source="file",
            arrival_scenario=arrival,
            breakdown_scenario=breakdown,
            reference_schedules=reference,
            stability_budget=1.0,
        )

        checkpoint = torch.load(
            str(PROJECT_DIR / "checkpoints/base/reference_save_10_5.pt"), map_location="cpu"
        )
        legacy = HGNNScheduler(model_parameters(0))
        legacy.load_state_dict(checkpoint)
        conditioned = HGNNScheduler(model_parameters(3))
        report = load_compatible_policy_checkpoint(conditioned, checkpoint)
        self.assertTrue(report["migrated"])
        self.assertEqual(
            set(report["expanded_keys"]),
            {"actor.linears.0.weight", "critic.linears.0.weight"},
        )
        self.assertEqual(
            torch.count_nonzero(conditioned.actor.linears[0].weight[:, -3:]),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(conditioned.critic.linears[0].weight[:, -3:]),
            0,
        )

        probability_errors = []
        while not bool(environment.done_batch.all().item()):
            legacy_probabilities, legacy_steps, _ = legacy.get_action_prob(
                environment.state, Memory(), flag_sample=False, flag_train=False
            )
            conditioned_probabilities, conditioned_steps, _ = (
                conditioned.get_action_prob(
                    environment.state,
                    Memory(),
                    flag_sample=False,
                    flag_train=False,
                )
            )
            probability_errors.append(
                float(
                    torch.max(
                        torch.abs(
                            legacy_probabilities - conditioned_probabilities
                        )
                    ).item()
                )
            )
            self.assertTrue(torch.equal(legacy_steps, conditioned_steps))
            self.assertTrue(torch.isfinite(conditioned_probabilities).all())
            self.assertTrue(
                torch.allclose(
                    conditioned_probabilities.sum(dim=1),
                    torch.ones(1),
                    atol=1e-6,
                    rtol=0,
                )
            )
            action_index = conditioned_probabilities.argmax(dim=1)
            jobs = (action_index % environment.num_jobs).long()
            machines = (action_index / environment.num_jobs).long()
            operations = conditioned_steps[environment.batch_idxes, jobs]
            actions = torch.stack((operations, machines, jobs), dim=1).t()
            environment.step(actions)

        self.assertLess(max(probability_errors), 1e-7)
        self.assertTrue(environment.validate_gantt()[0])


if __name__ == "__main__":
    unittest.main()
