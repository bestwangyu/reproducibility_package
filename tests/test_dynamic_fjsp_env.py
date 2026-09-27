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
from env.dynamic_fjsp_env import DynamicFJSPEnv
from env.fjsp_env import FJSPEnv
from PPO_model import HGNNScheduler, Memory, PPO


PROJECT_DIR = Path(__file__).resolve().parents[1]
INSTANCE = PROJECT_DIR / "data_dev" / "1005" / "10j_5m_001.fjs"


def environment_parameters():
    return {
        "num_jobs": 10,
        "num_mas": 5,
        "batch_size": 1,
        "ope_feat_dim": 6,
        "ma_feat_dim": 3,
        "show_mode": "print",
        "device": torch.device("cpu"),
    }


def shortest_processing_time_action(environment):
    opes = []
    machines = []
    jobs = []
    unreleased = getattr(environment, "mask_job_unreleased_batch", None)

    for batch_index in environment.batch_idxes.tolist():
        best = None
        for job in range(environment.num_jobs):
            if environment.mask_job_procing_batch[batch_index, job]:
                continue
            if environment.mask_job_finish_batch[batch_index, job]:
                continue
            if unreleased is not None and unreleased[batch_index, job]:
                continue
            operation = int(environment.ope_step_batch[batch_index, job].item())
            for machine in range(environment.num_mas):
                process_time = float(
                    environment.proc_times_batch[
                        batch_index, operation, machine
                    ].item()
                )
                if process_time <= 0 or environment.mask_ma_procing_batch[
                    batch_index, machine
                ]:
                    continue
                candidate = (process_time, job, machine, operation)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise AssertionError("No eligible action found at a decision point.")
        _, job, machine, operation = best
        opes.append(operation)
        machines.append(machine)
        jobs.append(job)

    return torch.tensor([opes, machines, jobs], dtype=torch.long)


def run_to_completion(environment, max_steps=200):
    steps = 0
    while not bool(environment.done_batch.all().item()):
        environment.step(shortest_processing_time_action(environment))
        steps += 1
        if steps > max_steps:
            raise AssertionError("Environment did not finish within the step limit.")
    return steps


class DynamicFJSPEnvironmentTest(unittest.TestCase):
    def test_disabled_arrivals_match_static_environment_step_by_step(self):
        parameters = environment_parameters()
        static_environment = FJSPEnv(
            case=[str(INSTANCE)], env_paras=parameters, data_source="file"
        )
        dynamic_parameters = copy.deepcopy(parameters)
        dynamic_parameters["dynamic_reward_mode"] = "arrival_compensated"
        dynamic_environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=dynamic_parameters,
            data_source="file",
            arrival_scenario=JobArrivalScenario.static(num_jobs=10),
        )

        while not bool(static_environment.done_batch.all().item()):
            static_action = shortest_processing_time_action(static_environment)
            dynamic_action = shortest_processing_time_action(dynamic_environment)
            self.assertTrue(torch.equal(static_action, dynamic_action))
            _, static_reward, _ = static_environment.step(static_action)
            _, dynamic_reward, _ = dynamic_environment.step(dynamic_action)
            self.assertTrue(torch.equal(static_reward, dynamic_reward))
            self.assertTrue(
                torch.equal(
                    dynamic_environment.raw_reward_batch,
                    dynamic_environment.compensated_reward_batch,
                )
            )
            self.assertEqual(
                dynamic_environment.release_potential_jump_batch.sum().item(), 0
            )
            self.assertTrue(torch.equal(static_environment.time, dynamic_environment.time))
            self.assertTrue(
                torch.equal(
                    static_environment.mask_job_finish_batch,
                    dynamic_environment.mask_job_finish_batch,
                )
            )
            self.assertTrue(
                torch.equal(
                    static_environment.schedules_batch,
                    dynamic_environment.schedules_batch,
                )
            )

        self.assertTrue(static_environment.validate_gantt()[0])
        self.assertTrue(dynamic_environment.validate_gantt()[0])
        self.assertTrue(
            torch.equal(
                static_environment.makespan_batch,
                dynamic_environment.makespan_batch,
            )
        )

    def test_arrival_compensation_changes_only_rewards(self):
        scenario = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260728,
            scenario_id="reward_compensation_test",
        )
        raw_parameters = environment_parameters()
        raw_parameters["dynamic_reward_mode"] = "raw"
        compensated_parameters = environment_parameters()
        compensated_parameters["dynamic_reward_mode"] = "arrival_compensated"
        raw_environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=raw_parameters,
            data_source="file",
            arrival_scenario=scenario,
        )
        compensated_environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=compensated_parameters,
            data_source="file",
            arrival_scenario=scenario,
        )
        initial_makespan = raw_environment.makespan_batch.clone()
        raw_return = torch.zeros(1)
        compensated_return = torch.zeros(1)
        release_jump = torch.zeros(1)

        while not bool(raw_environment.done_batch.all().item()):
            raw_action = shortest_processing_time_action(raw_environment)
            compensated_action = shortest_processing_time_action(
                compensated_environment
            )
            self.assertTrue(torch.equal(raw_action, compensated_action))
            _, raw_reward, _ = raw_environment.step(raw_action)
            _, compensated_reward, _ = compensated_environment.step(
                compensated_action
            )

            self.assertTrue(
                torch.equal(
                    raw_environment.schedules_batch,
                    compensated_environment.schedules_batch,
                )
            )
            self.assertTrue(
                torch.equal(raw_environment.time, compensated_environment.time)
            )
            self.assertTrue(
                torch.allclose(
                    raw_reward, compensated_environment.raw_reward_batch
                )
            )
            self.assertTrue(
                torch.allclose(
                    compensated_reward,
                    compensated_environment.raw_reward_batch
                    + compensated_environment.release_potential_jump_batch,
                )
            )
            raw_return += raw_reward.cpu()
            compensated_return += compensated_reward.cpu()
            release_jump += (
                compensated_environment.release_potential_jump_batch.cpu()
            )

        self.assertTrue(raw_environment.validate_gantt()[0])
        self.assertTrue(compensated_environment.validate_gantt()[0])
        self.assertGreater(release_jump.item(), 0)
        self.assertTrue(
            torch.allclose(
                raw_return,
                initial_makespan.cpu()
                - raw_environment.makespan_batch.cpu(),
            )
        )
        self.assertTrue(
            torch.allclose(compensated_return, raw_return + release_jump)
        )
        self.assertTrue(
            torch.allclose(
                release_jump,
                compensated_environment.release_potential_jump_total_batch.cpu(),
            )
        )
        self.assertTrue(
            all(
                "potential_jump" in event
                for event in compensated_environment.arrival_history[0]
            )
        )

        compensated_environment.reset()
        self.assertEqual(
            compensated_environment.release_potential_jump_total_batch.sum().item(),
            0,
        )
        self.assertEqual(compensated_environment.arrival_history, [[]])

    def test_invalid_dynamic_reward_mode_is_rejected(self):
        parameters = environment_parameters()
        parameters["dynamic_reward_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "dynamic_reward_mode"):
            DynamicFJSPEnv(
                case=[str(INSTANCE)],
                env_paras=parameters,
                data_source="file",
                arrival_scenario=JobArrivalScenario.static(num_jobs=10),
            )

    def test_jobs_are_hidden_until_release_and_reset_hides_them_again(self):
        scenario = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260728,
            scenario_id="release_test",
        )
        environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=scenario,
        )

        self.assertEqual(environment.mask_job_unreleased_batch.sum().item(), 4)
        first_future_ope = int(environment.num_ope_biases_batch[0, 6].item())
        self.assertEqual(
            torch.count_nonzero(
                environment.proc_times_batch[0, first_future_ope:]
            ).item(),
            0,
        )

        steps = run_to_completion(environment)
        self.assertGreater(steps, 0)
        self.assertTrue(environment.validate_gantt()[0])
        self.assertEqual(environment.mask_job_unreleased_batch.sum().item(), 0)
        self.assertEqual(
            sum(
                len(event["jobs"])
                for event in environment.arrival_history[0]
            ),
            4,
        )

        for job in range(6, 10):
            first_operation = int(environment.num_ope_biases_batch[0, job].item())
            actual_start = float(
                environment.schedules_batch[0, first_operation, 2].item()
            )
            self.assertGreaterEqual(actual_start, scenario.release_times[job])

        environment.reset()
        self.assertEqual(environment.mask_job_unreleased_batch.sum().item(), 4)
        self.assertEqual(environment.arrival_history, [[]])
        self.assertEqual(
            torch.count_nonzero(
                environment.proc_times_batch[0, first_future_ope:]
            ).item(),
            0,
        )

    def test_graph_policy_cannot_select_an_unreleased_job(self):
        scenario = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260728,
            scenario_id="policy_mask_test",
        )
        environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_parameters(),
            data_source="file",
            arrival_scenario=scenario,
        )
        config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
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

        self.assertLess(int(action[2, 0].item()), scenario.initial_jobs)

    def test_dynamic_episode_supports_one_ppo_update(self):
        torch.manual_seed(20260728)
        config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
        environment_params = environment_parameters()
        environment_params["dynamic_reward_mode"] = "arrival_compensated"
        model_parameters = copy.deepcopy(config["model_paras"])
        model_parameters["device"] = torch.device("cpu")
        model_parameters["actor_in_dim"] = (
            model_parameters["out_size_ma"] * 2
            + model_parameters["out_size_ope"] * 2
        )
        model_parameters["critic_in_dim"] = (
            model_parameters["out_size_ma"] + model_parameters["out_size_ope"]
        )
        train_parameters = copy.deepcopy(config["train_paras"])
        train_parameters["minibatch_size"] = 512
        scenario = JobArrivalScenario(
            num_jobs=10,
            initial_jobs=6,
            release_times=(0, 0, 0, 0, 0, 0, 2, 4, 6, 8),
            seed=20260728,
            scenario_id="ppo_update_test",
        )
        environment = DynamicFJSPEnv(
            case=[str(INSTANCE)],
            env_paras=environment_params,
            data_source="file",
            arrival_scenario=scenario,
        )
        model = PPO(
            model_parameters,
            train_parameters,
            num_envs=1,
        )
        memory = Memory()

        while not bool(environment.done_batch.all().item()):
            with torch.no_grad():
                action = model.policy_old.act(
                    environment.state,
                    memory,
                    environment.done_batch,
                    flag_sample=True,
                    flag_train=True,
                )
            _, rewards, dones = environment.step(action)
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)

        loss, reward = model.update(memory, environment_params, train_parameters)
        self.assertTrue(environment.validate_gantt()[0])
        self.assertTrue(torch.isfinite(torch.tensor([loss, reward])).all().item())


if __name__ == "__main__":
    unittest.main()
