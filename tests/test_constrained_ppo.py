import sys
import unittest
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from constrained_ppo import (
    ConstrainedPPO,
    _discounted_returns,
    clipped_lagrangian_policy_loss,
)


class ConstrainedPPOTest(unittest.TestCase):
    def test_discounted_returns_keep_terminal_reward(self):
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        terminals = [torch.tensor([False, False]), torch.tensor([True, True])]
        result = _discounted_returns(values, terminals, 1.0, torch.device("cpu"))
        self.assertTrue(torch.equal(result, torch.tensor([[4.0, 3.0], [6.0, 4.0]], dtype=torch.float64)))

    def test_lambda_projection(self):
        train = {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "gamma": 1.0,
            "K_epochs": 1,
            "eps_clip": 0.2,
            "A_coeff": 1.0,
            "vf_coeff": 0.5,
            "entropy_coeff": 0.0,
            "minibatch_size": 2,
            "stability_budget": 1.0,
            "lambda_lr": 0.5,
            "lambda_max": 2.0,
        }
        model = {
            "device": torch.device("cpu"),
            "in_size_ma": 3,
            "out_size_ma": 2,
            "in_size_ope": 6,
            "out_size_ope": 2,
            "hidden_size_ope": 4,
            "num_heads": [1],
            "dropout": 0.0,
            "n_latent_actor": 4,
            "n_latent_critic": 4,
            "n_hidden_actor": 2,
            "n_hidden_critic": 2,
            "action_dim": 1,
            "actor_in_dim": 8,
            "critic_in_dim": 4,
            "stability_context_dim": 3,
        }
        constrained = ConstrainedPPO(model, train, num_envs=1)
        self.assertEqual(constrained.update_lambda(3.0), 1.0)
        self.assertEqual(constrained.update_lambda(-1.0), 0.0)
        self.assertEqual(constrained.update_lambda(100.0), 2.0)

    def test_cost_surrogate_uses_conservative_upper_clip(self):
        ratios = torch.tensor([2.0, 2.0])
        reward_advantage = torch.zeros(2)
        cost_advantage = torch.tensor([1.0, -1.0])
        loss, _, cost_objective = clipped_lagrangian_policy_loss(
            ratios, reward_advantage, cost_advantage, 0.2, 1.0
        )
        self.assertTrue(torch.allclose(cost_objective, torch.tensor([2.0, -1.2])))
        self.assertTrue(torch.allclose(loss, cost_objective / 2.0))


if __name__ == "__main__":
    unittest.main()
