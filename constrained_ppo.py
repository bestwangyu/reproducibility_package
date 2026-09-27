"""PPO with an explicit stability-cost constraint.

This module is intentionally separate from the legacy PPO implementation.  It
reuses the graph policy and stores an additional per-step cost trajectory.  The
constraint is expressed in the same units as the environment's
``stability_cost_mean_batch`` (mean cost per operation).
"""

import copy
import math

import torch
import torch.nn as nn

from mlp import MLPCritic
from PPO_model import HGNNScheduler


class ConstrainedMemory:
    """Memory compatible with HGNNScheduler.act plus normalized cost rewards."""

    def __init__(self):
        self.variable_batch = True
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.costs = []
        self.is_terminals = []
        self.action_indexes = []
        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []
        self.stability_action_context = []
        self.stability_global_context = []

    def clear_memory(self):
        for name in vars(self):
            value = getattr(self, name)
            if isinstance(value, list):
                value.clear()


def _discounted_returns(values, terminals, gamma, device):
    """Return a [num_envs, num_steps] tensor without normalizing it."""
    stacked_values = torch.stack(values, dim=0).to(device=device, dtype=torch.float64)
    stacked_terminals = torch.stack(terminals, dim=0).to(device=device)
    num_steps, num_envs = stacked_values.shape
    result = torch.zeros_like(stacked_values)
    running = torch.zeros(num_envs, dtype=torch.float64, device=device)
    for step in range(num_steps - 1, -1, -1):
        running = torch.where(
            stacked_terminals[step].bool(), torch.zeros_like(running), running
        )
        running = stacked_values[step] + gamma * running
        result[step] = running
    return result.transpose(0, 1)


def clipped_lagrangian_policy_loss(
    ratios, reward_advantage, cost_advantage, eps_clip, lambda_value, reward_coeff=1.0
):
    """Conservative PPO loss for reward maximization and cost minimization."""
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative.")
    clipped_ratios = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
    reward_objective = torch.min(
        ratios * reward_advantage, clipped_ratios * reward_advantage
    )
    # Cost is minimized, so use an upper clipped surrogate.  Using min here
    # gives the wrong conservative bound when the cost advantage is negative.
    cost_objective = torch.max(
        ratios * cost_advantage, clipped_ratios * cost_advantage
    )
    loss = (
        -reward_coeff * reward_objective + lambda_value * cost_objective
    ) / (1.0 + lambda_value)
    return loss, reward_objective, cost_objective


class ConstrainedPPO:
    """Lagrangian PPO with a separate value function for stability cost."""

    def __init__(self, model_paras, train_paras, num_envs):
        self.device = model_paras["device"]
        self.lr = float(train_paras["lr"])
        self.betas = tuple(train_paras["betas"])
        self.gamma = float(train_paras["gamma"])
        self.eps_clip = float(train_paras["eps_clip"])
        self.K_epochs = int(train_paras["K_epochs"])
        self.A_coeff = float(train_paras["A_coeff"])
        self.vf_coeff = float(train_paras["vf_coeff"])
        self.entropy_coeff = float(train_paras["entropy_coeff"])
        self.cost_vf_coeff = float(train_paras.get("cost_vf_coeff", self.vf_coeff))
        self.cost_advantage_mode = train_paras.get(
            "cost_advantage_mode", "critic"
        )
        if self.cost_advantage_mode not in ("critic", "batch_centered"):
            raise ValueError(
                "cost_advantage_mode must be critic or batch_centered."
            )
        self.minibatch_size = int(train_paras.get("minibatch_size", 512))
        self.num_envs = int(num_envs)
        self.lambda_value = float(train_paras.get("lambda_init", 0.0))
        self.lambda_lr = float(train_paras.get("lambda_lr", 0.05))
        self.lambda_max = float(train_paras.get("lambda_max", 10.0))
        self.budget = float(train_paras.get("stability_budget", 1.0))
        if not math.isfinite(self.budget) or self.budget < 0:
            raise ValueError("stability_budget must be finite and non-negative.")
        if self.lambda_value < 0 or self.lambda_lr < 0 or self.lambda_max < 0:
            raise ValueError("Lambda parameters must be non-negative.")

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.cost_critic = MLPCritic(
            self.policy.n_hidden_critic,
            self.policy.critic_dim,
            self.policy.n_latent_critic,
            1,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )
        self.cost_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(),
            lr=float(train_paras.get("cost_lr", self.lr)),
            betas=self.betas,
        )
        self.MseLoss = nn.MSELoss()

    def update_lambda(self, episode_cost):
        """Projected dual ascent for ``E[cost] <= budget``."""
        episode_cost = float(episode_cost)
        if not math.isfinite(episode_cost):
            raise ValueError("episode_cost must be finite.")
        self.lambda_value = min(
            self.lambda_max,
            max(0.0, self.lambda_value + self.lambda_lr * (episode_cost - self.budget)),
        )
        return self.lambda_value

    def update(self, memory, env_paras, train_paras=None):
        if not memory.rewards or len(memory.rewards) != len(memory.costs):
            raise ValueError("Reward and stability-cost trajectories must have equal length.")
        device = env_paras["device"]
        minibatch_size = self.minibatch_size if train_paras is None else int(
            train_paras.get("minibatch_size", self.minibatch_size)
        )
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")

        # Dynamic instances may finish at different decision points.  State
        # tensors are therefore stored with a shrinking active batch and must
        # be concatenated by time step rather than stacked into a rectangle.
        old_ope_ma_adj = torch.cat(memory.ope_ma_adj, dim=0)
        old_ope_pre_adj = torch.cat(memory.ope_pre_adj, dim=0)
        old_ope_sub_adj = torch.cat(memory.ope_sub_adj, dim=0)
        old_raw_opes = torch.cat(memory.raw_opes, dim=0)
        old_raw_mas = torch.cat(memory.raw_mas, dim=0)
        old_proc_time = torch.cat(memory.proc_time, dim=0)
        old_jobs_gather = torch.cat(memory.jobs_gather, dim=0)
        old_eligible = torch.cat(memory.eligible, dim=0)
        old_nums_opes = torch.cat(memory.nums_opes, dim=0)
        old_logprobs = torch.cat(memory.logprobs, dim=0)
        old_action_envs = torch.cat(memory.action_indexes, dim=0)
        old_stability_action_context = None
        old_stability_global_context = None
        if self.policy.stability_context_dim:
            old_stability_action_context = torch.cat(
                memory.stability_action_context, dim=0
            )
            old_stability_global_context = torch.cat(
                memory.stability_global_context, dim=0
            )

        reward_returns = _discounted_returns(
            memory.rewards, memory.is_terminals, self.gamma, device
        )
        cost_returns = _discounted_returns(
            memory.costs, memory.is_terminals, self.gamma, device
        )
        reward_returns = (reward_returns - reward_returns.mean(dim=1, keepdim=True)) / (
            reward_returns.std(dim=1, keepdim=True) + 1e-5
        )
        active_reward_returns = []
        active_cost_returns = []
        active_centered_cost_advantages = []
        for step, batch_indexes in enumerate(memory.batch_idxes):
            indexes = batch_indexes.to(device=device, dtype=torch.long)
            active_reward_returns.append(reward_returns[indexes, step])
            step_cost_returns = cost_returns[indexes, step]
            active_cost_returns.append(step_cost_returns)
            centered = step_cost_returns - step_cost_returns.mean()
            if centered.numel() > 1:
                centered = centered / (
                    centered.std(unbiased=False) + 1e-5
                )
            active_centered_cost_advantages.append(centered)
        reward_returns = torch.cat(active_reward_returns, dim=0)
        cost_returns = torch.cat(active_cost_returns, dim=0)
        centered_cost_advantages = torch.cat(
            active_centered_cost_advantages, dim=0
        )
        old_logprobs = old_logprobs.to(device=device)
        old_action_envs = old_action_envs.to(device=device)
        full_batch_size = old_ope_ma_adj.size(0)
        total_loss = 0.0
        total_cost_loss = 0.0
        updates = 0

        for _ in range(self.K_epochs):
            for start_idx in range(0, full_batch_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, full_batch_size)
                sl = slice(start_idx, end_idx)
                evaluated = self.policy.evaluate(
                    old_ope_ma_adj[sl], old_ope_pre_adj[sl], old_ope_sub_adj[sl],
                    old_raw_opes[sl], old_raw_mas[sl], old_proc_time[sl],
                    old_jobs_gather[sl], old_eligible[sl], old_nums_opes[sl],
                    old_action_envs[sl],
                    None if old_stability_action_context is None else old_stability_action_context[sl],
                    None if old_stability_global_context is None else old_stability_global_context[sl],
                    return_features=True,
                )
                logprobs, state_values, entropy, pooled = evaluated
                # Keep the cost baseline separate from the policy/encoder graph.
                cost_values = self.cost_critic(pooled.detach()).squeeze().double()
                ratios = torch.exp(logprobs - old_logprobs[sl].detach())
                reward_advantage = reward_returns[sl] - state_values.detach()
                if self.cost_advantage_mode == "batch_centered":
                    cost_advantage = centered_cost_advantages[sl]
                else:
                    cost_advantage = cost_returns[sl] - cost_values.detach()
                policy_loss, _, _ = clipped_lagrangian_policy_loss(
                    ratios,
                    reward_advantage,
                    cost_advantage,
                    self.eps_clip,
                    self.lambda_value,
                    self.A_coeff,
                )
                reward_value_loss = self.vf_coeff * self.MseLoss(
                    state_values, reward_returns[sl]
                )
                cost_value_loss = self.cost_vf_coeff * self.MseLoss(
                    cost_values, cost_returns[sl]
                )
                loss = (
                    policy_loss + reward_value_loss - self.entropy_coeff * entropy
                ).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.cost_optimizer.zero_grad()
                cost_value_loss.backward()
                self.cost_optimizer.step()
                total_loss += float(loss.detach().item())
                total_cost_loss += float(cost_value_loss.detach().item())
                updates += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        return {
            "loss": total_loss / max(1, updates),
            "cost_value_loss": total_cost_loss / max(1, updates),
            "reward_return_mean": float(reward_returns.mean().item()),
            "cost_return_mean": float(cost_returns.mean().item()),
            "lambda": float(self.lambda_value),
        }
