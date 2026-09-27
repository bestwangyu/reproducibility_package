import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from graph.hgnn import GATedge, MLPsim
from mlp import MLPCritic, MLPActor


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
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
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        
        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]
        del self.stability_action_context[:]
        del self.stability_global_context[:]
        

class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime

class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.base_actor_dim = model_paras["actor_in_dim"]
        self.base_critic_dim = model_paras["critic_in_dim"]
        self.stability_context_dim = int(
            model_paras.get("stability_context_dim", 0)
        )
        if self.stability_context_dim not in (0, 3):
            raise ValueError("stability_context_dim must be either 0 or 3.")
        self.set_stability_context_mode(
            model_paras.get("stability_context_mode", "full")
        )
        self.actor_dim = self.base_actor_dim + self.stability_context_dim
        self.critic_dim = self.base_critic_dim + self.stability_context_dim
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def set_stability_context_mode(self, mode):
        if mode not in ("full", "zero"):
            raise ValueError("stability_context_mode must be full or zero.")
        self.stability_context_mode = mode

    def get_stability_context(self, state, batch_idxes, eligible):
        if self.stability_context_dim == 0:
            return None, None
        if self.stability_context_mode == "zero":
            action_shape = eligible.transpose(1, 2).shape
            action_context = torch.zeros(
                action_shape + (self.stability_context_dim,),
                dtype=torch.float32,
                device=eligible.device,
            )
            global_context = torch.zeros(
                (len(batch_idxes), self.stability_context_dim),
                dtype=torch.float32,
                device=eligible.device,
            )
            return action_context, global_context
        required = (
            "stability_budget_batch",
            "stability_remaining_batch",
            "stability_violation_batch",
            "candidate_stability_cost_batch",
        )
        missing = [name for name in required if getattr(state, name, None) is None]
        if missing:
            raise ValueError(
                "Budget-conditioned policy requires state fields: {0}.".format(
                    ", ".join(missing)
                )
            )

        budgets = state.stability_budget_batch[batch_idxes]
        remaining = state.stability_remaining_batch[batch_idxes]
        violation = state.stability_violation_batch[batch_idxes]
        epsilon = 1e-8
        remaining_ratio = torch.where(
            budgets > epsilon,
            remaining / budgets.clamp_min(epsilon),
            torch.zeros_like(remaining),
        ).clamp(0.0, 1.0)
        violation_ratio = violation / (budgets + violation + epsilon)
        candidate_cost = state.candidate_stability_cost_batch[batch_idxes]
        if not bool(torch.isfinite(candidate_cost).all().item()):
            raise RuntimeError("Candidate stability costs must be finite.")
        candidate_scaled = candidate_cost / (1.0 + candidate_cost)

        action_shape = candidate_scaled.shape
        remaining_actions = remaining_ratio[:, None, None].expand(action_shape)
        violation_actions = violation_ratio[:, None, None].expand(action_shape)
        action_context = torch.stack(
            (remaining_actions, violation_actions, candidate_scaled), dim=-1
        ).transpose(1, 2)

        infinity = torch.full_like(candidate_scaled, float("inf"))
        eligible_cost = torch.where(eligible, candidate_scaled, infinity)
        minimum_cost = eligible_cost.flatten(1).min(dim=1)[0]
        minimum_cost = torch.where(
            torch.isfinite(minimum_cost), minimum_cost, torch.zeros_like(minimum_cost)
        )
        global_context = torch.stack(
            (remaining_ratio, violation_ratio, minimum_cost), dim=-1
        )
        return action_context, global_context

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes,
                       flag_sample=False, flag_train=False, force_variable_opes=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if force_variable_opes or (not flag_sample and not flag_train):
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        unreleased_jobs = getattr(state, "mask_job_unreleased_batch", None)
        force_variable_opes = (
            unreleased_jobs is not None
            and bool(unreleased_jobs[batch_idxes].any().item())
        )
        features = self.get_normalized(
            raw_opes,
            raw_mas,
            proc_time,
            batch_idxes,
            nums_opes,
            flag_sample,
            flag_train,
            force_variable_opes=force_variable_opes,
        )
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)
            features = (features[0], h_mas, features[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if force_variable_opes or (not flag_sample and not flag_train):
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)
        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_blocked = state.mask_ma_procing_batch[batch_idxes]
        failed_machines = getattr(state, "mask_ma_failed_batch", None)
        if failed_machines is not None:
            ma_blocked = ma_blocked | failed_machines[batch_idxes]
        ma_eligible = ~ma_blocked.unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_blocked = (state.mask_job_procing_batch[batch_idxes] +
                       state.mask_job_finish_batch[batch_idxes])
        if unreleased_jobs is not None:
            job_blocked = job_blocked + unreleased_jobs[batch_idxes]
        job_eligible = ~job_blocked[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        stability_action_context, stability_global_context = (
            self.get_stability_context(state, batch_idxes, eligible)
        )
        if self.stability_context_dim:
            h_actions = torch.cat((h_actions, stability_action_context), dim=-1)
            h_pooled = torch.cat((h_pooled, stability_global_context), dim=-1)
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train == True:
            # Dynamic constrained training may have a shrinking active batch
            # when instances finish at different decision points.  Its memory
            # explicitly requests active rows; legacy Memory keeps the old
            # full-batch behavior for backward compatibility.
            variable_batch = bool(getattr(memories, "variable_batch", False))
            batch_rows = state.batch_idxes if variable_batch else slice(None)
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch[batch_rows]))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch[batch_rows]))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch[batch_rows]))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))
            if self.stability_context_dim:
                memories.stability_action_context.append(
                    copy.deepcopy(stability_action_context)
                )
                memories.stability_global_context.append(
                    copy.deepcopy(stability_global_context)
                )

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, nums_opes, action_envs,
                 stability_action_context=None, stability_global_context=None,
                 flag_sample=False, return_features=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        ope_indexes = torch.arange(h_opes.size(1), device=h_opes.device)[None, :]
        visible_opes = ope_indexes < nums_opes[:, None]
        h_opes_pooled = (
            h_opes * visible_opes[:, :, None]
        ).sum(dim=-2) / nums_opes.clamp_min(1)[:, None]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        if self.stability_context_dim:
            if stability_action_context is None or stability_global_context is None:
                raise ValueError("Missing stability context for policy evaluation.")
            h_actions = torch.cat((h_actions, stability_action_context), dim=-1)
            h_pooled = torch.cat((h_pooled, stability_global_context), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        if return_features:
            return action_logprobs, state_values.squeeze().double(), dist_entropys, h_pooled
        return action_logprobs, state_values.squeeze().double(), dist_entropys


def load_compatible_policy_checkpoint(policy, checkpoint):
    """Load a legacy policy into a context-extended policy by zero padding."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a policy state dictionary.")

    target_state = policy.state_dict()
    unexpected = sorted(set(checkpoint).difference(target_state))
    missing = sorted(set(target_state).difference(checkpoint))
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint keys do not match policy: missing={0}, unexpected={1}.".format(
                missing, unexpected
            )
        )

    expandable = {
        "actor.linears.0.weight",
        "critic.linears.0.weight",
        "actor.linear.weight",
        "critic.linear.weight",
    }
    migrated_state = {}
    expanded_keys = []
    for key, target in target_state.items():
        source = checkpoint[key]
        if source.shape == target.shape:
            migrated_state[key] = source
            continue
        can_expand = (
            key in expandable
            and source.dim() == 2
            and target.dim() == 2
            and source.size(0) == target.size(0)
            and target.size(1) - source.size(1)
            == getattr(policy, "stability_context_dim", 0)
        )
        if not can_expand:
            raise RuntimeError(
                "Cannot migrate checkpoint tensor {0}: {1} -> {2}.".format(
                    key, tuple(source.shape), tuple(target.shape)
                )
            )
        padded = torch.zeros_like(target)
        padded[:, :source.size(1)] = source.to(
            device=target.device, dtype=target.dtype
        )
        migrated_state[key] = padded
        expanded_keys.append(key)

    policy.load_state_dict(migrated_state, strict=True)
    return {
        "migrated": bool(expanded_keys),
        "expanded_keys": tuple(expanded_keys),
        "stability_context_dim": getattr(policy, "stability_context_dim", 0),
    }

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0,1).flatten(0,1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        old_nums_opes = torch.stack(memory.nums_opes, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0,1).flatten(0, 1)
        old_stability_action_context = None
        old_stability_global_context = None
        if self.policy.stability_context_dim:
            old_stability_action_context = torch.stack(
                memory.stability_action_context, dim=0
            ).transpose(0, 1).flatten(0, 1)
            old_stability_global_context = torch.stack(
                memory.stability_global_context, dim=0
            ).transpose(0, 1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_nums_opes[start_idx: end_idx],
                                         old_action_envs[start_idx: end_idx],
                                         None if old_stability_action_context is None else
                                         old_stability_action_context[start_idx: end_idx],
                                         None if old_stability_global_context is None else
                                         old_stability_global_context[start_idx: end_idx])

                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                advantages = rewards_envs[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1, surr2)\
                       + self.vf_coeff * self.MseLoss(state_values, rewards_envs[i*minibatch_size:(i+1)*minibatch_size])\
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
