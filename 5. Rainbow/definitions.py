import os
from collections import deque

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import save_checkpoint


### Rainbow components ###
# ---------------------- #
# 1. Double Q-leraning   # ok.
# 2. Prioritized Replay  # ok.
# 3. Dueling Networks    # ok.
# 4. Multi-step Learning # ok.
# 5. Distributional RL   # ok.
# 6. Noisy Nets          # ok.


class NoisyLinear(nn.Module):
    """Linear layer with added noise to ensure exploration during training."""

    def __init__(self, in_features, out_features, sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("epsilon_input", torch.zeros(in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features))
        self._init_layers()

    def forward(self, x):
        x = x.to(self.mu_weight.device)
        if self.training:
            self._sample_noise()
            weight = self.mu_weight + self.sigma_weight * torch.outer(self.epsilon_output, self.epsilon_input)
            bias = self.mu_bias + self.sigma_bias * self.epsilon_output
            return F.linear(x, weight, bias)
        return F.linear(x, self.mu_weight, self.mu_bias)
    
    @torch.no_grad()
    def _sample_noise(self):
        input = torch.randn(self.in_features)
        output = torch.randn(self.out_features)
        self.epsilon_input.copy_(input.sign() * input.abs().sqrt())
        self.epsilon_output.copy_(output.sign() * output.abs().sqrt())
    
    def _init_layers(self):
        bound = self.in_features ** -0.5
        for name, param in self.named_parameters():
            if "mu" in name:
                nn.init.uniform_(param.data, -bound, bound)
            if "sigma" in name:
                nn.init.constant_(param.data, bound * self.sigma)


class ReplayBuffer:
    """Replay buffer to store agent's experience supporting multi-step learning."""

    def __init__(self, memory_size, td_steps, gamma, device="cpu"):
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.new_states = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        self.n_step_buffer = deque(maxlen=td_steps)
        self.memory_size = memory_size
        self.td_steps = td_steps
        self.gamma = gamma
        self.device = device

    def add(self, state, action, reward, new_state, done):
        self.n_step_buffer.append((state, action, reward, new_state, done))
        if len(self.n_step_buffer) < self.td_steps:
            return
        
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        _, _, last_reward, last_state, last_done = self.n_step_buffer[-1]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            _, _, r, n_s, d = transition
            last_reward = r + self.gamma * last_reward * (1 - d)
            last_state, last_done = (n_s, d) if d else (last_state, last_done)
        
        self.states.append(first_state)
        self.actions.append(first_action)
        self.rewards.append(last_reward)
        self.new_states.append(last_state)
        self.dones.append(last_done)
        self.priorities.append(max(self.priorities, default=1.0))

    def set_priorities(self, indices, errors, offset):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def get_batch(self, size, alpha, beta):
        probs = self.get_probs(alpha)
        indices = np.random.choice(len(self), size, replace=False, p=probs)
        batch_data = [self[i] for i in indices]
        states, actions, rewards, new_states, dones = zip(*batch_data)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        new_states = torch.tensor(np.array(new_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.long, device=self.device)
        importances = self.get_importances(probs[indices], beta)
        return states, actions, rewards, new_states, dones, importances, indices
    
    def get_probs(self, alpha):
        scaled_priorities = np.array(self.priorities) ** alpha
        probs = scaled_priorities / scaled_priorities.sum()
        return probs
    
    def get_importances(self, probs, beta):
        importances = (1 / len(self) * 1 / probs) ** beta
        importances /= importances.max()
        return torch.tensor(importances, device=self.device)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        new_state = self.new_states[index]
        done = self.dones[index]
        return state, action, reward, new_state, done
    
    def __len__(self):
        return len(self.states)
    

class LinearSchedule:
    """Linear schedule with option to set minimum threshold after N steps."""

    def __init__(self, start, end, steps):
        self.values = np.linspace(start, end, steps)
        self.steps = steps
        self.end = end
        self.index = 0

    def step(self):
        eps = self.values[self.index] if self.index < self.steps else self.end
        self.index += 1
        return eps
    

class NoisyDQN(nn.Module):
    """DQN with distributional output and noisy layers."""

    def __init__(self, v_min, v_max, n_atoms, sigma, device="cpu"):
        super().__init__()
        self.n_atoms = n_atoms
        self.support = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.core = nn.Sequential(
            NoisyLinear(8, 128, sigma),
            nn.ReLU(),
            NoisyLinear(128, 256, sigma),
            nn.ReLU(),
        )
        self.V_head = NoisyLinear(256, 1*n_atoms, sigma)
        self.A_head = NoisyLinear(256, 4*n_atoms, sigma)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) != 2:
            x = x[None, :]
        dist = self.dist(x)
        q = (dist * self.support).sum(2)
        return q
    
    def dist(self, x):
        hidden = self.core(x)
        values = self.V_head(hidden).view(-1, 1, self.n_atoms)
        advantages = self.A_head(hidden).view(-1, 4, self.n_atoms)
        q_atoms = values + (advantages - advantages.mean(1, keepdim=True))
        dist = F.softmax(q_atoms, -1)
        return dist
    

class Rainbow:
    """Rainbow trainer."""

    def __init__(
            self, 
            config, 
            env, 
            replay_buffer, 
            policy_dqn, 
            target_dqn,
            lr_schedule,
            beta_schedule,
            tb_writer
        ):
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer
        self.policy_dqn = policy_dqn
        self.target_dqn = target_dqn
        self.lr_schedule = lr_schedule
        self.tb_writer = tb_writer
        self.beta_schedule = beta_schedule
        self.policy_dqn.to(self.config["device"])
        self.target_dqn.to(self.config["device"])
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=config["lr"], eps=1.5e-4)
        self.update_networks(tau=1.0)

    def update_networks(self, tau):
        """Update target net with policy net using Polyak Averaging."""
        for policy_dqn_param, target_dqn_param in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            target_dqn_param.data.copy_(tau * policy_dqn_param.data + (1.0 - tau) * target_dqn_param.data)

    def learn(self):
        """Training function."""

        episode = 0
        learning_step = 0
        experience_step = 0
        state, _ = self.env.reset()

        support = torch.linspace(self.config["v_min"], self.config["v_max"], self.config["n_atoms"]).to(self.config["device"])
        delta_z = (self.config["v_max"] - self.config["v_min"]) / (self.config["n_atoms"] - 1)

        reward_buffer = deque(maxlen=self.config["log_buff_size"])
        length_buffer = deque(maxlen=self.config["log_buff_size"])

        experience_pbar = tqdm(total=self.config["min_samples"], ncols=100, desc="Filling")
        learning_pbar = tqdm(total=self.config["n_steps"], ncols=100, desc="Learning")

        # Main training loop.
        while learning_step < self.config["n_steps"]:
            with torch.no_grad():
                action = self.policy_dqn(state).argmax(1).cpu().numpy()
            new_state, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.add(state[0], action[0], reward[0], new_state[0], int(terminated | truncated))
            state = new_state

            if experience_step < self.config["min_samples"]:
                experience_pbar.update(1)
                experience_step += 1

            if len(self.replay_buffer) >= self.config["min_samples"]:
                beta = self.beta_schedule.step()
                batch = self.replay_buffer.get_batch(self.config["batch_size"], self.config["alpha"], beta)
                (
                    batch_states, 
                    batch_actions, 
                    batch_rewards, 
                    batch_new_states, 
                    batch_dones, 
                    batch_importances, 
                    batch_indices
                ) = batch

                # Forward pass.
                with torch.no_grad():
                    next_action = self.policy_dqn(batch_new_states).argmax(1)
                    next_dist = self.target_dqn.dist(batch_new_states)
                    next_dist = next_dist[range(self.config["batch_size"]), next_action]
                    
                    discount = self.config["gamma"] ** self.config["n_steps"]
                    t_z = batch_rewards[:, None] + (1 - batch_dones[:, None]) * discount * support
                    t_z = t_z.clamp(self.config["v_min"], self.config["v_max"])
                    b = (t_z - self.config["v_min"]) / delta_z
                    l = b.floor().long()
                    u = b.ceil().long()

                    offset = (
                        torch.linspace(
                            0, (self.config["batch_size"] - 1) * self.config["n_atoms"], self.config["batch_size"]
                        )
                        .long()
                        .unsqueeze(1)
                        .expand(self.config["batch_size"], self.config["n_atoms"])
                        .to(self.config["device"])
                    )

                    proj_dist = torch.zeros(next_dist.size(), device=self.config["device"])
                    proj_dist.view(-1).index_add_(
                        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                    )
                    proj_dist.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                    )

                dist = self.policy_dqn.dist(batch_states)
                log_p = torch.log(dist[range(self.config["batch_size"]), batch_actions])
                loss = -(proj_dist * log_p).sum(1)
                deltas = loss.detach().cpu().numpy()
                scaled_loss = (loss * batch_importances).mean()

                # Backward pass.
                self.optimizer.zero_grad()
                scaled_loss.backward()
                if self.config["max_grad_norm"] is not None:
                    nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()

                # Update target network.
                if (learning_step + 1) % self.config["sync_freq"] == 0:
                    self.update_networks(tau=self.config["tau"])

                # Set new priorities. 
                self.replay_buffer.set_priorities(batch_indices, deltas, self.config["offset"])
            
                # Logging to Tensorboard.
                if "final_info" in info.keys():
                    for element in info["final_info"]:
                        if element and "episode" in element:

                            reward_buffer.append(element["episode"]["r"])
                            length_buffer.append(element["episode"]["l"])
                            episode += 1

                            if len(reward_buffer) == self.config["log_buff_size"]:
                                self.tb_writer.add_scalar("charts/episode_reward", np.mean(reward_buffer), episode)
                                self.tb_writer.add_scalar("charts/episode_length", np.mean(length_buffer), episode)

                if (learning_step + 1) % self.config["logging_freq"] == 0:
                    self.tb_writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], learning_step)
                    self.tb_writer.add_scalar("charts/beta", beta, learning_step)
                    self.tb_writer.add_scalar("losses/scaled_loss", scaled_loss.cpu().detach().numpy().mean(), learning_step)

                    grad_norms = []
                    for _, param in self.policy_dqn.named_parameters():
                        if param.grad is not None:
                            grad_norms.append(torch.norm(param.grad, p=2).item())

                    if len(grad_norms) > 0:
                        self.tb_writer.add_scalar("charts/grad_norm", sum(grad_norms), learning_step)

                # Save a checkpoint during training if the save frequency is specified.
                if self.config["save_freq"] is not None and (learning_step + 1) % self.config["save_freq"] == 0:
                    name, ext = os.path.splitext(self.config["save_to"])
                    checkpoint_name = f"{name}_{learning_step+1}{ext}"
                    save_checkpoint(checkpoint_name, self.model)

                self.optimizer.param_groups[0]["lr"] = self.lr_schedule.step()
                learning_pbar.update(1)
                learning_step += 1

        # Final checkpoint after training.
        save_checkpoint(self.config["save_to"], self.model)