import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from utils import save_checkpoint, layer_init


class ActorCritic(nn.Module):
    """Actor-Critic architecture, inspired by DeepMind's DQN paper."""

    def __init__(self):
        super().__init__()

        # Core network architecture.
        self.core = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4), gain=2**0.5),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2), gain=2**0.5),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1), gain=2**0.5),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*7*7, 512), gain=2**0.5),
            nn.ReLU()
        )

        # Policy head for action selection.
        self.policy_head = layer_init(nn.Linear(512, 4), gain=0.01)

        # Value head for state value estimation.
        self.value_head = layer_init(nn.Linear(512, 1), gain=1)

    def value(self, x):
        """Calculate the state value."""
        x = x / 255
        x = self.core(x)
        x = self.value_head(x)
        return x
    
    def policy(self, x, action=None):
        """Calculate the policy distribution and sample an action."""
        x = x / 255
        hidden = self.core(x)
        logits = self.policy_head(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value_head(hidden)


class Storage(Dataset):
    """Dataset class for storing trajectory data."""

    def __init__(self, trajectories, advantages, returns):
        self.obs = trajectories["obs"].reshape(-1, 4, 84, 84)
        self.log_probs = trajectories["log_probs"].reshape(-1)
        self.actions = trajectories["actions"].reshape(-1).long()
        self.advantages = advantages.reshape(-1)
        self.returns = returns.reshape(-1)

    def __getitem__(self, index):
        item = { 
            "obs": self.obs[index],
            "log_probs": self.log_probs[index],
            "actions": self.actions[index],
            "advantages": self.advantages[index],
            "returns": self.returns[index]
        }
        return item
    
    def __len__(self):
        return len(self.obs)


class PPO:
    """Proximal Policy Optimization trainer."""

    def __init__(self, config, envs, model, tb_writer):
        self.config = config
        self.envs = envs
        self.model = model.to(config["device"])
        self.tb_writer = tb_writer
        self.optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.episode_step = 0
        self.global_step = 0


    def gae(self, cur_obs, rewards, dones, values):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        with torch.no_grad():
            last_value = self.model.value(cur_obs).reshape(1, -1)

        for t in reversed(range(self.config["max_episode_steps"])):
            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[:, t] + self.config["gamma"] * last_value - values[:, t]
            last_advantage = delta + self.config["gamma"] * self.config["lambda"] * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        advantages = advantages.to(self.config["device"])
        returns = advantages + values
        return advantages, returns
    
    def rollout(self, cur_obs, cur_done):
        """Create trajectories to learn from."""

        # Initialize buffers of specific size.
        obs = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"], 4, 84, 84).to(self.config["device"])
        actions = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"]).to(self.config["device"])
        log_probs = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"]).to(self.config["device"])
        rewards = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"]).to(self.config["device"])
        dones = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"]).to(self.config["device"])
        values = torch.zeros(self.config["num_envs"], self.config["max_episode_steps"]).to(self.config["device"])

        # Collect the data.
        for t in range(self.config["max_episode_steps"]):
            with torch.no_grad():
                action, log_prob, _, value = self.model.policy(cur_obs)

            obs[:, t] = cur_obs
            dones[:, t] = cur_done
            values[:, t] = value.flatten()
            actions[:, t] = action
            log_probs[:, t] = log_prob

            cur_obs, reward, cur_done, _, info = self.envs.step(action.cpu().numpy())
            rewards[:, t] = torch.tensor(reward).to(self.config["device"]).view(-1)
            cur_obs = torch.tensor(cur_obs).to(self.config["device"])
            cur_done = torch.tensor(cur_done).to(self.config["device"])

            # Log to Tensorboard if episode has ended.
            if "final_info" in info.keys():
                for element in info["final_info"]:
                    if element and "episode" in element:
                        episode_info = element["episode"]
                        self.tb_writer.add_scalar("charts/episode_length", episode_info["l"], self.episode_step)
                        self.tb_writer.add_scalar("charts/episode_reward", episode_info["r"], self.episode_step)
                        self.episode_step += 1
                        break

        trajectories = {
            "cur_obs": cur_obs,
            "cur_done": cur_done,
            "obs": obs,
            "dones": dones,
            "values": values,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards
        }
        return trajectories
    
    @staticmethod
    def loss_clip(old_log_prob, new_log_prob, advantages, epsilon):
        """Part 1/3 of full PPO loss."""
        ratio = torch.exp(new_log_prob - old_log_prob)
        policy_loss = -advantages * ratio
        clipped_policy_loss = -advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = torch.max(policy_loss, clipped_policy_loss).mean()
        return loss


    @staticmethod
    def loss_vf(returns, values):
        """Part 2/3 of full PPO loss."""
        loss = 0.5 * ((values.view(-1) - returns) ** 2).mean()
        return loss

    @staticmethod
    def ent_loss(entropy):
        """Part 3/3 of full PPO loss."""
        loss = -entropy.mean()
        return loss

    def learn(self):
        """Training function."""

        # Initialize first observation and done.
        cur_obs = torch.tensor(self.envs.reset()[0]).to(self.config["device"])
        cur_done = torch.tensor(self.config["num_envs"]).to(self.config["device"])

        # Calculate total training steps.
        num_steps = self.config["total_train_steps"] // self.config["batch_size"]

        # Main training loop.
        for step in tqdm(range(num_steps), total=num_steps, ncols=100):

            # Collect trajectories and unpack them.
            trajectories = self.rollout(cur_obs, cur_done)
            cur_done = trajectories["cur_done"]
            cur_obs = trajectories["cur_obs"]
            rewards = trajectories["rewards"]
            dones = trajectories["dones"]
            values = trajectories["values"]

            # Calculate Advantages.
            advantages, returns = self.gae(cur_obs, rewards, dones, values)

            # Put calculated values in the dataset.
            storage = Storage(trajectories, advantages, returns)
            train_loader = DataLoader(storage, batch_size=self.config["minibatch_size"], shuffle=True)

            # Linear learning rate annealing.
            frac = 1.0 - (step - 1.0) / num_steps
            self.optimizer.param_groups[0]["lr"] = frac * self.config["learning_rate"]
            
            # Iterate over created dataset for N epochs.
            for _ in range(self.config["total_epochs"]):
                for batch in train_loader:
                    batch_obs = batch["obs"]
                    batch_log_probs = batch["log_probs"]
                    batch_actions = batch["actions"]
                    batch_advantages = batch["advantages"]
                    batch_returns = batch["returns"]

                    # Query the model.
                    _, new_log_probs, entropy, new_values = self.model.policy(batch_obs, batch_actions)

                    # Calculate PPO loss.
                    clip_loss = self.loss_clip(batch_log_probs, new_log_probs, batch_advantages, self.config["epsilon"]) 
                    value_loss = self.loss_vf(batch_returns, new_values)
                    entropy_loss = self.ent_loss(entropy)
                    loss = clip_loss + self.config["entropy_coeff"] * entropy_loss + self.config["value_coeff"] * value_loss

                    # Backward pass and clipping gradients for numerical stability.
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config["max_grad_norm"] is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    self.optimizer.step()
            
            # Logging to Tensorboard.
            if (step + 1) % self.config["logging_freq"] == 0:
                self.tb_writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], step)
                self.tb_writer.add_scalar("losses/clip_loss", clip_loss.item(), step)
                self.tb_writer.add_scalar("losses/value_loss", value_loss.item(), step)
                self.tb_writer.add_scalar("losses/entropy", entropy_loss.item(), step)

                grad_norms = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(torch.norm(param.grad, p=2).item())

                if len(grad_norms) > 0:
                    self.tb_writer.add_scalar("charts/grad_norm", sum(grad_norms), step)

            # Save a checkpoint during training if the save frequency is specified.
            if self.config["save_freq"] != -1 and (step + 1) % self.config["save_freq"] == 0:
                name, ext = os.path.splitext(self.config["save_to"])
                checkpoint_name = f"{name}_{step+1}{ext}"
                save_checkpoint(checkpoint_name, self.model, self.optimizer, self.episode_step, self.global_step)

            self.global_step += 1

        # Final checkpoint of trained model.
        save_checkpoint(self.config["save_to"], self.model, self.optimizer, self.episode_step, self.global_step)
