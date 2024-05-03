from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_sizes = config["layer_sizes"].copy()
        layer_sizes.insert(0, config["observation_space"])
        layer_sizes.append(config["action_space"])
        self.net = nn.Sequential()

        for i in range(1, len(layer_sizes)):
            self.net.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            if i != len(layer_sizes) - 1:
                self.net.append(nn.ReLU())

    def forward(self, x):
        x = self.net(x)
        return x
    

class LinearScheduler:
    def __init__(self, config):
        self.start_value = config["epsilon_start_value"]
        self.end_value = config["epsilon_end_value"]
        self.num_steps = config["epsilon_num_steps"]
        self.epsilons = np.linspace(self.start_value, self.end_value, self.num_steps)
        self.current_step = 0

    def step(self):
        eps = self.epsilons[self.current_step] if self.current_step < self.num_steps else self.end_value
        self.current_step += 1
        return eps
    

class MemoryBuffer:
    def __init__(self, config):
        self.states = deque(maxlen=config["memory_size"])
        self.actions = deque(maxlen=config["memory_size"])
        self.rewards = deque(maxlen=config["memory_size"])
        self.new_states = deque(maxlen=config["memory_size"])
        self.dones = deque(maxlen=config["memory_size"])

    def write(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        new_state = self.new_states[index]
        done = self.dones[index]
        return state, action, reward, new_state, done
    
    def __len__(self):
        return len(self.states)
    

class Agent:
    def __init__(
            self, 
            config, 
            env, 
            memory_buffer, 
            schedule,
            policy_dqn, 
            target_dqn,
            tb_writer
        ):
        self.config = config
        self.env = env
        self.memory_buffer = memory_buffer
        self.schedule = schedule
        self.policy_dqn = policy_dqn
        self.target_dqn = target_dqn
        self.tb_writer = tb_writer
        self.policy_dqn.to(self.config["device"])
        self.target_dqn.to(self.config["device"])
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), config["learning_rate"])

        self.state, _ = self.env.reset()
        self.steps_from_last_sync = 0
        self.train_step = 0
        self.maybe_update_target_dqn(force_update=True)

        self.tb_rewards = [] 
        self.tb_losses = [] 
        self.tb_epsilons = [] 
        self.tb_scores = [] 

    
    def collect_experience(self):
        action = self.sample_action(self.state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        self.memory_buffer.write(self.state, action, reward, new_state, terminated)
        self.state = new_state
        self.tb_scores.append(reward) 

        if terminated or truncated:
            self.state, _ = self.env.reset()
            self.tb_writer.add_scalar("Avg_scores/avg_score", sum(self.tb_scores), self.train_step)
            self.tb_scores = []
    
    
    def sample_action(self, state):
        eps = self.schedule.step()
        self.tb_epsilons.append(eps) 
        state = torch.tensor(state).to(self.config["device"])
        actions = self.policy_dqn(state).cpu().detach().numpy()
        action = np.argmax(actions) if np.random.rand() > eps else np.random.choice(np.arange(self.config["action_space"]))
        return action


    def learn_from_batch(self):
        indexes = np.random.choice(len(self.memory_buffer), self.config["batch_size"], replace=False)
        batch_data = [self.memory_buffer[i] for i in indexes]

        states, actions, rewards, new_states, dones = zip(*batch_data)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.config["device"])
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.config["device"])
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.config["device"])
        new_states = torch.tensor(np.array(new_states), dtype=torch.float32).to(self.config["device"])
        dones = torch.tensor(np.array(dones), dtype=torch.int32).to(self.config["device"])

        q = self.policy_dqn(states)[np.arange(len(actions)), actions]
        q_next = self.target_dqn(new_states)
        q_hat = rewards + (1 - dones) * self.config["gamma"] * q_next.max(dim=1)[0]
        
        loss = F.mse_loss(q, q_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1

        self.tb_losses.append(loss.item()) 
        self.tb_rewards.append(rewards.mean().item()) 
        

    def maybe_update_target_dqn(self, force_update=False):
        if self.steps_from_last_sync == self.config["target_dqn_update"] or force_update:
            for src, tgt in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
                tgt.data.copy_(self.config["tau"] * src.data + (1.0 - self.config["tau"]) * tgt.data)
            self.steps_from_last_sync = 0
        else: 
            self.steps_from_last_sync += 1


    def maybe_log(self):
        if self.train_step != 0 and self.train_step % self.config["logging_freq"] == 0:
            self.tb_writer.add_scalar("Loss/loss", sum(self.tb_losses) / len(self.tb_losses), self.train_step)
            self.tb_writer.add_scalar("Rewards/rewards", sum(self.tb_rewards) / len(self.tb_rewards), self.train_step)
            self.tb_writer.add_scalar("Epsilon/epsilon", sum(self.tb_epsilons) / len(self.tb_epsilons), self.train_step)
            self.tb_losses = []
            self.tb_rewards = []
            self.tb_epsilons = []


    def log_hyperparams(self):
        hyperparams = {
            "learning_rate": self.config["learning_rate"],
            "batch_size": self.config["batch_size"],
            "layer_sizes": "-".join(str(i) for i in self.config["layer_sizes"]),
            "epsilon_start_value": self.config["epsilon_start_value"],
            "epsilon_end_value": self.config["epsilon_end_value"],
            "epsilon_num_steps": self.config["epsilon_num_steps"],
            "total_train_steps": self.config["total_train_steps"],
            "gamma": self.config["gamma"],
            "tau": self.config["tau"],
            "target_dqn_update": self.config["target_dqn_update"],
        }

        tab = "<br>".join(f"{key} {val}" for key, val in hyperparams.items())
        self.tb_writer.add_text("params", tab)
