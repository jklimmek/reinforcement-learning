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
        self.step = 0

    def __call__(self):
        eps = self.epsilons[self.step] if self.step < self.num_steps else self.end_value
        self.step += 1
        return eps
    

class MemoryBuffer:
    def __init__(self, config):
        # todo: Add way of calculating memory_size based on max_mb_size
        self.states = np.zeros((config["memory_size"], config["observation_space"]), dtype=np.float32)
        self.actions = np.zeros((config["memory_size"], 1), dtype=np.uint8)
        self.rewards = np.zeros((config["memory_size"], 1), dtype=np.float32)
        self.new_states = np.zeros((config["memory_size"], config["observation_space"]), dtype=np.float32)
        self.dones = np.zeros((config["memory_size"], 1), dtype=np.uint8)
        self.counter = 0
    
    def write(self, state, action, reward, new_state):
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.new_states[self.counter] = new_state
        self.counter += 1

    def clear_memory(self):
        self.states = np.zeros_like(self.states, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.uint8)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.new_states = np.zeros_like(self.new_states, dtype=np.float32)
        self.dones = np.zeros_like(self.dones, dtype=np.uint8)
        self.counter = 0

    def is_full(self):
        return self.counter == len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        new_state = self.new_states[index]
        done = self.dones[index]
        return state, action, reward, new_state, done
    

class Agent:
    def __init__(
            self, 
            config, 
            env, 
            memory_buffer, 
            schedule,
            policy_dqn, 
            target_dqn
            
        ):
        self.config = config
        self.env = env
        self.memory_buffer = memory_buffer
        self.schedule = schedule
        self.policy_dqn = policy_dqn
        self.target_dqn = target_dqn
        self.policy_dqn.to(self.config["device"])
        self.target_dqn.to(self.config["device"])
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self._update_target_dqn()

        self.optimizer = optim.Adam(self.policy_dqn.parameters(), config["learning_rate"])
        self.available_indexes = np.arange(config["memory_size"])
        self.steps_from_last_sync = 0
        self.total_train_steps = 0

    
    def collect_experience(self):
        state, _ = self.env.reset()
        # for i in tqdm(range(self.config["memory_size"]), total=self.config["memory_size"]):
        while not self.memory_buffer.is_full():
            action = self.sample_action(state)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            self.memory_buffer.write(state, action, reward, new_state)
            state = new_state

            if terminated or truncated:
                state, _ = self.env.reset()
    
    
    def sample_action(self, state):
        eps = self.schedule.step()
        val = np.random.rand()
        actions = self.policy_dqn(torch.tensor(state)).detach().numpy()
        action = np.argmax(actions) if val < eps else np.random.choice(np.arange(self.config["action_space"]))
        return action


    def learn(self):
        if len(self.available_indexes) < self.config["batch_size"]:
             self.available_indexes = set(np.arange(self.config["memory_size"]))
             self.memory_buffer.collect_experience()
             print("Refreshing buffer...")
        
        if self.steps_from_last_sync == self.config["target_dqn_update"]:
            self._update_target_dqn()
            print("Updating target DQN...")

        indexes = np.random.choice(self.available_indexes, self.config["batch_size"], replace=False)
        self.available_indexes = np.setdiff1d(self.available_indexes, indexes, assume_unique=True)

        states, actions, rewards, new_states, dones = self.memory_buffer[indexes]
        states = torch.tensor(states).to(self.config["device"])
        actions = torch.tensor(actions).to(self.config["device"])
        rewards = torch.tensor(rewards).to(self.config["device"])
        new_states = torch.tensor(new_states).to(self.config["device"])
        dones = torch.tensor(dones).to(self.config["device"])

        q = self.policy_dqn(states)
        q_next = self.target_dqn(new_states)
        q_hat = rewards + (1 - dones) * self.config["gamma"] * q_next.max(dim=0)[0].unsqueeze(0)
        
        loss = F.mse_loss(q, q_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_from_last_sync += 1
        self.total_train_steps += 1
        

    def _update_target_dqn(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        for pol, tgt in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            assert (pol == tgt).all(), "`policy_dqn` and `target_dqn` must have the same weights."


if __name__ == "__main__":
    # todo: Implement testing ✔️❌
    pass