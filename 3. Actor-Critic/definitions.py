import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class AC(nn.Module):

    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        self.v = nn.Linear(1024, 1)
        self.pi = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.core(x)
        v = self.v(x)
        pi = F.softmax(self.pi(x), dim=1)
        return v, pi


class Agent:

    def __init__(
        self, 
        config, 
        env, 
        ac_net, 
        tb_writer
    ):
        self.config = config
        self.env = env
        self.ac_net = ac_net
        self.ac_net.to(self.config["device"])
        self.tb_writer = tb_writer
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=self.config["learning_rate"])
        self.tb_rewards = []
        self.episode_num = 0
    
    def learn_from_episode(self):
        done = truncated = False
        observation, _ = self.env.reset()
        episode_reward = 0

        while not (done or truncated):
            
            v, pi = self.ac_net(torch.tensor(observation).unsqueeze(0).to(self.config["device"]))
            probs = Categorical(pi[0])
            action = probs.sample().item()
            log_prob = probs.logits[action]
            new_observation, reward, done, truncated, _ = self.env.step(action)
            v_next, _ = self.ac_net(torch.tensor(new_observation).unsqueeze(0).to(self.config["device"]))

            delta = reward + (1 - done) * self.config["gamma"] * v_next - v
            actor_loss = -log_prob * delta.detach()
            critic_loss = delta ** 2
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            observation = new_observation.copy()
            episode_reward += reward

        self.tb_rewards.append(episode_reward)
        self.episode_num += 1

    def maybe_log(self):
        if self.episode_num % self.config["logging_freq"] == 0:
            self.tb_writer.add_scalar("Rewards/ac_rwd",  sum(self.tb_rewards), self.episode_num)
        self.tb_rewards = []
