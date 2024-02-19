import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class PG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        x = self.net(x)
        return x
    

    def sample_action(self, x):
        probs = self.net(x)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = probs[torch.arange(len(actions)), actions].log()
        return actions, log_probs



class Agent:
    def __init__(
            self, 
            config, 
            env, 
            pg_net, 
            tb_writer
    ):
        self.config = config
        self.env = env
        self.pg_net = pg_net
        self.pg_net.to(config["device"])
        self.tb_writer = tb_writer
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=config["learning_rate"])
        self.rewards = []
        self.log_probs = []
        self.episode_num = 0 
        self.experience_steps = 0


    def collect_experience(self):
        done = truncated = False
        observation, _ = self.env.reset()

        while not (done or truncated):
            action, log_prob = self.pg_net.sample_action(torch.tensor(observation).unsqueeze(0))
            new_observation, reward, done, truncated, _ = self.env.step(action.item())
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            observation = new_observation
            self.experience_steps += 1
        self.episode_num += 1
        
    
    def learn_from_experience(self):
        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt += r * self.config["gamma"] ** pw
                pw += 1
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std() + 1e-9

        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        policy_gradient = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.log_probs = []
        self.experience_steps = 0


    def log(self):
        self.tb_writer.add_scalar("Rewards/pg_rwd", sum(self.rewards), self.episode_num)
        self.tb_writer.add_scalar("Steps per Update/pg_spu", self.experience_steps, self.episode_num)