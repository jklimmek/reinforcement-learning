import argparse
import os
import gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything, save_pg, get_run_name
from definitions import Agent, PG



def parse_arguments():
    parser = argparse.ArgumentParser(description="Policy Gradients training script on LunarLander-v2.")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--total_episodes", type=int, default=3_000, help="Total number of training episodes.")
    parser.add_argument("--steps_per_update", type=int, default=0, help="After how many steps update PG network, small value means learning after each episode.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda) for training.")

    parser.add_argument("--save_to", type=str, default="./checkpoints/model.pt", help="Path to save model.")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save model every N episodes, -1 means model is saved only once at the end.")
    parser.add_argument("--logging_freq", type=int, default=1, help="Frequency of logging to Tensorboard.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to output logs.")
    parser.add_argument("--comment", type=str, default="", help="Comment for Tensorboard.")
    args = parser.parse_args()

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    return config



def train_pg(config):
    env = gym.make("LunarLander-v2")
    pg_net = PG()
    tb_writer =  SummaryWriter(os.path.join(config["log_dir"], get_run_name()), config["comment"])

    agent = Agent(config, env, pg_net, tb_writer)

    seed_everything(config["seed"])

    for _ in tqdm(range(config["total_episodes"]), total=config["total_episodes"], ncols=100):

        agent.collect_experience()
        
        if agent.experience_steps >= config["steps_per_update"]:
            agent.log()
            agent.learn_from_experience()

        if config["save_freq"] != -1 and agent.episode_num % config["save_freq"] == 0:
            file, ext = os.path.splitext(config['save_to'])
            save_pg(agent.pg_net, f"{file}_{agent.episode_num}{ext}")

    agent.tb_writer.close()
    return agent



def main():
    config = parse_arguments()
    agent = train_pg(config)
    save_pg(agent.pg_net, config["save_to"])



if __name__ == "__main__":
    main()
