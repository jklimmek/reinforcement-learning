import argparse
import os
import gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything, save_dqn, get_run_name
from definitions import Agent, DQN, MemoryBuffer, LinearScheduler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Deep Q Network training script.")

    parser.add_argument("--env_name", type=str, help="Name of the environment.")
    parser.add_argument("--save_to", type=str, default="./checkpoints/model.pt", help="Path to save model.")
    parser.add_argument("--save_freq", type=int, default=-1, help="Save model every N steps, -1 means model is saved only once at the end.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--observation_space", type=int, help="Observation space dimension.")
    parser.add_argument("--action_space", type=int, help="Action space dimension.")
    parser.add_argument("--layer_sizes", nargs="+", type=int, help="Sizes of layers in the neural network.")
    parser.add_argument("--memory_size", type=int, default=100_000, help="Size of the replay memory.")
    
    parser.add_argument("--epsilon_start_value", type=float, default=1.0, help="Initial epsilon value for epsilon-greedy exploration.")
    parser.add_argument("--epsilon_end_value", type=float, default=0.01, help="Final epsilon value for epsilon-greedy exploration.")
    parser.add_argument("--epsilon_num_steps", type=int, default=5_000, help="Number of steps to anneal epsilon.")
    
    parser.add_argument("--total_train_steps", type=int, default=200_000, help="Total number of training steps.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--tau", type=float, default=0.001, help="Soft update parameter for updating target DQN")
    parser.add_argument("--target_dqn_update", type=int, default=5, help="Number of steps to update the target DQN.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda) for training.")

    parser.add_argument("--logging_freq", type=int, default=10, help="Frequency of logging to Tensorboard.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to output logs.")
    parser.add_argument("--comment", type=str, default="", help="Comment for Tensorboard.")
    args = parser.parse_args()

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    return config


def train_dqn(config):
    env = gym.make(config["env_name"])
    policy_dqn = DQN(config)
    target_dqn = DQN(config)
    memory_buffer = MemoryBuffer(config)
    schedule = LinearScheduler(config)
    tb_writer =  SummaryWriter(os.path.join(config["log_dir"], get_run_name()), config["comment"])

    agent = Agent(config, env, memory_buffer, schedule, policy_dqn, target_dqn, tb_writer)

    seed_everything(env, config["seed"])
    agent.log_hyperparams()

    for _ in tqdm(range(config["total_train_steps"]), total=config["total_train_steps"], ncols=100):
        if len(agent.memory_buffer) >= config["batch_size"]:
            agent.learn_from_batch()

        agent.collect_experience()
        agent.maybe_update_target_dqn()
        agent.maybe_log()

        if config["save_freq"] != -1 and agent.train_step % config["save_freq"] == 0:
            save_dqn(agent.policy_dqn, f"{config["save_to"]}_{agent.train_step}.pt")

    agent.tb_writer.close()
    return agent


def main():
    config = parse_arguments()
    agent = train_dqn(config)
    save_dqn(agent.policy_dqn, config["save_to"])


if __name__ == "__main__":
    main()