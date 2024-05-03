import argparse
import warnings
import os
import gym
from torch.utils.tensorboard import SummaryWriter

from utils import make_env, get_run_name, seed_almost_everything
from definitions import ActorCritic, PPO


def parse_arguments():
    """Parse command line training arguments."""
    parser = argparse.ArgumentParser(description="PPO training script on Atari Breakout.")

    # Training configuration.
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--max_episode_steps", type=int, default=128, help="Number of steps to run in each environment.")
    parser.add_argument("--total_train_steps", type=int, default=5_000_000, help="Total timesteps for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Clip gradients at specific value.")
    parser.add_argument("--num_minibatches", type=int, default=8, help="Number of minibatches.")
    parser.add_argument("--total_epochs", type=int, default=2, help="Total number of epochs.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--lambda", type=float, default=0.95, help="Value for advantage estimation.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Clipping policy between [1 - epsilon, 1 + epsilon].")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Controls importance of entropy in loss.")
    parser.add_argument("--value_coeff", type=float, default=0.5, help="Controls importance of values in loss.")

    # Environment and model configuration.
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda) for training.")

    # Saving and logging.
    parser.add_argument("--save_to", type=str, default="./checkpoints/model.pt", help="Path to save model.")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save model every N episodes, -1 means model is saved only once at the end.")
    parser.add_argument("--logging_freq", type=int, default=1, help="Frequency of logging to Tensorboard.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to output logs.")
    parser.add_argument("--comment", type=str, default="", help="Comment for Tensorboard.")
    args = parser.parse_args()

    # Convert arguments to dict format.
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    # Additional config parameters.
    config["batch_size"] = int(config["num_envs"] * config["max_episode_steps"])
    config["minibatch_size"] = int(config["batch_size"] // config["num_minibatches"])
    return config


def train_ppo(config):
    """Start PPO training."""
    
    # Set seed if requested.
    seed_almost_everything(config["seed"])

    # Set up PPO trainer components.
    envs = gym.vector.SyncVectorEnv([make_env(config["seed"]) for _ in range(config["num_envs"])])

    model = ActorCritic()

    tb_writer =  SummaryWriter(os.path.join(config["log_dir"], get_run_name()), config["comment"])
    tb_writer.add_text(
        "hyperparams",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in config.items()))
    )
    
    # Set up PPO trainer.
    ppo = PPO(config, envs, model, tb_writer)

    # Start training.
    ppo.learn()


def main():
    warnings.filterwarnings('ignore', ".*env.*")
    config = parse_arguments()
    train_ppo(config)


if __name__ == "__main__":
    main()
