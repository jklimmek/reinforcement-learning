import argparse
import warnings
import os
import gym
from torch.utils.tensorboard import SummaryWriter

from utils import make_env, get_run_name, seed_almost_everything
from definitions import NoisyDQN, Rainbow, ReplayBuffer, LinearSchedule


def parse_arguments():
    """Parse command line training arguments."""
    parser = argparse.ArgumentParser(description="Rainbow training script on Atari Pacman.")

    # Training configuration.
    parser.add_argument("--n_steps", type=int, default=200_000, help="Total timesteps for training.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Clip gradients at specific value.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")

    # Double Q-leraning.
    parser.add_argument("--tau", type=float, default=0.005, help="Parameter for Polyak averaging.")
    parser.add_argument("--sync_freq", type=int, default=4, help="Synchronize nets every N steps.")

    # Prioritized Replay Buffer.
    parser.add_argument("--memory_size", type=int, default=100_000, help="Replay buffer size.")
    parser.add_argument("--min_samples", type=int, default=250, help="Minimum samples to start learning.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Scale priorities by value.")
    parser.add_argument("--beta_min", type=float, default=0.4, help="Minimum value of exponent in importance sampling.")
    parser.add_argument("--beta_max", type=float, default=1.0, help="Maximum value of exponent in importance sampling.")
    parser.add_argument("--beta_steps", type=int, default=180_000, help="Increase value of exponent in importance sampling over N steps.")
    parser.add_argument("--offset", type=float, default=0.1, help="Add small value when setting priority.")

    # Multi-step Learning, TD(n).
    parser.add_argument("--td_steps", type=int, default=2, help="Calculate Q-value N steps ahead.")

    # Distributional RL.
    parser.add_argument("--v_min", type=float, default=-100.0, help="Minimum boundary of distribution.")
    parser.add_argument("--v_max", type=float, default=100.0, help="Maximum boundary of distribution.")
    parser.add_argument("--n_atoms", type=int, default=51, help="Number of bins.")

    # Noisy Nets.
    parser.add_argument("--sigma", type=float, default=0.5, help="Scale sigma by value.")

    # Util, saving and logging.
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda') for training.")
    parser.add_argument("--save_to", type=str, default="./checkpoints/model.pt", help="Path to save model.")
    parser.add_argument("--save_freq", type=int, default=None, help="Save model every N steps.")
    parser.add_argument("--log_buff_size", type=int, default=10, help="Running mean of N past values to log rewards and episode length.")
    parser.add_argument("--logging_freq", type=int, default=1, help="Frequency of logging to Tensorboard.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to output logs.")
    parser.add_argument("--comment", type=str, default="", help="Comment for Tensorboard.")
    args = parser.parse_args()

    return vars(args)


def train_rainbow(config):
    """Start Rainbow training."""

    # Set seed if requested.
    seed_almost_everything()

    # Set up Rainbow trainer components.
    env = gym.vector.SyncVectorEnv([make_env()])

    replay_buffer = ReplayBuffer(config["memory_size"], config["td_steps"], config["gamma"], config["device"])

    policy_dqn = NoisyDQN(config["v_min"], config["v_max"], config["n_atoms"], config["sigma"], config["device"])
    target_dqn = NoisyDQN(config["v_min"], config["v_max"], config["n_atoms"], config["sigma"], config["device"])

    lr_schedule = LinearSchedule(config["lr"], 0, config["n_steps"])
    beta_schedule = LinearSchedule(config["beta_min"], config["beta_max"], config["beta_steps"])

    tb_writer =  SummaryWriter(os.path.join(config["log_dir"], get_run_name()), config["comment"])
    tb_writer.add_text(
        "hyperparams",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in config.items()))
    )

    # Set up Rainbow trainer.
    rainbow = Rainbow(config, env, replay_buffer, policy_dqn, target_dqn, lr_schedule, beta_schedule, tb_writer)

    # Start training.
    rainbow.learn()


def main():
    config = parse_arguments()
    train_rainbow(config)


if __name__ == "__main__":
    main()