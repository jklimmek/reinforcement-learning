import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="DQN Training Script")

    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--observation_space", type=int, help="Observation space dimension")
    parser.add_argument("--action_space", type=int, help="Action space dimension")
    parser.add_argument("--layer_sizes", nargs="+", type=int, help="Sizes of layers in the neural network")
    parser.add_argument("--memory_size", type=int, default=1000, help="Size of the replay memory")
    
    parser.add_argument("--epsilon_start_value", type=float, default=0.9, help="Initial epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_end_value", type=float, default=0.1, help="Final epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_num_steps", type=int, default=300_000, help="Number of steps to anneal epsilon")
    
    parser.add_argument("--total_train_steps", type=int, default=1_000_000, help="Total number of training steps")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_dqn_update", type=int, default=1_000, help="Number of steps to update the target DQN")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda) for training")

    # todo: Add logging params.

    args = parser.parse_args()
    return args