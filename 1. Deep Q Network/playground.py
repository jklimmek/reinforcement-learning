import argparse
import gym
import torch

from definitions import DQN


def parse_arguments():
    parser = argparse.ArgumentParser(description="Play with trained model.")
    
    parser.add_argument("--model", type=str, help="Path to trained model.")
    parser.add_argument("--layer_sizes", nargs="+", type=int, help="Sizes of layers in the neural network.")

    parser.add_argument("--env_name", type=str, default="LunarLander-v2", help="Name of the environment.")
    parser.add_argument("--action_space", type=int, help="Action space dimension.")
    parser.add_argument("--observation_space", type=int, help="Observation space dimension.")
    parser.add_argument("--total_steps", type=int, default=500, help="Total number of steps for running the environment.")
    parser.add_argument("--threshold", type=int, default=200, help="Threshold to consider enviroment as solved.")
    args = parser.parse_args()

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    return config


def main():
    config = parse_arguments()
    model = DQN(config)
    model.load_state_dict(torch.load(config["model"]))

    env = gym.make(config["env_name"], render_mode="human")
    observation, _ = env.reset()

    total_reward = 0

    for _ in range(config["total_steps"]):
        action = model(torch.tensor(observation)).argmax().item()
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            if total_reward >= config["threshold"]:
                print(f"Agent has beaten enviroment! Total reward: {total_reward:.2f}")
            else:
                print(f"Episode has ended. Total reward: {total_reward:.2f}")

            observation, _ = env.reset()
            total_reward = 0

    env.close()


if __name__ == "__main__":
    main()