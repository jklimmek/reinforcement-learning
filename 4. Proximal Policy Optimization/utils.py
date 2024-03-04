import datetime
import os

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.wrappers import (
    RecordEpisodeStatistics, 
    ResizeObservation, 
    GrayScaleObservation, 
    FrameStack
)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)


def seed_almost_everything(seed=None):
    """Set seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_env(seed):
    """Create Atari environment and add usefull functionality."""
    def env_func():
        env = gym.make("BreakoutNoFrameskip-v4")

        # Some Atair environments are stationary until first FIRE action is performed.
        # Does FIRE action, so the model does not have to learn it itself.
        env = FireResetEnv(env)

        # Record total reward as well as length of the episode.
        env = RecordEpisodeStatistics(env)

        # Stay idle for up to 30 frames. 
        # This wrapper adds stochasticity to the environments.
        env = NoopResetEnv(env, noop_max=30)

        # Skip 4 initial frames and execute selected action for the next 4 frames.
        # Also takes maximum pixel value over last two frames.
        # This wrapper hepls to save computational time.
        env = MaxAndSkipEnv(env, skip=4)

        # Treat end of life as the end of episode.
        env = EpisodicLifeEnv(env)

        # Clips rewards to {-1, 0, -1}.
        # This wrapper helps stabilize training.
        env = ClipRewardEnv(env)

        # Resize, convert to grayscale and stack 4 frames.
        # This preprocessing is suggested in the Atari DQN paper.
        # Resizing and converting to grayscale makes computations simpler.
        # Stacking frames adds information about the velocity of the ball.
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayScaleObservation(env)
        env = FrameStack(env , 4)

        # Seed environment for reproducibility.
        # However it has some issues.
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    return env_func


def layer_init(layer, gain=1.0, bias=0.0):
    """Initialize layer orthogonaly."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, val=bias)
    return layer


def get_run_name():
    """Create run's name for Tensorboard."""
    timestamp = datetime.datetime.now().strftime("%b-%d__%H-%M-%S")
    folder_name = f"run__{timestamp}"
    return folder_name


def save_checkpoint(path, model, optimizer=None, episode_step=None, global_step=None):
    """Save checkpoint."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "episode_step": episode_step,
            "global_step": global_step

        }, path
    )


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    episode_step = checkpoint["episode_step"]
    global_step = checkpoint["global_step"]
    return episode_step, global_step
