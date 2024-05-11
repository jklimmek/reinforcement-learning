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


def make_env():
    """Create Atari environment and add usefull functionality."""
    def env_func():
        # Testing on LunarLander until everything is implemented.
        env = gym.make("LunarLander-v2")

        # Record total reward as well as length of the episode.
        env = RecordEpisodeStatistics(env)

        return env
    return env_func


def layer_init(layer, gain=1.0, bias=None):
    """Initialize layer orthogonaly."""
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if bias is not None:
            nn.init.constant_(layer.bias, val=bias)
    return layer


def get_run_name():
    """Create run's name for Tensorboard."""
    timestamp = datetime.datetime.now().strftime("%b-%d__%H-%M-%S")
    folder_name = f"run__{timestamp}"
    return folder_name


def save_checkpoint(path, model):
    """Save checkpoint."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(path, model):
    """Load checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
