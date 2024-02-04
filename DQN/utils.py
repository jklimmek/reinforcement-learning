import torch
import numpy as np


def seed_everything(env, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False