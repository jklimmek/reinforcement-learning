import datetime
import os
import numpy as np
import torch


def seed_everything(env, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def save_dqn(model, path):
    dir_path, name = os.path.split(path)
    os.makedirs(dir_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir_path, name))


def get_run_name():
    timestamp = datetime.datetime.now().strftime("%b-%d__%H-%M-%S")
    folder_name = f"run__{timestamp}"
    return folder_name