import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set reproducibility seeds for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seeds locked with seed: {seed}")

def resolve_data_root(config: dict) -> str:
    """
    Return HAM10000 root path: Kaggle path if it exists, else local.
    """
    kaggle_path = config.get("paths", {}).get("kaggle_ham10000")
    local_path = config.get("paths", {}).get("data_raw")
    if kaggle_path and os.path.exists(kaggle_path):
        print(f"Data root resolved to Kaggle environment path: {kaggle_path}")
        return kaggle_path
    print(f"Data root resolved to local environment path: {local_path}")
    return local_path