import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets deterministic seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seed set: {seed}")

def resolve_data_root(config: dict) -> str:
    """
    Resolves data directory path, checking for local raw folders
    or Kaggle's standard mount path.
    """
    local_path = config["paths"]["data_raw"]
    kaggle_path = config["paths"]["kaggle_ham10000"]
    
    if os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
        print(f"Data root resolved locally: {local_path}")
        return local_path
    elif os.path.exists(kaggle_path):
        print(f"Data root resolved on Kaggle: {kaggle_path}")
        return kaggle_path
    else:
        # Fallback to local path (creates directory structure)
        os.makedirs(local_path, exist_ok=True)
        print(f"Data root directory created locally (empty): {local_path}")
        return local_path

def load_checkpoint_if_exists(checkpoint_dir: str, model, optimizer=None, scheduler=None):
    """
    Auto-resume logic. Scans checkpoint_dir for a 'latest_checkpoint.pt' file,
    loads it if it exists, and restores model state (and optionally optimizer/scheduler).

    Returns:
        start_epoch (int): The epoch to resume FROM (i.e., last completed epoch + 1).
                           Returns 1 if no checkpoint found (fresh start).
        best_val_acc (float): Best validation accuracy from the loaded checkpoint.
                              Returns 0.0 if no checkpoint found.

    Prints a clear resume or fresh-start message per Rule 9 (verbose prints contract).
    """
    if not os.path.isdir(checkpoint_dir):
        print("No checkpoint directory found. Starting training from scratch (Epoch 1)...")
        return 1, 0.0

    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    if not os.path.exists(latest_path):
        print("No 'latest_checkpoint.pt' found. Starting training from scratch (Epoch 1)...")
        return 1, 0.0

    print(f"Found checkpoint to resume: {latest_path}")
    checkpoint_data = torch.load(latest_path, map_location="cpu")

    model.load_state_dict(checkpoint_data["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    last_epoch = checkpoint_data["epoch"]
    best_val_acc = checkpoint_data.get("best_val_acc", 0.0)
    start_epoch = last_epoch + 1

    print(f"Resuming from Epoch {start_epoch}/{checkpoint_data.get('total_epochs', '?')} | Best Val Acc so far: {best_val_acc:.4f}")
    return start_epoch, best_val_acc