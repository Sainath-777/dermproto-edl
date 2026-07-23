import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
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
    local_path = config["paths"]["data_raw"]
    # Check dataset-specific Kaggle paths dynamically
    kaggle_path = (
        config["paths"].get("kaggle_isic2019") or 
        config["paths"].get("kaggle_sd198") or 
        config["paths"].get("kaggle_ham10000") or
        config["paths"].get("kaggle_path")
    )
    
    if os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
        print(f"Data root resolved locally: {local_path}")
        return local_path
    elif kaggle_path and os.path.exists(kaggle_path):
        print(f"Data root resolved on Kaggle: {kaggle_path}")
        return kaggle_path
    else:
        os.makedirs(local_path, exist_ok=True)
        print(f"Data root directory created locally (empty): {local_path}")
        return local_path

def load_checkpoint_if_exists(checkpoint_dir: str, model, optimizer=None, scheduler=None, edl_head=None):
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
    if edl_head is not None and "edl_head_state_dict" in checkpoint_data:
        edl_head.load_state_dict(checkpoint_data["edl_head_state_dict"])
        print("[Checkpoint] Evidential head weights restored.")

    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    last_epoch = checkpoint_data["epoch"]
    best_val_acc = checkpoint_data.get("best_val_acc", 0.0)
    start_epoch = last_epoch + 1

    print(f"Resuming from Epoch {start_epoch}/{checkpoint_data.get('total_epochs', '?')} | Best Val Acc so far: {best_val_acc:.4f}")
    return start_epoch, best_val_acc