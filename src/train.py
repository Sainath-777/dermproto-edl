import os
import argparse
import yaml
import wandb
import torch
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="DermProto-EDL Dry-run Pipeline Test")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    print("Loaded configuration schema successfully:")
    print(yaml.dump(config, indent=2))

    # Initialize Weights & Biases
    wandb_config = config.get("wandb", {})
    run = wandb.init(
        project=wandb_config.get("project", "dermproto-edl"),
        entity=wandb_config.get("entity"),
        config=config
    )

    print("Pipeline OK - Starting dummy training loop...")

    # Log dummy metrics
    epochs = config.get("training", {}).get("epochs", 10)
    for epoch in range(1, epochs + 1):
        # Generate dummy loss and accuracy
        dummy_loss = 2.0 / epoch + (torch.randn(1).item() * 0.05)
        dummy_acc = 0.5 + 0.4 * (1.0 - 1.0 / epoch) + (torch.randn(1).item() * 0.02)
        
        print(f"Epoch {epoch}/{epochs} - Dummy Loss: {dummy_loss:.4f} - Dummy Acc: {dummy_acc:.4f}")
        wandb.log({
            "epoch": epoch,
            "train/loss": dummy_loss,
            "train/acc": dummy_acc
        })

    print("Pipeline run completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main()