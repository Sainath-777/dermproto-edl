import os
import sys
import argparse
import yaml
import wandb
import torch
import torch.nn as nn
import numpy as np

# Ensure execution works from project root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))

from utils import set_seed, resolve_data_root
from data.datasets import HAM10000Dataset, build_transforms
from data.episode_sampler import EpisodeSampler
from models.prototypical import PrototypicalNet, compute_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode")
    args = parser.parse_args()

    # Load configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Locking seeding for reproducibility
    set_seed(config["seed"])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running execution loop on device: {device}")

    # Resolve dataset paths
    data_root = resolve_data_root(config)

    # Setup datasets
    train_transform = build_transforms("train")
    val_transform = build_transforms("val")

    train_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_train_classes"],
        transform=train_transform,
        mode="train"
    )

    val_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_train_classes"],
        transform=val_transform,
        mode="val"
    )

    # Setup episode samplers
    k_way = config["episode"]["k_way"]
    n_shot = config["episode"]["n_shot"]
    n_query = config["episode"]["n_query"]

    train_sampler = EpisodeSampler(
        dataset=train_dataset,
        k_way=k_way,
        n_shot=n_shot,
        n_query=n_query
    )

    val_sampler = EpisodeSampler(
        dataset=val_dataset,
        k_way=k_way,
        n_shot=n_shot,
        n_query=n_query
    )

    # Initialize model
    model = PrototypicalNet(pretrained=config["backbone"]["pretrained"])
    model = model.to(device)

    # Setup loss and optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    # Scheduler: StepLR reduces LR by 50% every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["scheduler_step_size"],
        gamma=config["training"]["scheduler_gamma"]
    )
    
    criterion = nn.CrossEntropyLoss()

    # Initialize W&B tracking
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config
    )

    if args.mode == "train":
        train_model(
            model=model,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config
        )

def train_model(model, train_sampler, val_sampler, optimizer, scheduler, criterion, device, config):
    best_val_acc = 0.0
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    total_epochs = config["training"]["epochs"]
    ep_per_epoch = config["training"]["episodes_per_epoch"]
    val_every = config["training"]["val_every"]
    val_episodes = config["training"]["val_episodes"]
    
    k_way = config["episode"]["k_way"]
    n_shot = config["episode"]["n_shot"]

    print("\nStarting Phase 2 training loop...")
    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accs = []
        
        for _ in range(ep_per_epoch):
            # Sample episode
            episode = train_sampler.sample_episode()
            
            # Load vectors to device
            support_images = episode["support_images"].to(device)
            support_labels = episode["support_labels"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(support_images, query_images, k_way, n_shot)
            loss = criterion(logits, query_labels)
            
            # Optimize weights
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = compute_accuracy(logits.detach(), query_labels)
            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        scheduler.step()
        
        # Log training epoch stats
        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch:3d}/{total_epochs} | Train Loss: {mean_loss:.4f} | Train Acc: {mean_acc:.4f} | LR: {current_lr:.6f}")
        wandb.log({
            "train/loss": mean_loss,
            "train/acc": mean_acc,
            "epoch": epoch,
            "lr": current_lr
        })
        
        # Run validation pass
        if epoch % val_every == 0:
            model.eval()
            val_accs = []
            
            with torch.no_grad():
                for _ in range(val_episodes):
                    episode = val_sampler.sample_episode()
                    
                    support_images = episode["support_images"].to(device)
                    support_labels = episode["support_labels"].to(device)
                    query_images = episode["query_images"].to(device)
                    query_labels = episode["query_labels"].to(device)
                    
                    logits = model(support_images, query_images, k_way, n_shot)
                    acc = compute_accuracy(logits, query_labels)
                    val_accs.append(acc)
                    
            mean_val_acc = np.mean(val_accs)
            print(f" >>> Validation | Epoch {epoch:3d} | Val Acc: {mean_val_acc:.4f}")
            wandb.log({
                "val/acc": mean_val_acc,
                "epoch": epoch
            })
            
            # Checkpoint saving on accuracy increase
            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config
                }
                
                checkpoint_filename = f"best_model_epoch{epoch}_acc{mean_val_acc:.4f}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                torch.save(checkpoint_data, checkpoint_path)
                print(f" >>> New best validation accuracy saved at: {checkpoint_path}")
                wandb.save(checkpoint_path)

    print("\nTraining completed!")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")
    
    # Evaluate go/no-go gate (validation target is 60%)
    if best_val_acc >= 0.60:
        print("GATE PASSED: Proceed to Phase 3")
    else:
        print("GATE FAILED: Debug before Phase 3")
        
    wandb.finish()

if __name__ == "__main__":
    main()