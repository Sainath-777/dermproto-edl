import os
import sys
import argparse
import yaml
import wandb
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))

from utils import set_seed, resolve_data_root, load_checkpoint_if_exists
from data.datasets import HAM10000Dataset, build_transforms
from data.episode_sampler import EpisodeSampler
from models.backbone import DINOv2Backbone
from models.prototypical import PrototypicalNet, compute_accuracy
from models.evidential_head import EvidentialHead
from losses.edl_loss import edl_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running execution loop on device: {device}")

    data_root = resolve_data_root(config)

    train_transform = build_transforms("train")
    val_transform = build_transforms("val")

    train_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_train_classes"],
        transform=train_transform,
        mode="train"
    )

    # Validation dataset evaluates on unseen rare classes (df, vasc)
    val_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_val_classes"],
        transform=val_transform,
        mode="val"
    )

    k_way = config["episode"]["k_way"]                           # 5-way for meta-training
    k_way_val = len(config["dataset"]["meta_val_classes"])       # 2-way for unseen meta-validation (df, vasc)
    n_shot = config["episode"]["n_shot"]                         # 5-shot
    n_query = config["episode"]["n_query"]                       # 15 query images

    train_sampler = EpisodeSampler(dataset=train_dataset, k_way=k_way, n_shot=n_shot, n_query=n_query)
    val_sampler = EpisodeSampler(dataset=val_dataset, k_way=k_way_val, n_shot=n_shot, n_query=n_query)

    backbone = DINOv2Backbone(
        pretrained=config["backbone"]["pretrained"],
        freeze=config["backbone"]["freeze"]
    )
    model = PrototypicalNet(backbone=backbone).to(device)

    edl_head = EvidentialHead(hidden_dim=config["edl"]["hidden_dim"]).to(device)

    optimizer = torch.optim.Adam(
        edl_head.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["scheduler_step_size"],
        gamma=config["training"]["scheduler_gamma"]
    )

    start_epoch, best_val_acc_resumed = load_checkpoint_if_exists(
        checkpoint_dir=config["paths"]["checkpoints"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        edl_head=edl_head
    )

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name="phase4-evidential-head",
        config=config
    )

    if args.mode == "train":
        train_model(
            model=model,
            edl_head=edl_head,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            start_epoch=start_epoch,
            initial_best_val_acc=best_val_acc_resumed,
            k_way_val=k_way_val
        )

def train_model(model, edl_head, train_sampler, val_sampler, optimizer, scheduler, device, config,
                start_epoch: int = 1, initial_best_val_acc: float = 0.0, k_way_val: int = 2):
    best_val_acc = initial_best_val_acc
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    total_epochs = config["training"]["epochs"]
    ep_per_epoch = config["training"]["episodes_per_epoch"]
    val_every = config["training"]["val_every"]
    val_episodes = config["training"]["val_episodes"]
    
    k_way = config["episode"]["k_way"]
    n_shot = config["episode"]["n_shot"]

    kl_start = config["edl"]["kl_annealing_start"]
    kl_end = config["edl"]["kl_annealing_end"]
    kl_max = config["edl"]["kl_max_weight"]

    if start_epoch > total_epochs:
        print(f"All {total_epochs} epochs already completed. Validation accuracy: {best_val_acc:.4f}")
        return

    print(f"\nStarting Phase 4 Evidential Training Loop (5-way train / {k_way_val}-way val)...")
    for epoch in range(start_epoch, total_epochs + 1):
        model.eval() # Backbone frozen
        edl_head.train()
        
        epoch_losses = []
        epoch_mses = []
        epoch_kls = []
        epoch_accs = []
        
        for ep_idx in range(ep_per_epoch):
            global_episode = (epoch - 1) * ep_per_epoch + ep_idx
            
            episode = train_sampler.sample_episode()
            
            support_images = episode["support_images"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way, n_shot)
            
            edl_out = edl_head(dists, disps)
            probs = edl_out["probs"]
            alpha = edl_out["alpha"]
            
            loss, mse, kl, kl_w = edl_loss(probs, alpha, query_labels, global_episode, kl_start, kl_end, kl_max)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_mses.append(mse.item())
            epoch_kls.append(kl.item())
            
            acc = compute_accuracy(probs.detach(), query_labels)
            epoch_accs.append(acc)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = 0.0
        
        mean_loss = np.mean(epoch_losses)
        mean_mse = np.mean(epoch_mses)
        mean_kl = np.mean(epoch_kls)
        mean_acc = np.mean(epoch_accs)
        
        print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {mean_loss:.4f} | MSE: {mean_mse:.4f} | KL: {mean_kl:.4f} | Acc: {mean_acc:.4f} | KL_w: {kl_w:.4f}")
        wandb.log({
            "train/loss": mean_loss,
            "train/mse": mean_mse,
            "train/kl": mean_kl,
            "train/acc": mean_acc,
            "train/kl_weight": kl_w,
            "epoch": epoch,
            "lr": current_lr
        })
        
        if epoch % val_every == 0:
            model.eval()
            edl_head.eval()
            val_accs = []
            val_uncertainties = []
            
            with torch.no_grad():
                for _ in range(val_episodes):
                    episode = val_sampler.sample_episode()
                    
                    support_images = episode["support_images"].to(device)
                    query_images = episode["query_images"].to(device)
                    query_labels = episode["query_labels"].to(device)
                    
                    # 2-way evaluation on unseen classes
                    dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way_val, n_shot)
                    edl_out = edl_head(dists, disps)
                    
                    acc = compute_accuracy(edl_out["probs"], query_labels)
                    val_accs.append(acc)
                    val_uncertainties.append(edl_out["uncertainty"].mean().item())
                    
            mean_val_acc = np.mean(val_accs)
            mean_val_u = np.mean(val_uncertainties)
            print(f" >>> Validation ({k_way_val}-way Unseen) | Epoch {epoch:3d} | Val Acc: {mean_val_acc:.4f} | Mean Uncertainty: {mean_val_u:.4f}")
            wandb.log({
                "val/acc": mean_val_acc,
                "val/uncertainty": mean_val_u,
                "epoch": epoch
            })
            
            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                checkpoint_data = {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "model_state_dict": model.state_dict(),
                    "edl_head_state_dict": edl_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config
                }
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}_acc{mean_val_acc:.4f}.pt")
                torch.save(checkpoint_data, checkpoint_path)
                print(f" >>> New best validation accuracy saved at: {checkpoint_path}")
                wandb.save(checkpoint_path)

        latest_data = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "model_state_dict": model.state_dict(),
            "edl_head_state_dict": edl_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config
        }
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(latest_data, latest_path)
        print(f" >>> Saved current progress to: {latest_path}")

    print("\nPhase 4 Stage 1 Sanity Run Completed!")
    wandb.finish()

if __name__ == "__main__":
    main()