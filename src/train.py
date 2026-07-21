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
from models.temp_scaled_proto import TempScaledProtoNet
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

    val_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_val_classes"],
        transform=val_transform,
        mode="val"
    )

    k_way = config["episode"]["k_way"]
    k_way_val = len(config["dataset"]["meta_val_classes"])
    n_shot = config["episode"]["n_shot"]
    n_query = config["episode"]["n_query"]

    train_sampler = EpisodeSampler(dataset=train_dataset, k_way=k_way, n_shot=n_shot, n_query=n_query)
    val_sampler = EpisodeSampler(dataset=val_dataset, k_way=k_way_val, n_shot=n_shot, n_query=n_query)

    backbone = DINOv2Backbone(
        pretrained=config["backbone"]["pretrained"],
        freeze=config["backbone"]["freeze"]
    )
    model = PrototypicalNet(backbone=backbone).to(device)

    uncertainty_mode = config["training"].get("uncertainty_mode", "edl")
    
    edl_head = None
    temp_head = None
    optimizer = None
    scheduler = None

    if uncertainty_mode == "edl":
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
    elif uncertainty_mode == "temp_scaled":
        temp_head = TempScaledProtoNet(initial_temp=1.0).to(device)
        optimizer = torch.optim.Adam(temp_head.parameters(), lr=config["training"]["lr"])
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

    run_name = config["wandb"].get("name", f"ablation-{uncertainty_mode}")
    wandb.init(
        project=config["wandb"].get("project", "dermproto-edl"),
        entity=config["wandb"].get("entity"),
        name=run_name,
        config=config
    )
    if args.mode == "train":
        train_model(
            model=model,
            edl_head=edl_head,
            temp_head=temp_head,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            start_epoch=start_epoch,
            initial_best_val_acc=best_val_acc_resumed,
            k_way_val=k_way_val,
            uncertainty_mode=uncertainty_mode
        )

def train_model(model, edl_head, temp_head, train_sampler, val_sampler, optimizer, scheduler, device, config,
                start_epoch: int = 1, initial_best_val_acc: float = 0.0, k_way_val: int = 2, uncertainty_mode: str = "edl"):
    best_val_acc = initial_best_val_acc
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    total_epochs = config["training"]["epochs"]
    ep_per_epoch = config["training"]["episodes_per_epoch"]
    val_every = config["training"]["val_every"]
    val_episodes = config["training"]["val_episodes"]
    
    k_way = config["episode"]["k_way"]
    n_shot = config["episode"]["n_shot"]
    distance_metric = config["episode"].get("distance_metric", "euclidean")
    use_dispersion = config["edl"].get("use_dispersion", True)

    kl_start = config["edl"]["kl_annealing_start"]
    kl_end = config["edl"]["kl_annealing_end"]
    kl_max = config["edl"]["kl_max_weight"]

    if start_epoch > total_epochs:
        print(f"All {total_epochs} epochs already completed. Validation accuracy: {best_val_acc:.4f}")
        return

    print(f"\nStarting Training Loop | Mode: {uncertainty_mode} | Metric: {distance_metric} | Dispersion: {use_dispersion}...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.eval()
        if edl_head is not None:
            edl_head.train()
        if temp_head is not None:
            temp_head.train()
        
        epoch_losses = []
        epoch_accs = []
        epoch_mses = []
        epoch_kls = []
        
        for ep_idx in range(ep_per_epoch):
            global_episode = (epoch - 1) * ep_per_epoch + ep_idx
            
            episode = train_sampler.sample_episode()
            support_images = episode["support_images"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)
            
            if optimizer is not None:
                optimizer.zero_grad()
            
            with torch.no_grad():
                dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way, n_shot, distance_metric=distance_metric)
            
            if not use_dispersion:
                disps = torch.zeros_like(disps)

            if uncertainty_mode == "edl":
                edl_out = edl_head(dists, disps)
                probs = edl_out["probs"]
                alpha = edl_out["alpha"]
                loss, mse, kl, kl_w = edl_loss(probs, alpha, query_labels, global_episode, kl_start, kl_end, kl_max)
                epoch_mses.append(mse.item())
                epoch_kls.append(kl.item())

            elif uncertainty_mode == "temp_scaled":
                out = temp_head(dists)
                probs = out["probs"]
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out["logits"], query_labels)
                kl_w = 0.0

            elif uncertainty_mode == "plain_proto":
                logits = -dists
                probs = torch.softmax(logits, dim=-1)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, query_labels)
                kl_w = 0.0

            if optimizer is not None:
                loss.backward()
                optimizer.step()
            
            epoch_losses.append(loss.item())
            acc = compute_accuracy(probs.detach(), query_labels)
            epoch_accs.append(acc)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = 0.0
        
        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        
        log_dict = {
            "train/loss": mean_loss,
            "train/acc": mean_acc,
            "epoch": epoch,
            "lr": current_lr
        }
        if uncertainty_mode == "edl":
            log_dict["train/mse"] = np.mean(epoch_mses)
            log_dict["train/kl"] = np.mean(epoch_kls)
            log_dict["train/kl_weight"] = kl_w

        print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f}")
        wandb.log(log_dict)
        
        if epoch % val_every == 0:
            model.eval()
            if edl_head is not None:
                edl_head.eval()
            if temp_head is not None:
                temp_head.eval()
            
            val_accs = []
            val_uncertainties = []
            
            with torch.no_grad():
                for _ in range(val_episodes):
                    episode = val_sampler.sample_episode()
                    support_images = episode["support_images"].to(device)
                    query_images = episode["query_images"].to(device)
                    query_labels = episode["query_labels"].to(device)
                    
                    dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way_val, n_shot, distance_metric=distance_metric)
                    if not use_dispersion:
                        disps = torch.zeros_like(disps)

                    if uncertainty_mode == "edl":
                        edl_out = edl_head(dists, disps)
                        acc = compute_accuracy(edl_out["probs"], query_labels)
                        u = edl_out["uncertainty"].mean().item()
                    elif uncertainty_mode == "temp_scaled":
                        out = temp_head(dists)
                        acc = compute_accuracy(out["probs"], query_labels)
                        u = out["uncertainty"].mean().item()
                    elif uncertainty_mode == "plain_proto":
                        probs = torch.softmax(-dists, dim=-1)
                        acc = compute_accuracy(probs, query_labels)
                        u = (1.0 - probs.max(dim=-1)[0]).mean().item()
                    
                    val_accs.append(acc)
                    val_uncertainties.append(u)
                    
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
                    "model_state_dict": model.state_dict(),
                    "edl_head_state_dict": edl_head.state_dict() if edl_head else None,
                    "temp_head_state_dict": temp_head.state_dict() if temp_head else None,
                    "best_val_acc": best_val_acc,
                    "config": config
                }
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}_acc{mean_val_acc:.4f}.pt")
                torch.save(checkpoint_data, checkpoint_path)
                print(f" >>> New best validation accuracy saved at: {checkpoint_path}")
                wandb.save(checkpoint_path)

        latest_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "edl_head_state_dict": edl_head.state_dict() if edl_head else None,
            "temp_head_state_dict": temp_head.state_dict() if temp_head else None,
            "best_val_acc": best_val_acc,
            "config": config
        }
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(latest_data, latest_path)

    print("\nAblation Run Completed!")
    wandb.finish()

if __name__ == "__main__":
    main()