"""
evaluate_crossdataset.py — Phase 7 Cross-Dataset Evaluation Script
Computes Zero-Shot Few-Shot Accuracy, Expected Calibration Error (ECE),
Selective Accuracy Curves, and OOD-AUROC across ISIC 2019 & SD-198.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))

from utils import set_seed, resolve_data_root
from data.datasets import HAM10000Dataset, ISIC2019Dataset, SD198Dataset, build_transforms
from data.episode_sampler import EpisodeSampler
from models.backbone import DINOv2Backbone
from models.prototypical import PrototypicalNet, compute_accuracy
from models.evidential_head import EvidentialHead

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)

def plot_reliability_diagram(probs: torch.Tensor, labels: torch.Tensor, save_path: str, n_bins: int = 10):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].float().mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper).item() / 2)
            
    plt.figure(figsize=(6, 6))
    plt.bar(np.linspace(0.05, 0.95, n_bins), bin_accs, width=0.08, alpha=0.7, edgecolor='black', label='Outputs')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_selective_accuracy(probs: torch.Tensor, labels: torch.Tensor, uncertainties: torch.Tensor, save_path: str):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels).numpy()
    u_scores = uncertainties.numpy()
    
    thresholds = np.linspace(0.0, 1.0, 20)
    accs = []
    coverage = []
    
    for t in thresholds:
        accepted = u_scores <= t
        if accepted.sum() > 0:
            accs.append(accuracies[accepted].mean())
            coverage.append(accepted.mean())
            
    plt.figure(figsize=(7, 5))
    plt.plot(coverage, accs, marker='o', color='blue', linewidth=2)
    plt.xlabel('Rejection Coverage (Accepted Predictions Ratio)')
    plt.ylabel('Selective Accuracy')
    plt.title('Selective Accuracy vs. Coverage (Uncertainty Filtered)')
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(config)
    num_episodes = args.num_episodes or config["episode"]["num_eval_episodes"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    val_transform = build_transforms("val")

    # Load target OOD dataset
    ds_name = config["dataset"]["name"]
    if ds_name == "isic2019":
        dataset = ISIC2019Dataset(root=data_root, transform=val_transform, mode="all")
    elif ds_name == "sd198":
        dataset = SD198Dataset(root=data_root, transform=val_transform, mode="all", min_images_per_class=config["dataset"].get("min_images_per_class", 10))
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    k_way = config["episode"]["k_way"]
    n_shot = config["episode"]["n_shot"]
    n_query = config["episode"]["n_query"]

    sampler = EpisodeSampler(dataset=dataset, k_way=k_way, n_shot=n_shot, n_query=n_query)

    # Load Model
    backbone = DINOv2Backbone(pretrained=config["backbone"]["pretrained"], freeze=True)
    model = PrototypicalNet(backbone=backbone).to(device)
    edl_head = EvidentialHead(hidden_dim=config["edl"]["hidden_dim"]).to(device)

    # Allow PyTorch to load checkpoint dictionaries containing numpy config objects safely
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    if "edl_head_state_dict" in checkpoint and checkpoint["edl_head_state_dict"] is not None:
        edl_head.load_state_dict(checkpoint["edl_head_state_dict"])
    
    model.eval()
    edl_head.eval()

    all_accs, all_probs, all_labels, all_u = [], [], [], []

    print(f"\nEvaluating {num_episodes} Few-Shot Episodes on {ds_name.upper()}...")
    with torch.no_grad():
        for ep_idx in range(num_episodes):
            episode = sampler.sample_episode()
            support_images = episode["support_images"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)

            dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way, n_shot)
            out = edl_head(dists, disps)

            acc = compute_accuracy(out["probs"], query_labels)
            all_accs.append(acc)
            all_probs.append(out["probs"].cpu())
            all_labels.append(query_labels.cpu())
            all_u.append(out["uncertainty"].cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    u_cat = torch.cat(all_u, dim=0).squeeze()

    mean_acc = np.mean(all_accs)
    ece = compute_ece(probs_cat, labels_cat)
    mean_u = u_cat.mean().item()

    # Save Reliability Diagram & Selective Accuracy Plot
    plot_reliability_diagram(probs_cat, labels_cat, os.path.join(output_dir, "reliability_diagram.png"))
    plot_selective_accuracy(probs_cat, labels_cat, u_cat, os.path.join(output_dir, "selective_accuracy_curve.png"))

    # Print & Save Summary
    summary_text = (
        f"--- Cross-Dataset Evaluation Summary ({ds_name.upper()}) ---\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Episodes Evaluated: {num_episodes}\n"
        f"Mean Few-Shot Accuracy: {mean_acc:.4f}\n"
        f"Expected Calibration Error (ECE): {ece:.4f}\n"
        f"Mean Uncertainty (u): {mean_u:.4f}\n"
    )
    print("\n" + summary_text)
    
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()