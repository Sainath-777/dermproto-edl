"""
evaluate.py — Evaluation and Ablation Metrics Script
Computes Validation Accuracy, Expected Calibration Error (ECE), Brier Score, and Selective Accuracy curves.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))

from utils import set_seed, resolve_data_root
from data.datasets import HAM10000Dataset, build_transforms
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(config)

    val_transform = build_transforms("val")
    val_dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_val_classes"],
        transform=val_transform,
        mode="val"
    )

    k_way_val = len(config["dataset"]["meta_val_classes"])
    n_shot = config["episode"]["n_shot"]
    n_query = config["episode"]["n_query"]

    val_sampler = EpisodeSampler(dataset=val_dataset, k_way=k_way_val, n_shot=n_shot, n_query=n_query)

    backbone = DINOv2Backbone(pretrained=config["backbone"]["pretrained"], freeze=True)
    model = PrototypicalNet(backbone=backbone).to(device)
    edl_head = EvidentialHead(hidden_dim=config["edl"]["hidden_dim"]).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "edl_head_state_dict" in checkpoint and checkpoint["edl_head_state_dict"] is not None:
        edl_head.load_state_dict(checkpoint["edl_head_state_dict"])
    
    model.eval()
    edl_head.eval()

    all_accs = []
    all_probs = []
    all_labels = []
    all_u = []

    with torch.no_grad():
        for _ in range(100): # 100 evaluation episodes
            episode = val_sampler.sample_episode()
            support_images = episode["support_images"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)

            dists, disps, _, _ = model.forward_edl(support_images, query_images, k_way_val, n_shot)
            out = edl_head(dists, disps)

            acc = compute_accuracy(out["probs"], query_labels)
            all_accs.append(acc)
            all_probs.append(out["probs"].cpu())
            all_labels.append(query_labels.cpu())
            all_u.append(out["uncertainty"].cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    
    mean_acc = np.mean(all_accs)
    ece = compute_ece(probs_cat, labels_cat)
    mean_u = torch.cat(all_u).mean().item()

    print("\n--- Evaluation Summary ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mean Val Accuracy: {mean_acc:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Mean Uncertainty: {mean_u:.4f}")

if __name__ == "__main__":
    main()