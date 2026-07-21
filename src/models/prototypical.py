"""
prototypical.py — Prototypical Networks (Snell et al., 2017) with Dispersion (σ_k) calculation
and optional Cosine Distance metric for A3 ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_prototypes(support_embeddings: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
    """
    Computes class prototypes (centroids) by averaging support embeddings.
    
    Args:
        support_embeddings: (K * N, D)
        k_way: K
        n_shot: N
    
    Returns:
        prototypes: (K, D)
    """
    D = support_embeddings.size(-1)
    reshaped = support_embeddings.view(k_way, n_shot, D)
    prototypes = reshaped.mean(dim=1)
    return prototypes


def compute_dispersion(support_embeddings: torch.Tensor, prototypes: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
    """
    Computes support-set dispersion σ_k for each class.
    σ_k = (1/N) * sum_i || x_{k,i} - c_k ||^2
    
    Args:
        support_embeddings: (K * N, D)
        prototypes: (K, D)
    
    Returns:
        dispersions: (K,)
    """
    D = support_embeddings.size(-1)
    reshaped = support_embeddings.view(k_way, n_shot, D)  # (K, N, D)
    proto_expanded = prototypes.unsqueeze(1)               # (K, 1, D)
    
    sq_dists = torch.sum((reshaped - proto_expanded) ** 2, dim=-1) # (K, N)
    dispersions = sq_dists.mean(dim=1)                             # (K,)
    return dispersions


def compute_distances(query_embeddings: torch.Tensor, prototypes: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """
    Computes distances from query embeddings to prototypes.
    
    Args:
        query_embeddings: (n_queries, D)
        prototypes: (K, D)
        metric: "euclidean" or "cosine"
    
    Returns:
        distances: (n_queries, K)
    """
    if metric == "cosine":
        # Cosine distance = 1 - Cosine Similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        sim = torch.mm(query_norm, proto_norm.t()) # (n_queries, K)
        distances = 1.0 - sim
    else:
        # Squared Euclidean distance
        n_queries = query_embeddings.size(0)
        K = prototypes.size(0)
        
        q_ext = query_embeddings.unsqueeze(1).expand(n_queries, K, -1)
        p_ext = prototypes.unsqueeze(0).expand(n_queries, K, -1)
        distances = torch.sum((q_ext - p_ext) ** 2, dim=-1)
        
    return distances


class PrototypicalNet(nn.Module):
    """
    Prototypical Network wrapper combining backbone and distance/dispersion calculations.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images: torch.Tensor, query_images: torch.Tensor, k_way: int, n_shot: int):
        support_embeddings = self.backbone(support_images)
        query_embeddings = self.backbone(query_images)
        
        prototypes = compute_prototypes(support_embeddings, k_way, n_shot)
        distances = compute_distances(query_embeddings, prototypes, metric="euclidean")
        
        logits = -distances
        return logits, prototypes

    def forward_edl(self, support_images: torch.Tensor, query_images: torch.Tensor, k_way: int, n_shot: int, distance_metric: str = "euclidean"):
        """
        Phase 4+ EDL Forward pass returning raw features required by EvidentialHead.
        """
        support_embeddings = self.backbone(support_images)
        query_embeddings = self.backbone(query_images)
        
        prototypes = compute_prototypes(support_embeddings, k_way, n_shot)
        dispersions = compute_dispersion(support_embeddings, prototypes, k_way, n_shot)
        distances = compute_distances(query_embeddings, prototypes, metric=distance_metric)
        
        return distances, dispersions, prototypes, query_embeddings


def compute_accuracy(logits_or_probs: torch.Tensor, query_labels: torch.Tensor) -> float:
    preds = torch.argmax(logits_or_probs, dim=1)
    acc = (preds == query_labels).float().mean().item()
    return acc