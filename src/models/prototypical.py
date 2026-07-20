import torch
import torch.nn as nn

class PrototypicalNet(nn.Module):
    """
    Prototypical Network with a pluggable backbone.
    Supports standard distance forward pass as well as EDL feature extraction.
    """
    
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def compute_prototypes(self, support_embeddings: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
        reshaped = support_embeddings.view(k_way, n_shot, -1)
        prototypes = reshaped.mean(dim=1)
        return prototypes

    def compute_distances(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        q_size = query_embeddings.size(0)
        k_way = prototypes.size(0)
        
        q_expanded = query_embeddings.unsqueeze(1).expand(q_size, k_way, -1)
        p_expanded = prototypes.unsqueeze(0).expand(q_size, k_way, -1)
        
        dists = torch.sum((q_expanded - p_expanded) ** 2, dim=-1)
        return dists

    def compute_dispersion(self, support_embeddings: torch.Tensor, prototypes: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
        """
        Computes per-class support-set dispersion (σ_k).
        σ_k = (1/N) Σ_i ||f(x_{k,i}) − c_k||²
        """
        support_reshaped = support_embeddings.view(k_way, n_shot, -1)
        proto_expanded = prototypes.unsqueeze(1).expand_as(support_reshaped)
        sq_diffs = torch.sum((support_reshaped - proto_expanded) ** 2, dim=-1)
        dispersions = sq_diffs.mean(dim=1)
        return dispersions

    def forward(self, support_images: torch.Tensor, query_images: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
        """Standard Phase 2/3 distance logits pass"""
        support_embeddings = self.encode(support_images)
        query_embeddings = self.encode(query_images)
        prototypes = self.compute_prototypes(support_embeddings, k_way, n_shot)
        dists = self.compute_distances(query_embeddings, prototypes)
        return -dists

    def forward_edl(self, support_images: torch.Tensor, query_images: torch.Tensor, k_way: int, n_shot: int) -> tuple:
        """Phase 4 forward pass returning distance and dispersion features"""
        support_embeddings = self.encode(support_images)
        query_embeddings = self.encode(query_images)
        prototypes = self.compute_prototypes(support_embeddings, k_way, n_shot)
        distances = self.compute_distances(query_embeddings, prototypes)
        dispersions = self.compute_dispersion(support_embeddings, prototypes, k_way, n_shot)
        return distances, dispersions, prototypes, query_embeddings


def compute_accuracy(logits_or_probs: torch.Tensor, query_labels: torch.Tensor) -> float:
    preds = torch.argmax(logits_or_probs, dim=1)
    correct = (preds == query_labels).sum().item()
    return correct / len(query_labels)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.backbone import DINOv2Backbone
    
    print("Running standalone shape verification test...")
    backbone = DINOv2Backbone(pretrained=False, freeze=True)
    model = PrototypicalNet(backbone=backbone)
    
    k_way, n_shot, n_query = 5, 5, 15
    dummy_support = torch.randn(k_way * n_shot, 3, 224, 224)
    dummy_query = torch.randn(k_way * n_query, 3, 224, 224)
    
    dists, disps, protos, q_emb = model.forward_edl(dummy_support, dummy_query, k_way, n_shot)
    assert dists.shape == (75, 5)
    assert disps.shape == (5,)
    print("EDL forward shape verification PASSED!")