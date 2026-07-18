import torch
import torch.nn as nn

class PrototypicalNet(nn.Module):
    """
    Prototypical Network with a pluggable backbone.
    Implements standard Snell et al. few-shot metric training.
    """
    
    def __init__(self, backbone: nn.Module):
        """
        Args:
            backbone: Any encoder module that implements `.encode(x)` and
                      provides `.embedding_dim`.
                      Pass DINOv2Backbone for Phase 3+.
                      Pass legacy ResNet-18 wrapper for Phase 2 baseline runs.
        """
        super().__init__()
        self.backbone = backbone

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from image tensor.
        Input: (B, 3, 224, 224)
        Output: (B, embedding_dim)
        """
        return self.backbone.encode(x)

    def compute_prototypes(self, support_embeddings: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
        """
        Computes prototypes (centroids) for each class in the support set.
        support_embeddings: (K * N, embedding_dim)
        Returns: (K, embedding_dim)
        """
        # Reshape to (K, N, embedding_dim) and take the mean along the shot dimension
        reshaped = support_embeddings.view(k_way, n_shot, -1)
        prototypes = reshaped.mean(dim=1)
        return prototypes

    def compute_distances(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Computes squared Euclidean distance from each query to each class prototype.
        query_embeddings: (K * Q, embedding_dim)
        prototypes: (K, embedding_dim)
        Returns: (K * Q, K) squared Euclidean distances
        """
        q_size = query_embeddings.size(0)
        k_way = prototypes.size(0)
        
        # Expand tensors to utilize PyTorch broadcasting
        q_expanded = query_embeddings.unsqueeze(1).expand(q_size, k_way, -1)
        p_expanded = prototypes.unsqueeze(0).expand(q_size, k_way, -1)
        
        # Calculate squared Euclidean distance
        dists = torch.sum((q_expanded - p_expanded) ** 2, dim=-1)
        return dists

    def forward(self, support_images: torch.Tensor, query_images: torch.Tensor, k_way: int, n_shot: int) -> torch.Tensor:
        """
        Episodic forward pass.
        support_images: (K * N, 3, 224, 224)
        query_images: (K * Q, 3, 224, 224)
        Returns Logites: (K * Q, K) (negative squared Euclidean distances)
        """
        # Extract features for both support and query sets
        support_embeddings = self.encode(support_images)
        query_embeddings = self.encode(query_images)
        
        # Calculate prototypes and distance matrix
        prototypes = self.compute_prototypes(support_embeddings, k_way, n_shot)
        dists = self.compute_distances(query_embeddings, prototypes)
        
        # Return negative distance as classification logits
        return -dists


def compute_accuracy(logits: torch.Tensor, query_labels: torch.Tensor) -> float:
    """
    Computes top-1 accuracy for episodic queries.
    logits: (K * Q, K)
    query_labels: (K * Q,) values in [0, K-1]
    Returns: accuracy float value in [0, 1]
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == query_labels).sum().item()
    return correct / len(query_labels)


if __name__ == "__main__":
    import sys
    import os
    # Append parent folder to sys.path so we can import backbone
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.backbone import DINOv2Backbone
    
    print("Running standalone shape verification test...")
    backbone = DINOv2Backbone(pretrained=False, freeze=True)
    model = PrototypicalNet(backbone=backbone)
    
    # 5-way 1-shot setup with 15 queries per class
    k_way = 5
    n_shot = 1
    n_query = 15
    
    dummy_support = torch.randn(k_way * n_shot, 3, 224, 224)
    dummy_query = torch.randn(k_way * n_query, 3, 224, 224)
    
    logits = model(dummy_support, dummy_query, k_way, n_shot)
    
    assert logits.shape == (k_way * n_query, k_way), f"Expected shape (75, 5), got {logits.shape}"
    print("Shape verification PASSED! Logits shape:", logits.shape)