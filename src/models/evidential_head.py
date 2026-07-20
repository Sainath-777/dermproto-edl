"""
evidential_head.py — Support-Set-Consistency-Aware Evidential Head

Novel component of DermProto-EDL.

Takes per-class [distance, dispersion] pairs from the Prototypical Network
and outputs evidence → Dirichlet parameters → (probability, uncertainty).

Architecture:
    Input:  (n_queries, K, 2)  — per-query, per-class [d_k, σ_k] feature
    Hidden: Linear(2 → 64) + ReLU
    Output: Linear(64 → 1) + ReLU → evidence per class  (K,) per query

Dirichlet math:
    α_k = evidence_k + 1            (Dirichlet parameter)
    S   = Σ_k α_k                   (total evidence strength)
    p_k = α_k / S                   (predicted class probability)
    u   = K / S                     (uncertainty score ∈ (0, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Support-set-consistency-aware evidential uncertainty head.

    Args:
        hidden_dim (int): Width of the hidden layer. Default 64.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),       # [d_k, σ_k] → hidden
            # NOTE: LayerNorm intentionally OMITTED here.
            # DINOv2 embeddings are L2-normalized → squared distances bounded in [0,4].
            # LayerNorm would normalize the 64 hidden activations per sample,
            # erasing the absolute magnitude of d_k/σ_k which IS the signal.
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),       # hidden → 1 evidence scalar per class
        )

        self._init_weights()

    def _init_weights(self):
        """
        Conservative initialization. Final Linear initialized near zero so that
        at the start of training, the head outputs small evidence values for all
        classes — preventing early KL divergence domination.
        """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Scale down the final layer specifically
        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)

    def forward(
        self,
        distances: torch.Tensor,
        dispersions: torch.Tensor,
    ) -> dict:
        """
        Args:
            distances:   (n_queries, K) — squared Euclidean distance from each
                         query to each class prototype.
            dispersions: (K,)           — support-set dispersion per class

        Returns:
            dict with keys:
                'evidence'    : (n_queries, K) — raw evidence values ≥ 0
                'alpha'       : (n_queries, K) — Dirichlet parameters (evidence + 1)
                'S'           : (n_queries, 1) — total evidence strength per query
                'probs'       : (n_queries, K) — predicted class probabilities
                'uncertainty' : (n_queries,)   — uncertainty score ∈ (0, 1]
                'k_way'       : int            — number of classes (for u = K/S)
        """
        n_queries, K = distances.shape

        # Broadcast dispersions to (n_queries, K)
        disp_expanded = dispersions.unsqueeze(0).expand(n_queries, K)
        features = torch.stack([distances, disp_expanded], dim=-1)

        # MLP maps each (d_k, σ_k) pair → 1 evidence scalar
        evidence = self.mlp(features.view(-1, 2))          # (n_queries * K, 1)
        evidence = F.relu(evidence)                        # enforce non-negativity
        evidence = evidence.view(n_queries, K)             # (n_queries, K)

        # Dirichlet parameter: α_k = e_k + 1  (always ≥ 1)
        alpha = evidence + 1.0                             # (n_queries, K)

        # Total evidence strength S = Σ_k α_k
        S = alpha.sum(dim=1, keepdim=True)                 # (n_queries, 1)

        # Predicted class probability: p_k = α_k / S
        probs = alpha / S                                  # (n_queries, K)

        # Uncertainty: u = K / S
        uncertainty = K / S.squeeze(1)                    # (n_queries,)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "S": S,
            "probs": probs,
            "uncertainty": uncertainty,
            "k_way": K,
        }


if __name__ == "__main__":
    print("Running EvidentialHead shape verification...")
    K = 5
    Q = 15
    n_queries = K * Q

    head = EvidentialHead(hidden_dim=64)
    distances = torch.rand(n_queries, K)
    dispersions = torch.rand(K)

    out = head(distances, dispersions)

    assert out["evidence"].shape == (n_queries, K)
    assert out["alpha"].shape == (n_queries, K)
    assert out["S"].shape == (n_queries, 1)
    assert out["probs"].shape == (n_queries, K)
    assert out["uncertainty"].shape == (n_queries,)

    prob_sums = out["probs"].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(n_queries), atol=1e-5)
    assert (out["uncertainty"] > 0).all() and (out["uncertainty"] <= 1).all()
    assert (out["alpha"] >= 1.0).all()

    print("EvidentialHead shape verification PASSED.")