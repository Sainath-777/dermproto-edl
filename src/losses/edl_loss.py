"""
edl_loss.py — EDL Loss with Annealed KL-Divergence Regularization

Implements the Evidential Deep Learning loss per:
  Sensoy et al. 2018 (base formulation)
  + "Revisiting Essential and Nonessential Settings of EDL" 2024 (KL corrections)

Loss = L_mse (data-fit) + λ(t) · L_kl (regularization)
"""

import math
import torch
import torch.nn.functional as F
from torch.special import gammaln


def kl_divergence_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1)) — KL divergence from Dirichlet(alpha) to uniform Dirichlet(1).
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)

    log_B_alpha = gammaln(alpha).sum(dim=1) - gammaln(S.squeeze(1))
    log_B_ones = -gammaln(torch.tensor(float(K), device=alpha.device))

    digamma_alpha = torch.digamma(alpha)
    digamma_S = torch.digamma(S)
    cross_term = ((alpha - 1.0) * (digamma_alpha - digamma_S)).sum(dim=1)

    kl = log_B_ones - log_B_alpha + cross_term
    return kl


def mse_loss_edl(
    probs: torch.Tensor,
    targets_onehot: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    EDL MSE data-fit term.
    """
    S = alpha.sum(dim=1, keepdim=True)
    sq_err = (targets_onehot - probs) ** 2
    variance = probs * (1.0 - probs) / (S + 1.0)
    loss = (sq_err + variance).sum(dim=1)
    return loss


def compute_kl_weight(
    episode: int,
    kl_annealing_start: int,
    kl_annealing_end: int,
    kl_max_weight: float,
) -> float:
    """
    Linear KL annealing schedule.
    """
    if episode < kl_annealing_start:
        return 0.0
    ramp_length = max(kl_annealing_end - kl_annealing_start, 1)
    progress = min((episode - kl_annealing_start) / ramp_length, 1.0)
    return progress * kl_max_weight


def edl_loss(
    probs: torch.Tensor,
    alpha: torch.Tensor,
    targets: torch.Tensor,
    episode: int,
    kl_annealing_start: int,
    kl_annealing_end: int,
    kl_max_weight: float,
) -> tuple:
    """
    Full EDL loss = L_mse + λ(t) · L_kl
    """
    K = probs.size(1)

    targets_onehot = F.one_hot(targets, num_classes=K).float()

    # 1. MSE data-fit term
    mse = mse_loss_edl(probs, targets_onehot, alpha)

    # 2. KL regularization: remove correct class evidence before computing KL
    alpha_tilde = targets_onehot + (1.0 - targets_onehot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde)

    # 3. Current KL weight (annealed)
    kl_w = compute_kl_weight(episode, kl_annealing_start, kl_annealing_end, kl_max_weight)

    # 4. Total loss
    total = mse + kl_w * kl

    return total.mean(), mse.mean(), kl.mean(), kl_w


if __name__ == "__main__":
    print("Running EDL loss sanity checks...")
    K, Q = 5, 15
    n_queries = K * Q
    targets = torch.arange(K).repeat_interleave(Q)

    alpha_confident = torch.ones(n_queries, K) + 0.01
    for i in range(n_queries):
        alpha_confident[i, targets[i]] += 99.0
    probs_confident = alpha_confident / alpha_confident.sum(dim=1, keepdim=True)

    loss, mse, kl, kl_w = edl_loss(
        probs_confident, alpha_confident, targets,
        episode=250,
        kl_annealing_start=0,
        kl_annealing_end=500,
        kl_max_weight=0.1
    )
    print(f"Confident prediction loss: {loss.item():.4f}, MSE: {mse.item():.4f}, KL: {kl.item():.4f}")

    assert compute_kl_weight(0, 0, 500, 0.1) == 0.0
    assert math.isclose(compute_kl_weight(250, 0, 500, 0.1), 0.05, rel_tol=1e-9)
    assert compute_kl_weight(500, 0, 500, 0.1) == 0.1

    print("EDL loss sanity checks PASSED.")