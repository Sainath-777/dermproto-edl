"""
temp_scaled_proto.py — Temperature-Scaled Prototypical Network for A2 Ablation
Learns a single scalar temperature parameter τ to scale negative distances before softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TempScaledProtoNet(nn.Module):
    """
    Learns a scalar temperature parameter tau (initialized at 1.0).
    Logits = -distances / tau
    """
    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temp], dtype=torch.float32))

    def forward(self, distances: torch.Tensor) -> dict:
        # Constrain temperature to be positive
        tau = F.softplus(self.temperature) + 1e-4
        logits = -distances / tau
        probs = F.softmax(logits, dim=-1)
        
        # Uncertainty derived from max softmax probability entropy or (1 - max_prob)
        max_probs, _ = torch.max(probs, dim=-1)
        uncertainty = 1.0 - max_probs
        
        return {
            "logits": logits,
            "probs": probs,
            "uncertainty": uncertainty,
            "temperature": tau.item()
        }