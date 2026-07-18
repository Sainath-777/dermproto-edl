"""
backbone.py — DINOv2 ViT-S/14 wrapper for DermProto-EDL.

Loads Meta's DINOv2 ViT-Small/14 from torch.hub.
Output: 384-dimensional L2-normalized embeddings.
Backbone weights are FROZEN (requires_grad=False for all parameters).
Only the episodic head (Phase 4 MLP) will be trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Backbone(nn.Module):
    """
    Thin wrapper around DINOv2 ViT-S/14.

    Args:
        pretrained (bool): If True, load Meta's pretrained weights from torch.hub.
                           If False, random init (only for unit tests, never for training).
        freeze (bool): If True, freeze all DINOv2 parameters (default: True, per design spec).

    Output shape: (batch_size, 384)  — 384-dim CLS token embedding, L2-normalized.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        if pretrained:
            # Load DINOv2 ViT-S/14 from PyTorch Hub (Meta AI)
            # This downloads ~85MB of weights on first call; cached after that.
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True
            )
        else:
            # Random init for shape testing only. Never use for real training.
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=False
            )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[DINOv2Backbone] All backbone parameters FROZEN.")
        else:
            print("[DINOv2Backbone] WARNING: Backbone parameters are NOT frozen. This increases GPU memory significantly.")

    @property
    def embedding_dim(self) -> int:
        """Returns the output embedding dimension (384 for ViT-S/14)."""
        return 384

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from image batch.

        Args:
            x: Image tensor of shape (B, 3, 224, 224). Values must be normalized
               with ImageNet mean/std (already handled by build_transforms in datasets.py).

        Returns:
            Tensor of shape (B, 384) — L2-normalized CLS token embeddings.
        """
        # DINOv2 forward() returns the CLS token embedding directly.
        embeddings = self.encoder(x)                     # shape: (B, 384)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize for stable cosine distances
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode(). Allows model(x) shorthand."""
        return self.encode(x)


if __name__ == "__main__":
    """
    Standalone shape verification. Run: python src/models/backbone.py
    Expected output: 'PASSED. Embedding shape: torch.Size([4, 384])'
    """
    print("Running DINOv2Backbone shape test (pretrained=False for speed)...")
    model = DINOv2Backbone(pretrained=False, freeze=True)
    dummy_input = torch.randn(4, 3, 224, 224)   # batch of 4 images

    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, 384), f"Shape error: expected (4, 384), got {out.shape}"
    norm = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norm, torch.ones(4), atol=1e-5), "L2 norm check FAILED — embeddings not normalized."
    print(f"PASSED. Embedding shape: {out.shape}. L2 norms: {norm}")