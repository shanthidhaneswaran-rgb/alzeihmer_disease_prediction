"""
fusion.py
Combines CNN features, Transformer context, and clinical data
into one unified embedding vector.
"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Concatenates three input streams and projects to a shared embedding.

    Inputs:
        cnn_feat:   (B, feat_dim)         — from LightweightCNN
        trans_feat: (B, feat_dim)         — from TinyTransformer
        clinical:   (B, n_clinical)       — normalised clinical scores

    Output:
        fused: (B, out_dim)
    """
    def __init__(self, feat_dim: int = 128, n_clinical: int = 3,
                 out_dim: int = 128):
        super().__init__()
        in_dim = feat_dim + feat_dim + n_clinical
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, cnn_feat, trans_feat, clinical):
        combined = torch.cat([cnn_feat, trans_feat, clinical], dim=1)
        return self.net(combined)