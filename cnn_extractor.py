"""
cnn_extractor.py
Uses a deeper 3D CNN with residual connections.
Pretrained weights optional via load_pretrained().
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3D Residual block — prevents vanishing gradients."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class LightweightCNN(nn.Module):
    """
    Deeper 3D CNN with residual connections.
    Input:  (B, 1, 64, 64, 64)
    Output: (B, out_features)
    """
    def __init__(self, out_features=128):
        super().__init__()

        self.encoder = nn.Sequential(
            # Stage 1: 64 -> 32
            nn.Conv3d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResBlock3D(32),

            # Stage 2: 32 -> 16
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResBlock3D(64),

            # Stage 3: 16 -> 8
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResBlock3D(128),

            # Stage 4: 8 -> 4
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            ResBlock3D(256),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x