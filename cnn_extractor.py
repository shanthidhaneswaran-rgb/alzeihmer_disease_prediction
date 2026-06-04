"""
cnn_extractor.py  -  Smaller CNN for small dataset
Reduced from 4 stages to 3 stages.
Parameters: ~800K instead of 5.8M
This directly fights overfitting.
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
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
    Smaller 3D CNN — 3 stages instead of 4.
    Fewer parameters = less overfitting on small dataset.
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
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x
