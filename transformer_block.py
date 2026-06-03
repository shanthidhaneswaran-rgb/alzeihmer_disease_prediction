"""
transformer_block.py
Tiny Transformer that processes a sequence of patch tokens from the CNN feature map.
Input:  (B, seq_len, embed_dim)
Output: (B, embed_dim)  — pooled global context vector
"""

import torch
import torch.nn as nn
import math


class TinyTransformerBlock(nn.Module):
    """
    A single Transformer encoder layer.
    n_heads=2, feedforward dim = 4 × embed_dim.
    """
    def __init__(self, embed_dim: int = 128, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.ln1(x + self.drop(attn_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x, attn_weights   # return weights for XAI


class TinyTransformer(nn.Module):
    """
    Wraps 1–2 TinyTransformerBlocks.
    Takes a flat CNN feature vector, reshapes into tokens, attends, then pools back.
    """
    def __init__(self, feat_dim: int = 128, n_tokens: int = 8,
                 n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.n_tokens = n_tokens
        token_dim = feat_dim // n_tokens   # e.g. 128 // 8 = 16

        # Project flat CNN features → token sequence
        self.input_proj = nn.Linear(feat_dim, n_tokens * token_dim)
        self.token_dim = token_dim

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)

        self.layers = nn.ModuleList([
            TinyTransformerBlock(token_dim, n_heads) for _ in range(n_layers)
        ])
        self.pool = nn.Linear(token_dim, feat_dim)   # project back to feat_dim

    def forward(self, x):
        # x: (B, feat_dim)
        B = x.size(0)
        tokens = self.input_proj(x)                      # (B, n_tokens * token_dim)
        tokens = tokens.view(B, self.n_tokens, self.token_dim)  # (B, n_tokens, token_dim)
        tokens = tokens + self.pos_emb

        last_weights = None
        for layer in self.layers:
            tokens, last_weights = layer(tokens)

        context = tokens.mean(dim=1)       # mean-pool over tokens → (B, token_dim)
        context = self.pool(context)       # (B, feat_dim)
        return context, last_weights       # also return attention weights