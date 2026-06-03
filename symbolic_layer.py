"""
symbolic_layer.py  -  Gated Symbolic Reasoning Layer
KEY NOVELTY of LENS-ADNet.

Change from original:
  OLD: softmax(rule_scores) -> prototypes
       This forced rules to compete — only one could be dominant
  NEW: sigmoid(rule_scores) -> each rule fires independently
       Gated fusion: learned alpha controls residual vs symbolic
       This preserves information flow AND keeps explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RULE_NAMES = [
    "Hippocampus shrinkage",
    "Temporal lobe atrophy",
    "Ventricular enlargement",
    "Low MMSE score",
    "APOE4 risk indicator",
    "White matter lesions",
    "Frontal cortex thinning",
    "Memory recall deficit",
]


class SymbolicReasoningLayer(nn.Module):
    """
    Gated Neurosymbolic Layer.

    Architecture:
      1. rule_scores = sigmoid(W * x)       each rule in [0,1] independently
      2. rule_weights = rule_scores / sum    normalised for explanation display
      3. symbolic_out = rule_scores @ prototypes
      4. gate = sigmoid(W_gate * x)         learned blend ratio
      5. output = gate * symbolic_out + (1-gate) * x   adaptive residual
      6. LayerNorm for stable training
    """

    def __init__(self, embed_dim: int = 128, n_rules: int = 8):
        super().__init__()
        assert n_rules == len(RULE_NAMES)

        self.n_rules   = n_rules
        self.embed_dim = embed_dim

        # Rule scorer — sigmoid so rules are independent
        self.rule_scorer = nn.Linear(embed_dim, n_rules)

        # Learnable prototype per rule
        self.rule_prototypes = nn.Parameter(
            torch.randn(n_rules, embed_dim) * 0.02
        )

        # Gate: controls how much symbolic vs residual to use
        self.gate_fc = nn.Linear(embed_dim, embed_dim)

        # Projection after symbolic combination
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, embed_dim)

        # Step 1: independent rule activations (sigmoid not softmax)
        rule_scores = torch.sigmoid(self.rule_scorer(x))    # (B, n_rules) in [0,1]

        # Step 2: normalise for display/explainability
        # (divide by sum so they sum to 1 for the explanation output)
        rule_weights = rule_scores / (rule_scores.sum(dim=1, keepdim=True) + 1e-8)

        # Step 3: weighted combination of rule prototypes
        symbolic_out = torch.mm(rule_scores, self.rule_prototypes)  # (B, embed_dim)
        symbolic_out = self.proj(symbolic_out)

        # Step 4: adaptive gate
        gate = torch.sigmoid(self.gate_fc(x))               # (B, embed_dim) in [0,1]

        # Step 5: gated fusion with residual
        out = gate * symbolic_out + (1.0 - gate) * x        # (B, embed_dim)

        # Step 6: LayerNorm
        out = self.ln(out)

        return out, rule_scores, rule_weights

    @staticmethod
    def explain(rule_weights_single: torch.Tensor, threshold: float = 0.05):
        """
        Given a single sample's rule_weights (shape: n_rules,),
        returns list of (rule_name, weight) for active rules.
        """
        active = []
        for i, w in enumerate(rule_weights_single.detach().cpu()):
            if float(w) >= threshold:
                active.append((RULE_NAMES[i], round(float(w), 3)))
        active.sort(key=lambda t: t[1], reverse=True)
        return active