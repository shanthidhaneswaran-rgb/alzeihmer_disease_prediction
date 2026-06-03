"""
lens_adnet.py  -  LENS-ADNet Full Model
Fixes applied:
  - Dropout reduced: 0.5/0.5/0.3 -> 0.3/0.2/0.1
  - Gated symbolic layer (replaces softmax bottleneck)
  - BatchNorm before classifier
"""

import torch
import torch.nn as nn
from model.cnn_extractor     import LightweightCNN
from model.transformer_block import TinyTransformer
from model.fusion            import FusionLayer
from model.symbolic_layer    import SymbolicReasoningLayer


class LENSADNet(nn.Module):
    def __init__(self, n_clinical=1, feat_dim=128,
                 n_classes=3, n_rules=8):
        super().__init__()

        # Block 1 - CNN
        self.cnn = LightweightCNN(out_features=feat_dim)

        # Block 2 - Transformer
        self.transformer = TinyTransformer(
            feat_dim=feat_dim, n_tokens=8,
            n_heads=2, n_layers=2
        )

        # Block 3 - Fusion
        self.fusion = FusionLayer(
            feat_dim=feat_dim,
            n_clinical=n_clinical,
            out_dim=feat_dim
        )

        # Dropout after fusion — reduced from 0.5 to 0.3
        self.fusion_drop = nn.Dropout(p=0.3)

        # Block 4 - Gated Symbolic Reasoning
        self.symbolic = SymbolicReasoningLayer(
            embed_dim=feat_dim, n_rules=n_rules
        )

        # Block 5 - Classifier
        # Dropout reduced: 0.5/0.3 -> 0.2/0.1
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mri, clinical):
        cnn_feat           = self.cnn(mri)
        trans_feat, attn_w = self.transformer(cnn_feat)
        fused              = self.fusion(cnn_feat, trans_feat, clinical)
        fused              = self.fusion_drop(fused)
        sym_out, rs, rw    = self.symbolic(fused)
        logits             = self.classifier(sym_out)
        return logits, rs, rw, attn_w

    def predict_with_explanation(self, mri, clinical, threshold=0.05):
        self.eval()
        with torch.no_grad():
            logits, _, rw, _ = self.forward(mri, clinical)
            probs = torch.softmax(logits, dim=1)[0]
            pred  = torch.argmax(probs).item()
            names = {
                0: "CN (Normal)",
                1: "MCI (Mild Impairment)",
                2: "AD (Alzheimer's)"
            }
            rules = SymbolicReasoningLayer.explain(rw[0], threshold)
        return {
            "diagnosis":     names[pred],
            "confidence":    f"{probs[pred].item()*100:.1f}%",
            "probabilities": {
                "CN":  f"{probs[0].item()*100:.1f}%",
                "MCI": f"{probs[1].item()*100:.1f}%",
                "AD":  f"{probs[2].item()*100:.1f}%",
            },
            "rules": rules,
        }