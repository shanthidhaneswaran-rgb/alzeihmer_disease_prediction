"""
lens_adnet.py  -  LENS-ADNet
Changes:
  - Smaller CNN (3 stages, ~800K params total)
  - Dropout reduced further: 0.2/0.1
  - Simpler classifier head
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

        self.cnn = LightweightCNN(out_features=feat_dim)

        self.transformer = TinyTransformer(
            feat_dim=feat_dim, n_tokens=8,
            n_heads=2, n_layers=1   # reduced to 1 layer
        )

        self.fusion = FusionLayer(
            feat_dim=feat_dim,
            n_clinical=n_clinical,
            out_dim=feat_dim
        )

        self.fusion_drop = nn.Dropout(p=0.2)

        self.symbolic = SymbolicReasoningLayer(
            embed_dim=feat_dim, n_rules=n_rules
        )

        # Simple classifier — fewer parameters
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
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
