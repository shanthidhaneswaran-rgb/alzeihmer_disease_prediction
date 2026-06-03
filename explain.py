import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.preprocess import get_dataloaders
from model.lens_adnet import LENSADNet
from model.symbolic_layer import RULE_NAMES

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "outputs/lens_adnet_best.pt"
CLASS_NAMES = ["CN (Normal)", "MCI (Mild Impairment)", "AD (Alzheimer's)"]
os.makedirs("outputs", exist_ok=True)


def plot_explanation(patient_idx: int = 0):
    _, _, test_loader, n_clinical = get_dataloaders(batch_size=32)
    model = LENSADNet(n_clinical=n_clinical).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    mri_batch, clin_batch, label_batch = next(iter(test_loader))
    mri_s    = mri_batch[patient_idx:patient_idx+1].to(DEVICE)
    clin_s   = clin_batch[patient_idx:patient_idx+1].to(DEVICE)
    true_lbl = CLASS_NAMES[label_batch[patient_idx].item()]

    with torch.no_grad():
        logits, _, rule_weights, _ = model(mri_s, clin_s)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        rw    = rule_weights[0].cpu().numpy()

    pred_idx = probs.argmax()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LENS-ADNet — Explainability Output", fontsize=14, fontweight="bold")

    # ── Left: Diagnosis confidence ──
    ax = axes[0]
    colors = ["#3B8BD4" if i != pred_idx else "#E24B4A" for i in range(3)]
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="none", height=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Diagnosis Confidence", fontsize=12)
    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=10)
    pred_patch = mpatches.Patch(color="#E24B4A", label=f"Predicted: {CLASS_NAMES[pred_idx]}")
    ax.legend(handles=[pred_patch], loc="lower right", fontsize=9)
    ax.text(0.02, -0.12, f"True label: {true_lbl}", transform=ax.transAxes,
            fontsize=9, color="gray")
    ax.spines[["top","right"]].set_visible(False)

    # ── Right: Symbolic rule weights ──
    ax2 = axes[1]
    sorted_idx = np.argsort(rw)[::-1]
    rule_labels = [RULE_NAMES[i] for i in sorted_idx]
    rule_vals   = rw[sorted_idx]
    colors2 = ["#EF9F27" if v >= 0.1 else "#B4B2A9" for v in rule_vals]
    ax2.barh(rule_labels, rule_vals, color=colors2, edgecolor="none", height=0.55)
    ax2.set_xlim(0, max(rule_vals) * 1.25)
    ax2.set_xlabel("Rule weight", fontsize=11)
    ax2.set_title("Symbolic Reasoning — Active Rules", fontsize=12)
    for i, (v, lbl) in enumerate(zip(rule_vals, rule_labels)):
        ax2.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    highlight = mpatches.Patch(color="#EF9F27", label="Active rule (weight ≥ 0.10)")
    ax2.legend(handles=[highlight], loc="lower right", fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_path = f"outputs/explanation_patient{patient_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Explanation plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_explanation(patient_idx=0)