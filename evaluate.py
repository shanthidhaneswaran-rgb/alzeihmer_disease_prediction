import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score)
from data.preprocess import get_dataloaders
from model.lens_adnet import LENSADNet

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "outputs/lens_adnet_best.pt"
CLASS_NAMES = ["CN", "MCI", "AD"]
os.makedirs("outputs", exist_ok=True)


def evaluate():
    _, _, test_loader, n_clinical = get_dataloaders(batch_size=16)

    model = LENSADNet(n_clinical=n_clinical).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for mri, clinical, labels in test_loader:
            mri, clinical = mri.to(DEVICE), clinical.to(DEVICE)
            logits, _, _, _ = model(mri, clinical)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Print metrics ──
    print("\n" + "="*50)
    print("  LENS-ADNet Evaluation Results")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"  AUC-ROC (macro OvR): {auc:.4f}")

    # ── Confusion matrix plot ──
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("LENS-ADNet — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("\n  Confusion matrix saved → outputs/confusion_matrix.png")

    # ── Sample explanation for first test patient ──
    print("\n" + "="*50)
    print("  Sample Symbolic Explanation (Patient #1)")
    print("="*50)
    mri_sample, clin_sample, label_sample = next(iter(test_loader))
    result = model.predict_with_explanation(
        mri_sample[0:1].to(DEVICE), clin_sample[0:1].to(DEVICE)
    )
    true_label = CLASS_NAMES[label_sample[0].item()]
    print(f"  True label  : {true_label}")
    print(f"  Prediction  : {result['diagnosis']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Probabilities: CN={result['probabilities']['CN']} | "
          f"MCI={result['probabilities']['MCI']} | "
          f"AD={result['probabilities']['AD']}")
    print("\n  Active symbolic rules:")
    for rule_name, weight in result["rules"]:
        bar = "#" * int(weight * 40)
        print(f"    [{bar:<40}] {weight:.3f}  {rule_name}")


if __name__ == "__main__":
    evaluate()