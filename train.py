import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from data.preprocess import get_dataloaders
from model.lens_adnet import LENSADNet

# ===============================================================
#  CONFIGURATION
# ===============================================================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 150
STAGE1_END  = 30
BATCH_SIZE  = 8
LR_HEAD     = 5e-4
LR_FULL     = 5e-5
WARMUP      = 5
FEAT_DIM    = 128
N_RULES     = 8
N_CLASSES   = 3
PATIENCE    = 30
TTA_STEPS   = 5

SAVE_DIR    = os.path.join(ROOT, "outputs")
SAVE_PATH   = os.path.join(SAVE_DIR, "lens_adnet_best.pt")
PLOT_PATH   = os.path.join(SAVE_DIR, "training_curves.png")
LOG_PATH    = os.path.join(SAVE_DIR, "training_log.txt")
CLASS_NAMES = ["CN", "MCI", "AD"]

os.makedirs(SAVE_DIR, exist_ok=True)
# ===============================================================


class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard examples.
    Specifically helps with MCI which is the hardest class.
    gamma=2 means easy examples get downweighted strongly.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels,
                                   weight=self.weight,
                                   reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


def log(msg, log_file):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def predict_with_tta(model, mri, clinical, device, n_steps=TTA_STEPS):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(n_steps):
            aug = mri.clone()
            if i > 0:
                if torch.rand(1) > 0.5:
                    aug = torch.flip(aug, dims=[2])
                shift = torch.FloatTensor(1).uniform_(-0.03, 0.03).item()
                aug   = torch.clamp(aug + shift, 0.0, 1.0)
            logits, _, _, _ = model(aug.to(device), clinical.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(all_probs).mean(0)


def save_plot(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LENS-ADNet — Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"],
            label="Train", color="#378ADD", lw=2)
    ax.plot(epochs, history["val_loss"],
            label="Val",   color="#E24B4A", lw=2)
    if STAGE1_END < len(history["train_loss"]):
        ax.axvline(STAGE1_END, color="gray", linestyle=":",
                   lw=1.5, label=f"Stage2 ep{STAGE1_END}")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history["train_acc"]],
            label="Train", color="#378ADD", lw=2)
    ax.plot(epochs, [a*100 for a in history["val_acc"]],
            label="Val",   color="#1D9E75", lw=2)
    if history["val_acc"]:
        be = int(np.argmax(history["val_acc"])) + 1
        ba = max(history["val_acc"]) * 100
        ax.axvline(be, color="#EF9F27", linestyle="--", lw=1.5,
                   label=f"Best ep{be} ({ba:.1f}%)")
    ax.axhline(75, color="red", linestyle=":", lw=1, label="75% target")
    ax.set_title("Accuracy (%)"); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.plot(epochs, history["val_f1"],
            color="#D85A30", lw=2, label="Val F1")
    ax.axhline(0.65, color="red", linestyle=":", lw=1, label="0.65 target")
    ax.set_title("Val F1 (macro)"); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {PLOT_PATH}")


def freeze_backbone(model):
    for p in model.cnn.parameters():
        p.requires_grad = False
    for p in model.transformer.parameters():
        p.requires_grad = False
    log("  Backbone FROZEN (stage 1)", LOG_PATH)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
    log("  All layers UNFROZEN (stage 2)", LOG_PATH)


def run_validation(model, val_loader, criterion, use_tta=False):
    model.eval()
    vloss, vcorr, vtotal = 0.0, 0, 0
    vp, vl = [], []
    with torch.no_grad():
        for mri, clinical, labels in val_loader:
            mri      = mri.to(DEVICE)
            clinical = clinical.to(DEVICE)
            labels   = labels.to(DEVICE)

            if use_tta:
                probs = predict_with_tta(model, mri, clinical, DEVICE)
                preds = probs.argmax(dim=1).to(DEVICE)
            else:
                logits, _, _, _ = model(mri, clinical)
                preds = logits.argmax(dim=1)

            logits2, _, _, _ = model(mri, clinical)
            vloss  += criterion(logits2, labels).item()
            vcorr  += (preds == labels).sum().item()
            vtotal += labels.size(0)
            vp.extend(preds.cpu().numpy())
            vl.extend(labels.cpu().numpy())

    return (vcorr / vtotal,
            f1_score(vl, vp, average="macro", zero_division=0),
            vloss / len(val_loader))


def train():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("LENS-ADNet Training Log\n")
        f.write("=" * 60 + "\n")

    log("", LOG_PATH)
    log("  LENS-ADNet: Lightweight Explainable Neural-Symbolic Network", LOG_PATH)
    log("  Key changes: Focal Loss + Smaller model + Lower LR", LOG_PATH)
    log("  Realistic target: 72-80% accuracy", LOG_PATH)
    log("=" * 60, LOG_PATH)
    log(f"  Device      : {DEVICE}", LOG_PATH)
    log(f"  Epochs      : {EPOCHS}", LOG_PATH)
    log(f"  Batch size  : {BATCH_SIZE}", LOG_PATH)
    log(f"  LR stage 1  : {LR_HEAD}", LOG_PATH)
    log(f"  LR stage 2  : {LR_FULL}", LOG_PATH)
    log(f"  Loss        : Focal Loss (gamma=2, fixes MCI bias)", LOG_PATH)
    log(f"  Split       : Patient-level (no data leakage)", LOG_PATH)
    log("", LOG_PATH)

    train_loader, val_loader, test_loader, n_clinical = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = LENSADNet(
        n_clinical=n_clinical,
        feat_dim=FEAT_DIM,
        n_classes=N_CLASSES,
        n_rules=N_RULES
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    log(f"  Total parameters : {total_p:,}", LOG_PATH)
    log(f"  (Reduced from 6.1M to fix overfitting)", LOG_PATH)

    # Class weights
    class_counts = torch.zeros(N_CLASSES)
    for _, _, labels in train_loader:
        for lbl in labels:
            class_counts[lbl.item()] += 1
    class_weights = (class_counts.sum() /
                     (N_CLASSES * class_counts)).to(DEVICE)

    log("", LOG_PATH)
    log("  Class distribution:", LOG_PATH)
    for i, n in enumerate(CLASS_NAMES):
        log(f"    {n}   : {int(class_counts[i])} scans  "
            f"(weight = {class_weights[i]:.3f})", LOG_PATH)

    # Focal loss — fixes the MCI underprediction problem
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    log("  Loss function: Focal Loss (gamma=2.0)", LOG_PATH)

    history    = {"train_loss": [], "train_acc": [],
                  "val_loss":   [], "val_acc":   [], "val_f1": []}
    best_val_acc = 0.0
    best_val_f1  = 0.0
    best_epoch   = 0

    # ============================================================
    #  STAGE 1 — backbone frozen
    # ============================================================
    log("\n" + "=" * 60, LOG_PATH)
    log("  STAGE 1 — Backbone frozen, training head only", LOG_PATH)
    log("=" * 60, LOG_PATH)

    freeze_backbone(model)
    opt1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-3
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=STAGE1_END, eta_min=1e-5
    )

    for epoch in range(1, STAGE1_END + 1):
        model.train()
        tloss, tcorr, ttotal = 0.0, 0, 0

        for mri, clinical, labels in tqdm(
                train_loader,
                desc=f"  [S1] Epoch {epoch:02d}/{STAGE1_END}",
                leave=False, ncols=72):
            mri, clinical, labels = (mri.to(DEVICE),
                                     clinical.to(DEVICE),
                                     labels.to(DEVICE))
            opt1.zero_grad()
            logits, _, rw, _ = model(mri, clinical)
            loss  = criterion(logits, labels)
            loss += 0.005 * (-(rw * torch.log(rw + 1e-8)).sum(1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()

            tloss  += loss.item()
            tcorr  += (logits.argmax(1) == labels).sum().item()
            ttotal += labels.size(0)

        sch1.step()
        train_acc              = tcorr / ttotal
        val_acc, val_f1, vloss = run_validation(
            model, val_loader, criterion, use_tta=False)

        history["train_loss"].append(tloss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(vloss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        log(f"  [S1] Epoch {epoch:02d}/{STAGE1_END} | "
            f"Train Acc: {train_acc*100:.1f}% | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Val F1: {val_f1:.3f}", LOG_PATH)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            best_epoch   = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            log(f"    >> Best model saved "
                f"(Val Acc: {val_acc*100:.2f}%  F1: {val_f1:.3f})",
                LOG_PATH)

    # ============================================================
    #  STAGE 2 — full fine-tuning
    # ============================================================
    log("\n" + "=" * 60, LOG_PATH)
    log("  STAGE 2 — Full fine-tuning", LOG_PATH)
    log("=" * 60, LOG_PATH)

    unfreeze_all(model)
    opt2      = AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-3)
    remaining = EPOCHS - STAGE1_END

    def lr_lambda(ep):
        if ep < WARMUP:
            return float(ep + 1) / WARMUP
        progress = (ep - WARMUP) / max(remaining - WARMUP, 1)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sch2       = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda)
    no_improve = 0

    for epoch in range(STAGE1_END + 1, EPOCHS + 1):
        model.train()
        tloss, tcorr, ttotal = 0.0, 0, 0

        for mri, clinical, labels in tqdm(
                train_loader,
                desc=f"  [S2] Epoch {epoch:03d}/{EPOCHS}",
                leave=False, ncols=72):
            mri, clinical, labels = (mri.to(DEVICE),
                                     clinical.to(DEVICE),
                                     labels.to(DEVICE))
            opt2.zero_grad()
            logits, _, rw, _ = model(mri, clinical)
            loss  = criterion(logits, labels)
            loss += 0.005 * (-(rw * torch.log(rw + 1e-8)).sum(1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

            tloss  += loss.item()
            tcorr  += (logits.argmax(1) == labels).sum().item()
            ttotal += labels.size(0)

        sch2.step()
        train_acc              = tcorr / ttotal
        val_acc, val_f1, vloss = run_validation(
            model, val_loader, criterion, use_tta=True)

        history["train_loss"].append(tloss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(vloss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        log(f"  [S2] Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Acc: {train_acc*100:.1f}% | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Val F1: {val_f1:.3f}", LOG_PATH)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            best_epoch   = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            log(f"    >> Best model saved "
                f"(Val Acc: {val_acc*100:.2f}%  F1: {val_f1:.3f})",
                LOG_PATH)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            log(f"\n  Early stopping at epoch {epoch}", LOG_PATH)
            break

        if epoch % 10 == 0:
            save_plot(history)

    # ============================================================
    #  FINAL TEST EVALUATION
    # ============================================================
    log("\n" + "=" * 60, LOG_PATH)
    log("  FINAL TEST EVALUATION (with TTA)", LOG_PATH)
    log("=" * 60, LOG_PATH)

    model.load_state_dict(
        torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    tp_all, tl_all = [], []
    for mri, clinical, labels in test_loader:
        probs = predict_with_tta(model, mri, clinical, DEVICE)
        tp_all.append(probs)
        tl_all.extend(labels.numpy())

    tp     = torch.cat(tp_all).argmax(1).numpy()
    tl     = np.array(tl_all)
    t_acc  = np.mean(tp == tl)
    t_f1   = f1_score(tl, tp, average="macro", zero_division=0)
    report = classification_report(tl, tp,
                                    target_names=CLASS_NAMES,
                                    zero_division=0)

    log(f"\n  Test Accuracy : {t_acc*100:.2f}%", LOG_PATH)
    log(f"  Test F1       : {t_f1:.3f}",         LOG_PATH)
    log(f"\n  Per-class report:\n{report}",       LOG_PATH)
    log("=" * 60,                                 LOG_PATH)
    log("  TRAINING COMPLETE",                    LOG_PATH)
    log("=" * 60,                                 LOG_PATH)
    log(f"  Best Val Acc  : {best_val_acc*100:.2f}%", LOG_PATH)
    log(f"  Best Val F1   : {best_val_f1:.3f}",       LOG_PATH)
    log(f"  Best Epoch    : {best_epoch}",             LOG_PATH)
    log(f"  Test Accuracy : {t_acc*100:.2f}%",         LOG_PATH)
    log(f"  Test F1       : {t_f1:.3f}",               LOG_PATH)
    log(f"  Model saved   : {SAVE_PATH}",              LOG_PATH)

    save_plot(history)
    return model, history, test_loader, n_clinical


if __name__ == "__main__":
    train()
