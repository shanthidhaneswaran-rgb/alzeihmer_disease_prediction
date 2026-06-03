import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
import pandas as pd

from model.lens_adnet import LENSADNet
from model.symbolic_layer import RULE_NAMES
from data.preprocess import load_nifti, collect_nifti_paths

# ===============================================================
#  CONFIG
# ===============================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH   = "F:/Program/LENS_ADNet/outputs/lens_adnet_best.pt"
CLINICAL_CSV = "F:/Program/LENS_ADNet/data/clinical_data.csv"
SCAN_CSV     = "F:/Program/LENS_ADNet/data/ADNI1_Complete_3Yr_1.5T_3_28_2026.csv"
OUTPUT_DIR   = "F:/Program/LENS_ADNet/outputs"
FEAT_DIM     = 128
N_RULES      = 8
N_CLASSES    = 3
CLASS_NAMES  = ["CN (Normal)", "MCI (Mild Impairment)", "AD (Alzheimer's)"]
CLASS_SHORT  = ["CN", "MCI", "AD"]
# ===============================================================


def detect_n_clinical():
    """
    Auto-detect number of clinical features from saved checkpoint.
    This prevents architecture mismatch errors.
    """
    if not os.path.exists(MODEL_PATH):
        return 1
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    # fusion layer input = feat_dim + feat_dim + n_clinical
    # fusion.net.0.weight shape = (256, feat_dim*2 + n_clinical)
    for key in ckpt:
        if "fusion.net.0.weight" in key:
            in_dim     = ckpt[key].shape[1]
            n_clinical = in_dim - FEAT_DIM * 2
            print(f"  Auto-detected n_clinical = {n_clinical} "
                  f"from saved checkpoint")
            return max(1, n_clinical)
    return 1


def load_model(n_clinical):
    """Load the trained LENS-ADNet model."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Model not found at {MODEL_PATH}")
        print(f"  Please run:  python train/train.py  first.")
        sys.exit(1)

    model = LENSADNet(
        n_clinical=n_clinical,
        feat_dim=FEAT_DIM,
        n_classes=N_CLASSES,
        n_rules=N_RULES
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "clinical_head.0.weight",
        "clinical_head.0.bias",
        "clinical_head.3.weight",
        "clinical_head.3.bias",
    }

    if unexpected:
        print(f"\n  WARNING: {len(unexpected)} unexpected keys in checkpoint.")
        print("  Checkpoint does not match current model definition.")
        print(f"  Unexpected keys: {unexpected}")
        print(f"  Please retrain:  python train/train.py")
        sys.exit(1)

    if missing:
        if set(missing).issubset(allowed_missing):
            # Older checkpoints were trained before the clinical head existed.
            # Zero init keeps its contribution neutral at inference time.
            for param in model.clinical_head.parameters():
                torch.nn.init.zeros_(param)
            print("\n  NOTE: Older checkpoint detected (missing clinical head).")
            print("  Initialised clinical head to zeros for compatibility.")
        else:
            print(f"\n  WARNING: {len(missing)} missing keys in checkpoint.")
            print(f"  Missing keys: {missing}")
            print(f"  This means the model architecture changed after training.")
            print(f"  Please retrain:  python train/train.py")
            sys.exit(1)

    print(f"  Model loaded successfully from: {MODEL_PATH}")

    model.eval()
    return model


def build_clinical_scaler():
    """
    Rebuild the StandardScaler from training data
    so new patient inputs are normalised correctly.
    """
    if not os.path.exists(CLINICAL_CSV):
        print(f"  WARNING: clinical_data.csv not found.")
        return None, ["TOTAL13"]

    df = pd.read_csv(CLINICAL_CSV)
    df["PTID"] = df["PTID"].str.strip().str.replace("-", "_")

    has_age = False
    if os.path.exists(SCAN_CSV):
        try:
            scan_df = pd.read_csv(SCAN_CSV)
            for col in scan_df.columns:
                if col.strip().lower() == "subject":
                    scan_df = scan_df.rename(columns={col: "PTID"})
                    break
            scan_df["PTID"] = (scan_df["PTID"].astype(str)
                                .str.strip().str.replace("-", "_"))
            scan_df = scan_df.groupby("PTID").first().reset_index()
            if "Age" in scan_df.columns:
                df = df.merge(scan_df[["PTID", "Age"]],
                              on="PTID", how="left")
                df["Age"] = df["Age"].fillna(df["Age"].median())
                has_age = True
        except Exception:
            pass

    feat_cols = ["TOTAL13", "Age"] if has_age else ["TOTAL13"]
    scaler    = StandardScaler()
    scaler.fit(df[feat_cols].dropna())
    return scaler, feat_cols


def normalise_clinical(total13, age=None,
                        scaler=None, feat_cols=None):
    """Normalise raw clinical values using the training scaler."""
    if scaler is None:
        return [0.0]
    if feat_cols and "Age" in feat_cols and age is not None:
        raw = np.array([[total13, age]], dtype=np.float32)
    else:
        raw = np.array([[total13]], dtype=np.float32)
    return scaler.transform(raw)[0].tolist()


def predict_tta(model, mri_tensor, clinical_tensor, n_steps=5):
    """
    Test-Time Augmentation: run scan through model n_steps times
    with slight augmentation, average the predictions.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(n_steps):
            aug = mri_tensor.clone()
            if i > 0:
                if torch.rand(1) > 0.5:
                    aug = torch.flip(aug, dims=[2])
                shift = torch.FloatTensor(1).uniform_(-0.05, 0.05).item()
                aug   = torch.clamp(aug + shift, 0.0, 1.0)

            logits, _, _, _ = model(
                aug.to(DEVICE), clinical_tensor.to(DEVICE)
            )
            all_probs.append(torch.softmax(logits, dim=1).cpu())

    avg_probs = torch.stack(all_probs).mean(0)[0]

    with torch.no_grad():
        _, _, rule_weights, attn = model(
            mri_tensor.to(DEVICE), clinical_tensor.to(DEVICE)
        )

    return avg_probs, rule_weights[0].cpu(), attn


def print_report(probs, rule_weights, patient_info):
    """Print a detailed prediction report to the console."""
    pred_idx  = probs.argmax().item()
    pred_name = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx].item() * 100

    print("\n" + "=" * 62)
    print("  LENS-ADNet  PREDICTION REPORT")
    print("=" * 62)

    print("\n  Patient Information:")
    for k, v in patient_info.items():
        print(f"    {k:<22}: {v}")

    print("\n" + "-" * 62)
    print(f"  DIAGNOSIS      : {pred_name}")
    print(f"  CONFIDENCE     : {conf:.1f}%")
    print("-" * 62)

    print("\n  Class Probabilities:")
    for i, name in enumerate(CLASS_SHORT):
        p      = probs[i].item() * 100
        bar    = "#" * int(p / 2)
        marker = "  <-- PREDICTED" if i == pred_idx else ""
        print(f"    {name}  [{bar:<50}] {p:5.1f}%{marker}")

    print("\n  Symbolic Reasoning  (why this prediction):")
    print("  " + "-" * 58)
    rw       = rule_weights.detach().numpy()
    sorted_i = np.argsort(rw)[::-1]
    for rank, idx in enumerate(sorted_i):
        w      = rw[idx]
        bar    = "#" * int(w * 80)
        active = "  [ACTIVE]" if w >= 0.12 else ""
        print(f"    {rank+1}. [{bar:<40}] {w:.3f}  "
              f"{RULE_NAMES[idx]}{active}")

    print("\n" + "-" * 62)
    print("  Clinical Interpretation:")
    if pred_idx == 0:
        print("  No significant signs of Alzheimer's detected.")
        print("  Brain structure appears within normal range.")
    elif pred_idx == 1:
        print("  Early cognitive impairment indicators detected.")
        print("  Follow-up scans and monitoring recommended.")
    else:
        print("  Significant Alzheimer's indicators detected.")
        print("  Key regions: hippocampus, temporal lobe,")
        print("  frontal cortex show atrophy patterns.")
    print("=" * 62)


def save_output_plot(probs, rule_weights, mri_data,
                     patient_info, save_path):
    """
    Save a 3-panel figure:
      Panel 1 - Diagnosis confidence
      Panel 2 - Symbolic rule weights
      Panel 3 - MRI brain slice
    """
    pred_idx  = probs.argmax().item()
    pred_name = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx].item() * 100

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")

    # -- Panel 1: Confidence --
    ax1    = fig.add_subplot(1, 3, 1)
    colors = ["#B5D4F4"] * 3
    colors[pred_idx] = "#E24B4A"

    bars = ax1.barh(CLASS_SHORT,
                    [p.item() * 100 for p in probs],
                    color=colors, edgecolor="none", height=0.5)
    ax1.set_xlim(0, 120)
    ax1.set_xlabel("Confidence (%)", fontsize=11)
    ax1.set_title("Diagnosis Confidence",
                  fontsize=12, fontweight="bold")

    for bar, p in zip(bars, probs):
        ax1.text(p.item() * 100 + 1.5,
                 bar.get_y() + bar.get_height() / 2,
                 f"{p.item()*100:.1f}%",
                 va="center", fontsize=11, fontweight="bold")

    true_lbl = patient_info.get("True label", "Unknown")
    result   = patient_info.get("Result", "")
    color_r  = "#3B6D11" if result == "CORRECT" else "#A32D2D"

    ax1.legend(
        handles=[mpatches.Patch(color="#E24B4A",
                 label=f"Prediction: {pred_name}\n"
                       f"Confidence: {conf:.1f}%")],
        loc="lower right", fontsize=9
    )

    info_str = (f"Subject: {patient_info.get('Subject ID','?')}\n"
                f"True label: {true_lbl}\n"
                f"ADAS-13: {patient_info.get('ADAS-13','?')}")
    ax1.text(0.02, -0.22, info_str,
             transform=ax1.transAxes,
             fontsize=9, color="gray")

    result_str = f"Result: {result}"
    ax1.text(0.02, -0.38, result_str,
             transform=ax1.transAxes,
             fontsize=10, fontweight="bold", color=color_r)

    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor("white")

    # -- Panel 2: Symbolic rules --
    ax2       = fig.add_subplot(1, 3, 2)
    rw_vals   = rule_weights.detach().numpy()
    sorted_i  = np.argsort(rw_vals)[::-1]
    rule_lbls = [RULE_NAMES[i] for i in sorted_i]
    rule_vals = rw_vals[sorted_i]
    bar_cols  = ["#EF9F27" if v >= 0.12 else "#D3D1C7"
                 for v in rule_vals]

    ax2.barh(rule_lbls, rule_vals,
             color=bar_cols, edgecolor="none", height=0.6)
    for i, v in enumerate(rule_vals):
        ax2.text(v + 0.001, i, f"{v:.3f}",
                 va="center", fontsize=9)

    ax2.set_xlabel("Rule weight", fontsize=11)
    ax2.set_title("Symbolic Reasoning  —  Active Rules",
                  fontsize=12, fontweight="bold")
    ax2.set_xlim(0, max(rule_vals) * 1.35 + 0.01)
    ax2.legend(
        handles=[mpatches.Patch(color="#EF9F27",
                 label="Active rule (weight >= 0.12)")],
        loc="lower right", fontsize=9
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("white")

    # -- Panel 3: MRI slice --
    ax3  = fig.add_subplot(1, 3, 3)
    mid  = mri_data[mri_data.shape[0] // 2, :, :]
    ax3.imshow(mid.T, cmap="gray", origin="lower")
    ax3.set_title(
        f"MRI Brain Slice  (mid-axial)\nPrediction: {pred_name}",
        fontsize=12, fontweight="bold"
    )
    ax3.axis("off")

    border = {"CN (Normal)":         "#1D9E75",
              "MCI (Mild Impairment)":"#EF9F27",
              "AD (Alzheimer's)":     "#E24B4A"}
    for spine in ax3.spines.values():
        spine.set_edgecolor(border.get(pred_name, "gray"))
        spine.set_linewidth(4)
        spine.set_visible(True)

    plt.suptitle(
        f"LENS-ADNet  |  {pred_name}  |  Confidence: {conf:.1f}%",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved -> {save_path}")


def run_demo_from_test_set(model, scaler, feat_cols):
    """
    Auto mode: pick one CN, one MCI, one AD patient
    from the real ADNI data and run prediction on each.
    """
    print("\n  Scanning ADNI data for sample patients ...")
    all_records = collect_nifti_paths()

    if len(all_records) == 0:
        print("  ERROR: No scans found in data/raw/")
        return

    # Load clinical CSV for ADAS-13 lookup
    clinical_df  = pd.read_csv(CLINICAL_CSV)
    clinical_df["PTID"] = (clinical_df["PTID"].astype(str)
                            .str.strip().str.replace("-", "_"))
    clin_lookup  = {row["PTID"]: row["TOTAL13"]
                    for _, row in clinical_df.iterrows()}

    label_names  = {0: "CN", 1: "MCI", 2: "AD"}
    samples      = {}

    for fpath, label, subj_id in all_records:
        grp = label_names[label]
        if grp not in samples:
            samples[grp] = (fpath, label, subj_id)
        if len(samples) == 3:
            break

    print(f"  Selected {len(samples)} patients (one per class)\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for grp, (fpath, true_label, subj_id) in samples.items():

        print(f"\n{'='*62}")
        print(f"  Patient: {subj_id}   True label: {grp}")
        print(f"{'='*62}")

        # Load MRI
        mri_data   = load_nifti(fpath)
        mri_tensor = (torch.tensor(mri_data, dtype=torch.float32)
                      .unsqueeze(0).unsqueeze(0))

        # Clinical score
        raw_t13     = clin_lookup.get(subj_id, 17.0)
        clin_normed = normalise_clinical(
            raw_t13, scaler=scaler, feat_cols=feat_cols
        )
        clin_tensor = torch.tensor([clin_normed], dtype=torch.float32)

        # Predict
        probs, rule_weights, _ = predict_tta(
            model, mri_tensor, clin_tensor
        )

        pred_idx = probs.argmax().item()
        result   = "CORRECT" if CLASS_SHORT[pred_idx] == grp else "WRONG"

        patient_info = {
            "Subject ID" : subj_id,
            "True label" : grp,
            "ADAS-13"    : f"{raw_t13:.2f}",
            "Result"     : result,
        }

        print_report(probs, rule_weights, patient_info)

        plot_path = os.path.join(
            OUTPUT_DIR, f"demo_{subj_id}_{grp}.png"
        )
        save_output_plot(
            probs, rule_weights, mri_data,
            patient_info, plot_path
        )

    print("\n" + "=" * 62)
    print("  Demo complete! Output images saved to outputs/")
    print("  Files created:")
    for grp, (_, _, subj_id) in samples.items():
        print(f"    outputs/demo_{subj_id}_{grp}.png")
    print("=" * 62)


def run_custom_patient(model, scaler, feat_cols):
    """
    Custom mode: user enters their own MRI file path
    and ADAS-13 score manually.
    """
    print("\n" + "=" * 62)
    print("  LENS-ADNet  Custom Patient Demo")
    print("=" * 62)
    print("\n  Enter patient details below.")
    print("  (Press Enter without typing to use auto demo instead)")

    mri_path = input("\n  MRI file path (.nii or .nii.gz): ").strip()

    if not mri_path:
        print("\n  No path given. Running auto demo on test set.")
        run_demo_from_test_set(model, scaler, feat_cols)
        return

    if not os.path.exists(mri_path):
        print(f"\n  ERROR: File not found:\n  {mri_path}")
        return

    try:
        total13 = float(
            input("  ADAS-13 score (0 to 85, higher = worse): ").strip()
        )
    except ValueError:
        total13 = 17.0
        print(f"  Invalid — using default ADAS-13 = {total13}")

    subject_id = input(
        "  Patient ID (optional, press Enter to skip): "
    ).strip()
    if not subject_id:
        subject_id = (os.path.basename(mri_path)
                      .replace(".nii.gz", "").replace(".nii", ""))

    print(f"\n  Loading MRI scan ...")
    mri_data   = load_nifti(mri_path)
    mri_tensor = (torch.tensor(mri_data, dtype=torch.float32)
                  .unsqueeze(0).unsqueeze(0))

    clin_normed = normalise_clinical(
        total13, scaler=scaler, feat_cols=feat_cols
    )
    clin_tensor = torch.tensor([clin_normed], dtype=torch.float32)

    print("  Running prediction ...")
    probs, rule_weights, _ = predict_tta(
        model, mri_tensor, clin_tensor
    )

    patient_info = {
        "Subject ID" : subject_id,
        "ADAS-13"    : f"{total13:.2f}",
    }

    print_report(probs, rule_weights, patient_info)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, f"demo_{subject_id}.png")
    save_output_plot(
        probs, rule_weights, mri_data,
        patient_info, plot_path
    )


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  LENS-ADNet  Patient Prediction Demo")
    print("  Dataset : ADNI1 Complete 3Yr 1.5T")
    print(f"  Device  : {DEVICE}")
    print("=" * 62)

    # Auto-detect clinical features from saved checkpoint
    n_clinical = detect_n_clinical()

    # Load model
    model = load_model(n_clinical)

    # Build clinical scaler
    scaler, feat_cols = build_clinical_scaler()

    print("\n  Choose mode:")
    print("  1. Auto  — runs on 3 real ADNI patients (CN, MCI, AD)")
    print("  2. Custom — enter your own MRI file path + ADAS-13 score")

    choice = input("\n  Enter 1 or 2: ").strip()

    if choice == "2":
        run_custom_patient(model, scaler, feat_cols)
    else:
        run_demo_from_test_set(model, scaler, feat_cols)
