import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import rotate, zoom

LABEL_MAP    = {"CN": 0, "MCI": 1, "AD": 2}
CLASS_NAMES  = ["CN", "MCI", "AD"]
RAW_DIR      = "F:/Program/LENS_ADNet/data/raw"
CLINICAL_CSV = "F:/Program/LENS_ADNet/data/clinical_data.csv"
SCAN_CSV     = "F:/Program/LENS_ADNet/data/ADNI1_Complete_3Yr_1.5T_3_28_2026.csv"
TARGET_SHAPE = (64, 64, 64)


def collect_nifti_paths(raw_dir=RAW_DIR):
    records = []
    for group in ["CN", "MCI", "AD"]:
        group_dir = os.path.join(raw_dir, group)
        if not os.path.exists(group_dir):
            print(f"  WARNING: folder not found -> {group_dir}")
            continue
        for root, dirs, files in os.walk(group_dir):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    fpath   = os.path.join(root, f)
                    parts   = root.replace("\\", "/").split("/")
                    subj_id = None
                    for part in parts:
                        segs = part.split("_")
                        if (len(segs) >= 3 and segs[0].isdigit()
                                and segs[1].upper() == "S"
                                and segs[2].isdigit()):
                            subj_id = part
                            break
                    subj_id = subj_id or f
                    records.append((fpath, LABEL_MAP[group], subj_id))

    counts = {"CN": 0, "MCI": 0, "AD": 0}
    for _, lbl, _ in records:
        counts[CLASS_NAMES[lbl]] += 1
    print(f"\n  NIfTI files found:")
    print(f"    CN    : {counts['CN']}")
    print(f"    MCI   : {counts['MCI']}")
    print(f"    AD    : {counts['AD']}")
    print(f"    TOTAL : {len(records)}")
    return records


def resize_volume(data, target_shape=TARGET_SHAPE):
    """Centre-crop or zero-pad each dimension to target_shape."""
    for i in range(3):
        if data.shape[i] >= target_shape[i]:
            start  = (data.shape[i] - target_shape[i]) // 2
            idx    = [slice(None)] * 3
            idx[i] = slice(start, start + target_shape[i])
            data   = data[tuple(idx)]
        else:
            pad    = [(0, 0)] * 3
            pad[i] = (0, target_shape[i] - data.shape[i])
            data   = np.pad(data, pad)
    return data


def load_nifti(path):
    """
    Load NIfTI, resize, apply robust normalisation.
    Uses percentile clipping to remove extreme outlier voxels
    before normalising — this is standard in medical imaging.
    """
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    data = resize_volume(data)

    # Robust normalisation: clip to 1st–99th percentile
    # then scale to [0, 1]
    p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
    if p99 > p1:
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1)
    else:
        data = np.zeros_like(data)

    return data.astype(np.float32)


def augment_volume(data):
    """
    Moderate 3D augmentation — not too aggressive for small dataset.
    """
    # 1. Random flip (brain is roughly symmetric L/R)
    if np.random.rand() > 0.5:
        data = np.flip(data, axis=0).copy()

    # 2. Small rotation only +-5 degrees
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-5, 5)
        axes  = (0, 1)
        data  = rotate(data, angle, axes=axes,
                       reshape=False, mode="nearest")
        data  = resize_volume(data)

    # 3. Small zoom only 0.95-1.05
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.95, 1.05)
        data   = zoom(data, factor, mode="nearest")
        data   = resize_volume(data)

    # 4. Mild intensity shift
    shift = np.random.uniform(-0.05, 0.05)
    data  = np.clip(data + shift, 0.0, 1.0)

    # 5. Very small gaussian noise
    noise = np.random.normal(0, 0.005, data.shape).astype(np.float32)
    data  = np.clip(data + noise, 0.0, 1.0)

    return data.astype(np.float32)


def load_clinical(clinical_csv=CLINICAL_CSV, scan_csv=SCAN_CSV):
    """
    Load clinical features and normalise them.
    Returns dict and number of features.
    """
    if not os.path.exists(clinical_csv):
        print(f"  WARNING: clinical_data.csv not found.")
        return {}, 1

    df = pd.read_csv(clinical_csv)
    df["PTID"] = df["PTID"].astype(str).str.strip().str.replace("-", "_")

    # Try to merge Age from scan CSV
    n_features = 1
    if os.path.exists(scan_csv):
        try:
            scan_df = pd.read_csv(scan_csv)
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
                n_features = 2
                print(f"  Age merged successfully -> 2 clinical features")
        except Exception as e:
            print(f"  Age merge skipped: {e}")

    feat_cols = ["TOTAL13"] if n_features == 1 else ["TOTAL13", "Age"]
    df        = df.dropna(subset=feat_cols)
    scaler    = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    print(f"\n  Clinical data:")
    print(f"    Features  : {feat_cols}")
    print(f"    Subjects  : {df['PTID'].nunique()}")
    print(f"    DX dist   : {df['DX'].value_counts().to_dict()}")

    clinical_dict = {}
    for _, row in df.iterrows():
        subj = str(row["PTID"]).strip()
        clinical_dict[subj] = [float(row[c]) for c in feat_cols]

    print(f"  Clinical dict: {len(clinical_dict)} entries")
    return clinical_dict, n_features


class ADNIDataset(Dataset):
    def __init__(self, records, clinical_dict, n_clinical, augment=False):
        self.records       = records
        self.clinical_dict = clinical_dict
        self.n_clinical    = n_clinical
        self.augment       = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fpath, label, subj_id = self.records[idx]

        data = load_nifti(fpath)
        if self.augment:
            data = augment_volume(data)

        mri = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        clin = (self.clinical_dict.get(subj_id)
                or self.clinical_dict.get(subj_id.replace("_", "-"))
                or [0.0] * self.n_clinical)

        return (mri,
                torch.tensor(clin, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))


def get_dataloaders(batch_size=8, val_size=0.15, test_size=0.15):
    print("\n" + "=" * 50)
    print("  Loading ADNI Dataset")
    print("=" * 50)

    records                   = collect_nifti_paths()
    clinical_dict, n_clinical = load_clinical()

    if len(records) == 0:
        raise FileNotFoundError("No .nii/.nii.gz files found in data/raw/")

    matched = sum(1 for _, _, sid in records if sid in clinical_dict)
    print(f"\n  MRI matched to clinical : {matched}/{len(records)}")

    # ── Patient-level split ───────────────────────────────────────
    # Extract unique subject IDs as groups so the same patient
    # never appears in both train and test sets
    subject_ids = [r[2] for r in records]
    labels      = [r[1] for r in records]
    indices     = np.arange(len(records))

    # First split: train vs temp (val+test)
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size + test_size,
        random_state=42
    )
    train_idx, temp_idx = next(
        gss1.split(indices, labels, groups=subject_ids)
    )

    # Second split: val vs test from temp
    temp_subjects = [subject_ids[i] for i in temp_idx]
    temp_labels   = [labels[i]      for i in temp_idx]
    temp_indices  = np.arange(len(temp_idx))

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size / (val_size + test_size),
        random_state=42
    )
    rel_val_idx, rel_test_idx = next(
        gss2.split(temp_indices, temp_labels, groups=temp_subjects)
    )

    val_idx  = temp_idx[rel_val_idx]
    test_idx = temp_idx[rel_test_idx]

    print(f"\n  Patient-level split (no data leakage):")
    print(f"    Train : {len(train_idx)} scans  (augmentation ON)")
    print(f"    Val   : {len(val_idx)} scans  (augmentation OFF)")
    print(f"    Test  : {len(test_idx)} scans  (augmentation OFF)")

    # Verify no subject overlap between splits
    train_subjects = set(subject_ids[i] for i in train_idx)
    val_subjects   = set(subject_ids[i] for i in val_idx)
    test_subjects  = set(subject_ids[i] for i in test_idx)
    overlap_tv = train_subjects & val_subjects
    overlap_tt = train_subjects & test_subjects
    if overlap_tv or overlap_tt:
        print(f"  WARNING: Subject overlap detected! TV={len(overlap_tv)} TT={len(overlap_tt)}")
    else:
        print(f"  Patient-level integrity: VERIFIED (zero overlap)")

    train_ds = ADNIDataset([records[i] for i in train_idx],
                            clinical_dict, n_clinical, augment=True)
    val_ds   = ADNIDataset([records[i] for i in val_idx],
                            clinical_dict, n_clinical, augment=False)
    test_ds  = ADNIDataset([records[i] for i in test_idx],
                            clinical_dict, n_clinical, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=0)

    print(f"  Clinical features : {n_clinical}")
    print("=" * 50)

    return train_loader, val_loader, test_loader, n_clinical