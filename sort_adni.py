"""
sort_adni.py
════════════════════════════════════════════════════════════════════
Reads ADNI1_Complete_3Yr_1.5T_3_28_2026.csv,
matches every patient folder in the ADNI/ download folder
to its group (CN / MCI / AD),
and copies them into data/raw/CN/, data/raw/MCI/, data/raw/AD/

YOUR PATHS ARE ALREADY SET — just run it directly:
    cd F:/Program/LENS_ADNet
    python data/sort_adni.py
════════════════════════════════════════════════════════════════════
"""

import os
import shutil
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
#  YOUR EXACT PATHS — already configured for your machine
# ═══════════════════════════════════════════════════════════════════
DOWNLOAD_FOLDER = "F:/Program/LENS_ADNet/data/ADNI"
CSV_FILE        = "F:/Program/LENS_ADNet/data/ADNI1_Complete_3Yr_1.5T_3_28_2026.csv"
OUTPUT_DIR      = "F:/Program/LENS_ADNet/data/raw"
# ═══════════════════════════════════════════════════════════════════


def find_subject_id_from_path(path):
    """
    ADNI folder names always look like:  941_S_1202  or  137_S_1414
    Format: digits_S_digits
    This function checks every part of the path and returns the subject ID.
    """
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        segments = part.split("_")
        if (len(segments) >= 3 and
                segments[0].isdigit() and
                segments[1].upper() == "S" and
                segments[2].isdigit()):
            return part
    return None


def load_subject_group_map(csv_file):
    """
    Reads the LONI portal CSV file.
    Returns a dict:  { "941_S_1202": "CN",  "137_S_1414": "MCI", ... }

    Handles all possible column name variations from LONI portal.
    """
    print(f"\n  Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"  CSV shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns found: {list(df.columns)}")

    # ── Auto-detect Subject column ──────────────────────────────────
    subject_col = None
    for col in df.columns:
        if col.strip().lower() in ["subject", "subject id", "subjectid",
                                    "ptid", "subject_id", "id"]:
            subject_col = col
            break

    # ── Auto-detect Group column ────────────────────────────────────
    group_col = None
    for col in df.columns:
        if col.strip().lower() in ["group", "dx group", "dx_group",
                                    "diagnosis", "dx", "research group"]:
            group_col = col
            break

    # ── Fallback: print columns and guess ───────────────────────────
    if subject_col is None or group_col is None:
        print("\n  Could not auto-detect columns. Printing all columns:")
        for i, c in enumerate(df.columns):
            print(f"    [{i}] '{c}'  — sample values: {df[c].dropna().head(3).tolist()}")

        # Use first column as subject, look for group manually
        subject_col = df.columns[0]
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            upper_vals  = [str(v).strip().upper() for v in unique_vals]
            if any(v in ["CN", "MCI", "AD"] for v in upper_vals):
                group_col = col
                break

        if group_col is None:
            group_col = df.columns[3]  # fallback to 4th column
            print(f"\n  WARNING: Could not find group column.")
            print(f"  Using '{subject_col}' as Subject and '{group_col}' as Group.")
            print(f"  If sorting is wrong, check sort_adni.py and set them manually.")

    print(f"\n  Using  Subject column : '{subject_col}'")
    print(f"  Using  Group column   : '{group_col}'")

    # ── Build the mapping dict ──────────────────────────────────────
    mapping = {}
    for _, row in df.iterrows():
        # Normalise subject ID: replace dashes with underscores
        # LONI sometimes uses 941-S-1202 format in CSV
        subj = str(row[subject_col]).strip().replace("-", "_")
        grp  = str(row[group_col]).strip().upper()

        # Normalise group name to CN / MCI / AD
        if grp in ["CN", "CONTROL", "NORMAL", "NL", "COGNITIVELY NORMAL"]:
            grp = "CN"
        elif grp in ["MCI", "LMCI", "EMCI", "SMC",
                      "MILD COGNITIVE IMPAIRMENT", "EARLY MCI", "LATE MCI"]:
            grp = "MCI"
        elif grp in ["AD", "ALZHEIMER", "DEMENTED",
                      "ALZHEIMER'S DISEASE", "ALZHEIMERS DISEASE"]:
            grp = "AD"

        mapping[subj] = grp

    # Count per group
    counts = {"CN": 0, "MCI": 0, "AD": 0, "OTHER": 0}
    for g in mapping.values():
        if g in counts:
            counts[g] += 1
        else:
            counts["OTHER"] += 1

    print(f"\n  Subjects found in CSV:")
    print(f"    CN    : {counts['CN']}")
    print(f"    MCI   : {counts['MCI']}")
    print(f"    AD    : {counts['AD']}")
    if counts["OTHER"] > 0:
        print(f"    Other : {counts['OTHER']} (will be skipped)")

    return mapping


def sort_dataset():
    print("=" * 60)
    print("  ADNI Dataset Sorter — LENS-ADNet")
    print("=" * 60)

    # ── Validate paths ──────────────────────────────────────────────
    errors = False
    if not os.path.exists(DOWNLOAD_FOLDER):
        print(f"\n  ERROR: ADNI folder not found:")
        print(f"         {DOWNLOAD_FOLDER}")
        errors = True
    if not os.path.exists(CSV_FILE):
        print(f"\n  ERROR: CSV file not found:")
        print(f"         {CSV_FILE}")
        errors = True
    if errors:
        print("\n  Please fix the paths at the top of sort_adni.py and run again.")
        return

    print(f"\n  ADNI folder : {DOWNLOAD_FOLDER}")
    print(f"  CSV file    : {CSV_FILE}")
    print(f"  Output dir  : {OUTPUT_DIR}")

    # ── Create output folders ───────────────────────────────────────
    for group in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(OUTPUT_DIR, group), exist_ok=True)
    print(f"\n  Created output folders: raw/CN/  raw/MCI/  raw/AD/")

    # ── Load CSV mapping ────────────────────────────────────────────
    subj_map = load_subject_group_map(CSV_FILE)

    # ── Walk ADNI folder and sort ───────────────────────────────────
    print(f"\n  Scanning ADNI folder...")
    print(f"  (This may take a few minutes for 2182 subjects)\n")

    # Get all top-level subject folders directly inside ADNI/
    all_entries = os.listdir(DOWNLOAD_FOLDER)
    subject_folders = []
    for entry in all_entries:
        full_path = os.path.join(DOWNLOAD_FOLDER, entry)
        if os.path.isdir(full_path):
            subject_folders.append(entry)

    print(f"  Found {len(subject_folders)} folders inside ADNI/\n")

    moved      = 0
    skipped    = 0
    not_in_csv = 0
    already    = 0

    for folder_name in sorted(subject_folders):
        # Normalise folder name
        subj_id = folder_name.strip().replace("-", "_")

        # Look up group from CSV
        group = subj_map.get(subj_id)

        if group not in ("CN", "MCI", "AD"):
            # Try without normalisation
            group = subj_map.get(folder_name.strip())

        if group not in ("CN", "MCI", "AD"):
            print(f"  SKIP  {folder_name}  (not found in CSV or unknown group)")
            not_in_csv += 1
            continue

        src = os.path.join(DOWNLOAD_FOLDER, folder_name)
        dst = os.path.join(OUTPUT_DIR, group, folder_name)

        if os.path.exists(dst):
            print(f"  EXISTS  {folder_name}  →  {group}/")
            already += 1
            continue

        try:
            shutil.copytree(src, dst)
            print(f"  SORTED  {folder_name}  →  {group}/")
            moved += 1
        except Exception as e:
            print(f"  ERROR   {folder_name}  →  {e}")
            skipped += 1

    # ── Final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SORTING COMPLETE")
    print("=" * 60)
    print(f"  Sorted successfully : {moved}")
    print(f"  Already existed     : {already}")
    print(f"  Not found in CSV    : {not_in_csv}")
    print(f"  Errors              : {skipped}")

    print(f"\n  Final subject counts in data/raw/:")
    total = 0
    for group in ["CN", "MCI", "AD"]:
        folder = os.path.join(OUTPUT_DIR, group)
        count  = len([
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ])
        print(f"    {group}/   →   {count} subjects")
        total += count
    print(f"    TOTAL  →   {total} subjects")

    print("\n" + "=" * 60)
    print("  NEXT STEPS:")
    print("  1. python train/train.py")
    print("  2. python train/evaluate.py")
    print("  3. python explainability/explain.py")
    print("=" * 60)


if __name__ == "__main__":
    sort_dataset()