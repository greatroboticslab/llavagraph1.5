"""
Combine old FFT data (fft_augmented_nm/) and new 2.20.2026 FFT data,
then split 80/20 train/test, stratified by frequency within each category.

Sources:
  OLD: MSEC/data/FFT/fft_augmented_nm/{noise,sine,square,pulse,ramp}/
  NEW: MSEC/data/2.20.2026/fft/{noise,sine,Square,pulse,ramp}/

Notes:
  - Old pulse/ramp have no Hz label (10Hz input signal only)
  - New data has 1Hz–400Hz for all categories
  - Square folder in new data uses capital S → normalized to 'square'
  - 500Hz excluded (indistinguishable at Nyquist)
  - All segments from the same source CSV stay in the same split

Usage:
    python split_combined.py
"""

import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
OLD_DATA_DIR = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/FFT/fft_augmented_nm")
NEW_DATA_DIR = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/2.20.2026/fft")
OUTPUT_DIR   = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/training_VIT/split_dataset_combined")

TRAIN_RATIO  = 0.8
RANDOM_SEED  = 42
CATEGORIES   = ["noise", "sine", "square", "pulse", "ramp"]

# Old data: folder names match CATEGORIES exactly (lowercase)
# New data: same, except square is stored as 'Square' (capital S)
NEW_FOLDER_MAP = {
    "noise":  "noise",
    "sine":   "sine",
    "square": "Square",  # capital S in new data
    "pulse":  "pulse",
    "ramp":   "ramp",
}
# ──────────────────────────────────────────────────────────────────────────────


def extract_csv_id(filename: str) -> str:
    """
    Group all segments from the same recording together.
    Strips '_seg<n>_FFT' or '_seg<n>_TIME' suffix.
    Works for both old (_seg1_FFT) and new (-seg1_FFT) naming.

    Old: sine_100Hz_100Hz_10_absolute_seg1_FFT  -> sine_100Hz_100Hz_10_absolute
    New: sine-100Hz-1_seg1_FFT                 -> sine-100Hz-1
    """
    stem = Path(filename).stem
    return re.sub(r'_seg\d+_(TIME|FFT)$', '', stem, flags=re.IGNORECASE)


def extract_freq_group(csv_id: str) -> str:
    """
    Extract frequency label for stratified sampling.
    Handles both old (_100Hz_) and new (-100Hz-) naming conventions.

    Returns e.g. '100hz', '1hz', 'unknown'.
    """
    match = re.search(r'[-_](\d+)h[zx]', csv_id, re.IGNORECASE)
    if match:
        return f"{match.group(1)}hz"
    return "unknown"


def collect_category(category: str):
    """
    Gather all PNG paths for a category from both old and new sources.
    Returns list of (path, csv_id) tuples.
    """
    files = []

    # ── Old data ──────────────────────────────────────────────────────────────
    old_folder = OLD_DATA_DIR / category
    if old_folder.exists():
        for f in sorted(old_folder.glob("*.png")):
            # Exclude 500Hz
            if re.search(r'[-_]500h[zx]', f.name, re.IGNORECASE):
                continue
            files.append((f, extract_csv_id(f.name)))
        print(f"  Old {category}: {sum(1 for f in old_folder.glob('*.png'))} images "
              f"({len([x for x in files if str(x[0]).startswith(str(old_folder))])} after filtering)")
    else:
        print(f"  Old {category}: folder not found ({old_folder})")

    # ── New data ──────────────────────────────────────────────────────────────
    new_folder_name = NEW_FOLDER_MAP[category]
    new_folder = NEW_DATA_DIR / new_folder_name
    old_count = len(files)
    if new_folder.exists():
        for f in sorted(new_folder.glob("*.png")):
            if re.search(r'[-_]500h[zx]', f.name, re.IGNORECASE):
                continue
            files.append((f, extract_csv_id(f.name)))
        print(f"  New {category}: {len(files) - old_count} images (from '{new_folder_name}/')")
    else:
        print(f"  New {category}: folder not found ({new_folder})")

    return files


def split_category(category: str):
    """
    Collect + stratified 80/20 split for one category.
    Groups by CSV source, stratifies by frequency.
    """
    all_files = collect_category(category)  # [(path, csv_id), ...]

    # Group files by CSV source
    csv_groups: dict[str, list] = defaultdict(list)
    for path, csv_id in all_files:
        csv_groups[csv_id].append(path)

    # Group CSV sources by frequency stratum
    freq_to_csvs: dict[str, list] = defaultdict(list)
    for csv_id in csv_groups:
        freq = extract_freq_group(csv_id)
        freq_to_csvs[freq].append(csv_id)

    train_files, test_files = [], []
    random.seed(RANDOM_SEED)

    for freq, csv_ids in sorted(freq_to_csvs.items()):
        random.shuffle(csv_ids)
        n_train = max(1, round(len(csv_ids) * TRAIN_RATIO))
        if len(csv_ids) == 1:
            n_train = 1

        train_csvs = csv_ids[:n_train]
        test_csvs  = csv_ids[n_train:]

        for csv_id in train_csvs:
            train_files.extend(csv_groups[csv_id])
        for csv_id in test_csvs:
            test_files.extend(csv_groups[csv_id])

        print(f"    [{freq}] {len(csv_ids)} sources → {len(train_csvs)} train, {len(test_csvs)} test")

    return train_files, test_files


def main():
    # Clean up old output
    for split in ["train", "test"]:
        d = OUTPUT_DIR / split
        if d.exists():
            print(f"Removing old directory: {d}")
            shutil.rmtree(d)

    total_train = total_test = 0

    for cat in CATEGORIES:
        print(f"\n{'='*50}")
        print(f"Category: {cat}")
        train_files, test_files = split_category(cat)

        # Copy files
        for f in train_files:
            dest = OUTPUT_DIR / "train" / cat / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        for f in test_files:
            dest = OUTPUT_DIR / "test" / cat / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        print(f"  → {len(train_files)} train, {len(test_files)} test")
        total_train += len(train_files)
        total_test  += len(test_files)

    print(f"\n{'='*50}")
    print(f"Done! Total: {total_train} train, {total_test} test")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
