"""
Split FFT augmented images into stratified train/test sets (80/20).

Splits by CSV source file so all 3 segments from the same CSV stay together.
Stratifies by frequency so each frequency is proportionally represented.
Excludes 500Hz sine/square (indistinguishable at Nyquist).

Usage:
    python split_data.py \
        --image-dir ../data/FFT/fft_augmented_nm \
        --output-dir .
"""

import re
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


TRAIN_RATIO = 0.8
EXCLUDE_FREQ = 500
RANDOM_SEED = 42


def extract_freq_group(filename):
    """Extract frequency from filename for stratification.
    Returns 'noise' for noise files, or frequency string like '100Hz'."""
    fname = filename.lower()
    if fname.startswith("noise"):
        return "noise"
    match = re.search(r'_(\d+)(?:hz|hx)', fname, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Hz"
    return "unknown"


def extract_csv_id(filename):
    """Extract the base CSV identity (without segment label).
    e.g. 'sine_100Hz_100Hz_10_absolute_seg1_FFT.png' -> 'sine_100Hz_100Hz_10_absolute'
    """
    # Remove _seg{N}_FFT.png suffix
    stem = Path(filename).stem
    return re.sub(r'_seg\d+_FFT$', '', stem)


def should_exclude(filename, category):
    """Exclude 500Hz sine/square files."""
    if category == "noise":
        return False
    match = re.search(r'_500(?:hz|hx)', filename, re.IGNORECASE)
    return match is not None


def split_category(image_dir, category):
    """Split one category into train/test, stratified by frequency."""
    cat_path = image_dir / category
    if not cat_path.exists():
        return [], []

    all_files = sorted(cat_path.glob("*.png"))

    # Filter out 500Hz
    if category in ("sine", "square"):
        before = len(all_files)
        all_files = [f for f in all_files if not should_exclude(f.name, category)]
        excluded = before - len(all_files)
        if excluded:
            print(f"  Excluded {excluded} files at {EXCLUDE_FREQ}Hz")

    # Group files by CSV source
    csv_groups = defaultdict(list)
    for f in all_files:
        csv_id = extract_csv_id(f.name)
        csv_groups[csv_id].append(f)

    # Group CSVs by frequency for stratified split
    freq_to_csvs = defaultdict(list)
    for csv_id in csv_groups:
        freq = extract_freq_group(csv_id)
        freq_to_csvs[freq].append(csv_id)

    train_files = []
    test_files = []

    random.seed(RANDOM_SEED)

    for freq, csv_ids in sorted(freq_to_csvs.items()):
        random.shuffle(csv_ids)
        n_train = max(1, round(len(csv_ids) * TRAIN_RATIO))

        train_csvs = csv_ids[:n_train]
        test_csvs = csv_ids[n_train:]

        for csv_id in train_csvs:
            train_files.extend(csv_groups[csv_id])
        for csv_id in test_csvs:
            test_files.extend(csv_groups[csv_id])

        print(f"    {freq}: {len(csv_ids)} CSVs -> {len(train_csvs)} train, {len(test_csvs)} test")

    return train_files, test_files


def copy_files(files, dest_dir, category):
    """Copy files to destination directory."""
    cat_dest = dest_dir / category
    cat_dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, cat_dest / f.name)


def main(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Clean existing split
    for d in [train_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)

    categories = ["noise", "sine", "square"]
    total_train = 0
    total_test = 0

    for cat in categories:
        print(f"\n{cat}:")
        train_files, test_files = split_category(image_dir, cat)

        copy_files(train_files, train_dir, cat)
        copy_files(test_files, test_dir, cat)

        print(f"  -> {len(train_files)} train, {len(test_files)} test")
        total_train += len(train_files)
        total_test += len(test_files)

    print(f"\n{'='*50}")
    print(f"Total: {total_train} train, {total_test} test")
    print(f"Train dir: {train_dir}")
    print(f"Test dir:  {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FFT data into train/test")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with noise/sine/square subfolders")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for train/ and test/ folders")
    args = parser.parse_args()

    main(args.image_dir, args.output_dir)
