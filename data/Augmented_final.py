import numpy as np
import pandas as pd
import tsaug
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import signal
import glob
from pathlib import Path
import shutil


def load_time_series_data(csv_path, n_points=250):
    """Load time series data from CSV file and take first n_points"""
    df = pd.read_csv(csv_path)
    time_ms = df['Time_ms'].values
    displacement = df['Absolute_Displacement_nm'].values

    if len(time_ms) < n_points:
        n_points = len(time_ms)
        print(f"  Warning: Using only {n_points} points (file too short)")

    return displacement[:n_points], time_ms[:n_points], n_points


def safe_plot(aug_data, plot_time, base_filename, aug_name, actual_points, output_dir):
    """Safely plot with automatic length matching"""
    # Force length matching
    min_len = min(len(aug_data), len(plot_time))
    aug_data = aug_data[:min_len]
    plot_time = plot_time[:min_len]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_time, aug_data, linewidth=1.5)
    plt.title(f'{base_filename} - {aug_name.title()} (First {actual_points} pts)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Displacement (nm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_name = f"{base_filename}_{aug_name}_first{actual_points}.png"
    png_path = output_dir / png_name

    # Handle filename conflicts
    counter = 1
    while png_path.exists():
        stem = png_path.stem
        png_name = f"{stem}_{counter}.png"
        png_path = output_dir / png_name
        counter += 1

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    return png_path.name


def augment_and_save_to_output(csv_path, output_dir, points_to_use=250):
    """Generate exactly 7 PNGs per CSV file"""
    try:
        data, time_ms, actual_points = load_time_series_data(csv_path, points_to_use)
        base_filename = Path(csv_path).stem
        print(f"\nProcessing: {base_filename} ({actual_points} points)")

        png_count = 0

        # 1-5: Standard augmentations (length preserved)
        simple_augs = [
            ("original", data),
            ("timewarp_mild",
             tsaug.TimeWarp(n_speed_change=2, max_speed_ratio=1.2).augment(data.reshape(1, -1, 1))[0, :, 0]),
            ("timewarp_moderate",
             tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=1.5).augment(data.reshape(1, -1, 1))[0, :, 0]),
            ("timewarp_strong",
             tsaug.TimeWarp(n_speed_change=4, max_speed_ratio=1.8).augment(data.reshape(1, -1, 1))[0, :, 0]),
            ("reverse", tsaug.Reverse().augment(data.reshape(1, -1, 1))[0, :, 0]),
        ]

        for aug_name, aug_data in simple_augs:
            try:
                safe_plot(aug_data, time_ms, base_filename, aug_name, actual_points, output_dir)
                png_count += 1
                print(f"  Saved: {base_filename}_{aug_name}_first{actual_points}.png")
            except Exception as e:
                print(f"  Failed {aug_name}: {e}")

        # 6-7: Pool augmentations with special time handling
        print("  Processing pool augmentations...")

        # Pool size 2
        try:
            pool2_data = tsaug.Pool(size=2).augment(data.reshape(1, -1, 1))[0, :, 0]
            pool2_time = np.linspace(time_ms[0], time_ms[-1], len(pool2_data))
            safe_plot(pool2_data, pool2_time, base_filename, "pool_size2", actual_points, output_dir)
            png_count += 1
            print(f"  Saved: {base_filename}_pool_size2_first{actual_points}.png")
        except Exception as e:
            print(f"  Failed pool_size2: {e}")

        # Pool size 3
        try:
            pool3_data = tsaug.Pool(size=3).augment(data.reshape(1, -1, 1))[0, :, 0]
            pool3_time = np.linspace(time_ms[0], time_ms[-1], len(pool3_data))
            safe_plot(pool3_data, pool3_time, base_filename, "pool_size3", actual_points, output_dir)
            png_count += 1
            print(f"  Saved: {base_filename}_pool_size3_first{actual_points}.png")
        except Exception as e:
            print(f"  Failed pool_size3: {e}")

        print(f"  Completed: {base_filename} -> {png_count}/7 PNGs")
        return png_count

    except Exception as e:
        print(f"CRITICAL ERROR {Path(csv_path).stem}: {e}")
        return 0


def one_click_pipeline(csv_input_pattern, output_folder_name, points_to_use=250):
    """
    Complete pipeline: CSV augmentation -> Single output folder

    CONFIGURATION (CHANGE THESE 3 LINES):
    csv_input_pattern: Path to your CSV files (e.g., "/path/to/sine/*.csv")
    output_folder_name: Output folder name (e.g., "SineWave", "SquareWave", "NoiseWave")
    points_to_use: Number of points to use from each CSV (default: 250)
    """
    desktop = Path.home() / "Desktop"
    output_dir = desktop / output_folder_name

    # Clear/create output folder
    if output_dir.exists():
        file_count = len(list(output_dir.rglob('*')))
        resp = input(f"'{output_folder_name}' exists ({file_count} files). Clear? (y/n): ").lower()
        if resp == 'y':
            shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Pipeline started. Output: {output_dir}")
    print("=" * 70)

    # Find CSV files
    csv_files = glob.glob(csv_input_pattern)
    if not csv_files:
        print(f"No CSV files found at: {csv_input_pattern}")
        return

    csv_files = sorted(list(set(csv_files)))
    print(f"Found {len(csv_files)} CSV files")
    print(f"Expecting {len(csv_files) * 7} PNG files")

    total_pngs = 0
    success_count = 0

    for csv_path in csv_files:
        png_count = augment_and_save_to_output(csv_path, output_dir, points_to_use)
        total_pngs += png_count
        if png_count > 0:
            success_count += 1

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print(f"Success: {success_count}/{len(csv_files)} CSV files processed")
    print(f"Generated: {total_pngs}/{len(csv_files) * 7} PNG files")
    print(f"Output folder: {output_dir}")
    print(f"Verified files: {len(list(output_dir.glob('*.png')))}")
    print("=" * 70)


# ============================================================================
# *** CONFIGURATION - MODIFY THESE 3 LINES ONLY ***
# ============================================================================
if __name__ == "__main__":
    # ============ CHANGE THESE SETTINGS ============
    CSV_INPUT_PATH = "desktop / "csv" / "output_noise_csv" / *.csv" # Update your path here!
    OUTPUT_FOLDER_NAME = "NoiseWave"  # Change to "SquareWave", "SineWave", etc.
    POINTS_TO_USE = 250  # Number of points from each CSV

    # ==============================================

    print("Time Series Augmentation Pipeline")
    print(f"Input:  {CSV_INPUT_PATH}")
    print(f"Output: Desktop/{OUTPUT_FOLDER_NAME}")
    print(f"Points: {POINTS_TO_USE}")
    print("-" * 50)

    one_click_pipeline(CSV_INPUT_PATH, OUTPUT_FOLDER_NAME, POINTS_TO_USE)
