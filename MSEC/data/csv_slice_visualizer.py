import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import shutil


def load_time_series_range(csv_path, start_idx=50, end_idx=200):
    """
    Load specific range of time series data from CSV file.
    Default range is 0 to 100 to avoid initial fluctuations.
    """
    df = pd.read_csv(csv_path)

    # Check if file has enough data points
    if len(df) < end_idx:
        print(f"  Warning: File {Path(csv_path).name} only has {len(df)} points. Adjusting range.")
        end_idx = len(df)
        if start_idx >= end_idx:
            start_idx = 0

    # Extract slice
    time_ms = df['Time_ms'].values[start_idx:end_idx]
    displacement = df['Absolute_Displacement_nm'].values[start_idx:end_idx]

    actual_count = len(time_ms)
    return displacement, time_ms, actual_count, start_idx, end_idx


def plot_original_data(data, time_ms, base_filename, output_dir, start, end):
    """Generate a clean plot for the original data slice"""
    plt.figure(figsize=(10, 6))
    plt.plot(time_ms, data, linewidth=1.5, color='#1f77b4')

    plt.title(f'{base_filename} - Original (Indices {start}-{end})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Displacement (nm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_name = f"{base_filename}_slice_{start}_{end}.png"
    png_path = output_dir / png_name

    # Handle filename conflicts
    counter = 1
    while png_path.exists():
        png_path = output_dir / f"{base_filename}_slice_{start}_{end}_{counter}.png"
        counter += 1

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    return png_path.name


def process_files(csv_input_pattern, output_folder_name, start_idx=50, end_idx=200):
    """Main pipeline to process CSVs and generate plots for original data only"""

    # Setup Paths
    desktop = Path.home() / "Desktop"
    output_dir = desktop / output_folder_name

    # Folder Management
    if output_dir.exists():
        resp = input(f"'{output_folder_name}' exists. Clear it? (y/n): ").lower()
        if resp == 'y':
            shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find Files
    csv_files = glob.glob(os.path.expanduser(csv_input_pattern))
    if not csv_files:
        print(f"No CSV files found at: {csv_input_pattern}")
        return

    print(f"Starting Visualization Pipeline")
    print(f"Target Range: Indices {start_idx} to {end_idx}")
    print(f"Found {len(csv_files)} files.")
    print("-" * 50)

    success_count = 0
    for csv_path in sorted(csv_files):
        try:
            base_name = Path(csv_path).stem
            # Load the specific 50-200 range
            data, time, count, s, e = load_time_series_range(csv_path, start_idx, end_idx)

            # Save plot
            file_saved = plot_original_data(data, time, base_name, output_dir, s, e)
            print(f"Processed: {base_name} -> {file_saved} ({count} pts)")
            success_count += 1

        except Exception as err:
            print(f"Error processing {csv_path}: {err}")

    print("-" * 50)
    print(f"COMPLETE: Successfully plotted {success_count} files.")
    print(f"Location: {output_dir}")


# ============================================================================
# CONFIGURATION
# ============================================================================
if __name__ == "__main__":
    # 1. Update this to your actual CSV folder path
    # Example: "C:/Users/Name/Desktop/csv/*.csv" or "~/Desktop/csv/*.csv"
    CSV_PATH_PATTERN = str(Path.home() / "Desktop" / "csv" / "output_square_csv" / "*.csv")

    # 2. Name of the folder on your desktop
    FOLDER_NAME = "square_Plots"

    # 3. Range selection
    START = 500
    END = 900

    process_files(CSV_PATH_PATTERN, FOLDER_NAME, START, END)
