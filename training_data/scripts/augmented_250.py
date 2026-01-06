import numpy as np
import pandas as pd
import tsaug
import matplotlib.pyplot as plt
import os
from scipy import signal
import glob

# ============================================================================
# CONFIGURATION (TEMPLATE)
# ============================================================================

# Name of the output root directory for augmented files
OUTPUT_DIR = "augmented_first250_output"

# Number of points to use from each file
POINTS_TO_USE = 250


# Replace BASE_DIR with your own directory when using this script
BASE_DIR = "/path/to/your/csv_directory"

file_patterns = [
    os.path.join(BASE_DIR, "*.csv"),
    # Add more patterns if needed, e.g.:
    # os.path.join(BASE_DIR, "subfolder", "*.csv"),
]

# ============================================================================
# MAIN LOGIC
# ============================================================================

# Create output directory for augmented files
output_dir = OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")


def load_time_series_data(csv_path, n_points=250):
    """Load time series data from CSV file and take first n_points."""
    df = pd.read_csv(csv_path)

    # Extract time and displacement values
    time_ms = df["Time_ms"].values
    displacement = df["Delta_Displacement_nm"].values

    # Check if we have enough points
    if len(time_ms) < n_points:
        print(
            f" Warning: File has only {len(time_ms)} points, "
            f"using all available points"
        )
        n_points = len(time_ms)

    # Take first n_points
    time_ms_first = time_ms[:n_points]
    displacement_first = displacement[:n_points]

    # Reshape for tsaug: (n_series, n_timesteps, n_features)
    X = displacement_first.reshape(1, -1, 1)

    return X, time_ms_first, n_points


def augment_and_save_single_file(csv_path, output_dir, points_to_use=250):
    """Process a single CSV file and generate all augmentations."""
    # Load first n points of data
    X, time_ms, actual_points = load_time_series_data(
        csv_path, n_points=points_to_use
    )

    # Extract base filename for plot titles (without extension)
    base_filename = os.path.basename(csv_path)
    filename_no_ext = os.path.splitext(base_filename)[0]

    print(f"\nProcessing: {base_filename}")
    print(f" Using first {actual_points} points")
    print(f" Data shape: {X.shape}")

    # Create subdirectory for this file's augmentations
    file_output_dir = os.path.join(output_dir, filename_no_ext)
    os.makedirs(file_output_dir, exist_ok=True)

    # 1. Save original data plot
    print(" Saving original plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(time_ms, X[0, :, 0], linewidth=1.5)
    plt.title(
        f"{filename_no_ext} - Original (First {actual_points} points)",
        fontsize=14,
    )
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Displacement (nm)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plot_filename = os.path.join(
        file_output_dir,
        f"{filename_no_ext}_original_first{actual_points}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # 2. Time Warp augmentations
    print(" Applying Time Warp augmentations...")
    timewarp_configs = [
        ("timewarp_mild", tsaug.TimeWarp(n_speed_change=2, max_speed_ratio=1.2)),
        ("timewarp_moderate", tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=1.5)),
        ("timewarp_strong", tsaug.TimeWarp(n_speed_change=4, max_speed_ratio=1.8)),
    ]

    for aug_name, augmenter in timewarp_configs:
        X_aug = augmenter.augment(X)
        plt.figure(figsize=(10, 6))
        plt.plot(time_ms, X_aug[0, :, 0], linewidth=1.5)

        # Format display name for title
        display_name = aug_name.replace("_", " ").title()

        plt.title(
            f"{filename_no_ext} - {display_name} "
            f"(First {actual_points} points)",
            fontsize=14,
        )
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Displacement (nm)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plot_filename = os.path.join(
            file_output_dir,
            f"{filename_no_ext}_{aug_name}_first{actual_points}.png",
        )
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    # 3. Reverse augmentation
    print(" Applying Reverse augmentation...")
    X_reverse = tsaug.Reverse().augment(X)
    plt.figure(figsize=(10, 6))
    plt.plot(time_ms, X_reverse[0, :, 0], linewidth=1.5)
    plt.title(
        f"{filename_no_ext} - Reverse (First {actual_points} points)",
        fontsize=14,
    )
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Displacement (nm)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plot_filename = os.path.join(
        file_output_dir,
        f"{filename_no_ext}_reverse_first{actual_points}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # 4. Pool augmentations
    print(" Applying Pool augmentations...")
    pool_configs = [
        ("pool_size2", tsaug.Pool(size=2)),
        ("pool_size3", tsaug.Pool(size=3)),
    ]

    for aug_name, augmenter in pool_configs:
        X_aug = augmenter.augment(X)
        pool_size = int(aug_name.split("size")[1])

        # Adjust time axis for plotting (pooling reduces length)
        time_pooled = signal.resample(time_ms, len(X_aug[0, :, 0]))

        plt.figure(figsize=(10, 6))
        plt.plot(time_pooled, X_aug[0, :, 0], linewidth=1.5)
        plt.title(
            f"{filename_no_ext} - Pool Size {pool_size} "
            f"(First {actual_points} points)",
            fontsize=14,
        )
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Displacement (nm)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plot_filename = os.path.join(
            file_output_dir,
            f"{filename_no_ext}_{aug_name}_first{actual_points}.png",
        )
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    print(f" ✓ Completed processing: {filename_no_ext}")
    print(f" Output saved to: {file_output_dir}/")

    return file_output_dir, actual_points


def process_multiple_files(file_patterns, points_to_use=250):
    """Process multiple CSV files matching the given patterns."""
    all_csv_files = []

    # Collect all CSV files from the patterns
    for pattern in file_patterns:
        matched_files = glob.glob(pattern)
        all_csv_files.extend(matched_files)

    # Remove duplicates and sort
    all_csv_files = sorted(list(set(all_csv_files)))

    if not all_csv_files:
        print("No CSV files found matching the patterns!")
        return

    print(f"Found {len(all_csv_files)} CSV file(s) to process:")
    for i, file in enumerate(all_csv_files, 1):
        print(f" {i}. {file}")

    # Process each file
    print("\n" + "=" * 60)
    print(f"STARTING BATCH PROCESSING (First {points_to_use} points only)")
    print("=" * 60)

    processed_files = []

    for csv_path in all_csv_files:
        try:
            # Check if file has enough data points
            df_check = pd.read_csv(csv_path)
            total_points = len(df_check)

            if total_points < points_to_use:
                print(
                    f"\n⚠️ Warning: {os.path.basename(csv_path)} has only "
                    f"{total_points} points (< {points_to_use})"
                )

            output_subdir, actual_used = augment_and_save_single_file(
                csv_path, output_dir, points_to_use
            )
            processed_files.append((csv_path, output_subdir, actual_used))

        except Exception as e:
            print(f"\n✗ Error processing {csv_path}: {str(e)}")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f" Total files processed: {len(processed_files)}")
    print(f" Output directory: {os.path.abspath(output_dir)}")
    print(f" Points used per file: First {points_to_use} points")

    # Show points used for each file
    print("\nPoints used for each file:")
    for csv_path, _, points_used in processed_files:
        filename = os.path.basename(csv_path)
        print(f" {filename}: {points_used} points")

    # Show directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)

        # Show only first few files to avoid clutter
        png_files = [f for f in sorted(files) if f.endswith(".png")]
        for file in png_files[:5]:
            print(f"{subindent}{file}")
        if len(png_files) > 5:
            print(f"{subindent}... and {len(png_files) - 5} more files")

    return processed_files


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("CONFIGURATION:")
    print(f" Using first {POINTS_TO_USE} points from each file")
    print(f" Output directory: {OUTPUT_DIR}")
    print(f" Base directory: {BASE_DIR}")
    print(f" File patterns: {file_patterns}")

    processed = process_multiple_files(
        file_patterns, points_to_use=POINTS_TO_USE
    )

    if processed:
        print(
            f"\nTotal augmentations generated: {len(processed) * 7} PNG files"
        )
        print(
            "(Each file generates: Original + 3 TimeWarp + 1 Reverse + 2 Pool)"
        )
        print(
            f"\nAll plots show only the first {POINTS_TO_USE} points "
            f"of each time series."
        )
