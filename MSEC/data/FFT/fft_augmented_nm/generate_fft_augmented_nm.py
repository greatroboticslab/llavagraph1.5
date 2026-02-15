"""
Generate augmented FFT plots from piezo displacement CSVs.
Uses LINEAR AMPLITUDE (nm) on y-axis for clearer peak vs noise distinction.

Features:
- Y-axis: Linear amplitude in nanometers (not dB)
- Adaptive x-axis range based on auto-detected dominant peak (5x peak freq)
- Multiple segment offsets per CSV for data augmentation
- Generic plot titles (no wave type label) for model training
- Filenames retain wave type for human identification
- Only processes input signals up to 500Hz

Usage:
    python generate_fft_augmented_nm.py \
        --csv-dir /path/to/csv/output_sine_csv \
        --output-dir /path/to/output/sine \
        --wave-type sine
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import argparse
from pathlib import Path


# Segment offsets (start indices) for augmentation
# Each produces a different window from the same CSV
SEGMENT_OFFSETS = [100, 5000, 10000]
FFT_WINDOW_SIZE = 1024

# Larger window for low-frequency signals (better frequency resolution)
FFT_WINDOW_SIZE_LARGE = 4096
# Threshold: if dominant peak is below this, use the larger window
LOW_FREQ_THRESHOLD = 5.0  # Hz

# Minimum x-axis range in Hz
MIN_X_RANGE = 10

# Frequencies to include (up to 500Hz input signals only)
MAX_INPUT_FREQ = 500


def load_time_series_range(csv_path, start_idx, n_points):
    """Load a segment of time series data from CSV."""
    with open(csv_path, 'r') as f:
        header = f.readline()  # skip header
        lines = f.readlines()

    if start_idx + n_points > len(lines):
        return None, None

    time_ms = np.zeros(n_points)
    displacement = np.zeros(n_points)
    for i, line in enumerate(lines[start_idx:start_idx + n_points]):
        parts = line.strip().split(',')
        time_ms[i] = float(parts[0])
        displacement[i] = float(parts[1])

    return displacement, time_ms


def detect_dominant_frequency(fft_freqs, amplitude):
    """Find the frequency of the highest amplitude peak (excluding DC)."""
    if len(amplitude) < 2:
        return MIN_X_RANGE

    peak_idx = np.argmax(amplitude[1:]) + 1
    return fft_freqs[peak_idx]


def compute_adaptive_xlim(peak_freq, nyquist):
    """Compute x-axis limit: 5x the dominant peak, clamped to [MIN_X_RANGE, nyquist].
    Add 5% padding so edge peaks are not cut off."""
    x_max = max(10.0 * peak_freq, MIN_X_RANGE)
    x_max = min(x_max, nyquist)
    x_max_padded = x_max * 1.05
    x_min_padded = -x_max * 0.02
    return x_min_padded, x_max_padded


def generate_fft_plot(data, time_ms, base_filename, segment_label, output_dir):
    """Compute FFT and generate a linear amplitude plot with adaptive x-axis."""
    n = len(data)
    if n < 2:
        return None

    # 1. Detrend
    data_centered = data - np.mean(data)

    # 2. Hanning window
    window = np.hanning(n)
    data_windowed = data_centered * window

    # 3. FFT
    dt = np.mean(np.diff(time_ms)) / 1000.0  # convert ms to seconds
    fs = 1.0 / dt
    nyquist = fs / 2.0
    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)

    # 4. Amplitude in nanometers (compensate for windowing)
    amplitude = np.abs(fft_values) * (2.0 / n) * 2.0

    # 5. Remove DC component for plotting
    amp_no_dc = amplitude[1:]
    freqs_no_dc = fft_freqs[1:]

    # 6. Detect dominant frequency for adaptive x-axis
    peak_freq = detect_dominant_frequency(fft_freqs, amplitude)
    x_min, x_max = compute_adaptive_xlim(peak_freq, nyquist)

    # 7. Plot with linear amplitude
    plt.figure(figsize=(12, 7))
    plt.plot(freqs_no_dc, amp_no_dc, linewidth=1.5, color='#0055aa')

    # Generic title — strip wave type prefix for training
    display_name = re.sub(r'^(sine|square|noise|pulse|ramp)[-_]', '', base_filename, flags=re.IGNORECASE)
    plt.title(f'Piezo FFT Spectrum: {display_name} ({segment_label})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude (nm)', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.xlim(x_min, x_max)

    # Y-axis: start at 0, add 20% headroom above the tallest visible peak
    mask = (freqs_no_dc >= max(0, x_min)) & (freqs_no_dc <= x_max)
    if np.any(mask):
        visible_max = np.max(amp_no_dc[mask])
        plt.ylim(0, visible_max * 1.2)
    else:
        plt.ylim(0, np.max(amp_no_dc) * 1.2)

    plt.tight_layout()

    # Filename keeps the wave type for human identification
    out_name = f"{base_filename}_{segment_label}_FFT.png"
    out_path = output_dir / out_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_name


def should_include_file(filename):
    """Check if the file's input frequency is <= 500Hz."""
    fname = filename.lower()

    # Noise/pulse/ramp files have no frequency — always include
    if any(x in fname for x in ['noise', 'pulse', 'ramp']):
        return True

    # Extract frequency from filename pattern like sine_100Hz_100Hz_10_absolute.csv
    match = re.search(r'_(\d+)(?:hz|hx)', fname, re.IGNORECASE)
    if match:
        freq = int(match.group(1))
        return freq <= MAX_INPUT_FREQ

    # If we can't determine frequency, include it
    return True


def process_directory(csv_dir, output_dir, wave_type):
    """Process all CSVs in a directory, generating FFT plots with augmentation."""
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    csv_files = [f for f in csv_files if should_include_file(Path(f).name)]
    print(f"Found {len(csv_files)} CSV files (filtered to <= {MAX_INPUT_FREQ}Hz)")

    total_plots = 0
    failed = 0

    for csv_path in csv_files:
        base_filename = Path(csv_path).stem

        # Use larger window for ramp trail files (low-freq ~1Hz needs finer resolution)
        is_ramp_trail = 'trail' in base_filename.lower()
        win_size = FFT_WINDOW_SIZE_LARGE if is_ramp_trail else FFT_WINDOW_SIZE

        for i, offset in enumerate(SEGMENT_OFFSETS):
            segment_label = f"seg{i+1}"

            try:
                data, time_ms = load_time_series_range(csv_path, offset, win_size)
                if data is None:
                    print(f"  Skipped {base_filename} seg{i+1}: not enough data at offset {offset}")
                    continue

                out_name = generate_fft_plot(data, time_ms, base_filename, segment_label, output_dir)
                if out_name:
                    total_plots += 1
                    print(f"  Generated: {out_name}")
                else:
                    failed += 1
            except Exception as e:
                print(f"  Error {base_filename} seg{i+1}: {e}")
                failed += 1

    print(f"\nComplete! Generated {total_plots} FFT plots ({failed} failed)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented FFT plots (linear amplitude nm)")
    parser.add_argument("--csv-dir", type=str, required=True,
                        help="Directory containing CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save FFT plot PNGs")
    parser.add_argument("--wave-type", type=str, required=True,
                        choices=["sine", "square", "noise", "pulse", "ramp"],
                        help="Wave type (used for logging only, not in plot titles)")
    args = parser.parse_args()

    print(f"FFT Augmented Plot Generator (Linear Amplitude nm)")
    print(f"Input:  {args.csv_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Type:   {args.wave_type}")
    print(f"Segments per CSV: {len(SEGMENT_OFFSETS)} (offsets: {SEGMENT_OFFSETS})")
    print(f"FFT window: {FFT_WINDOW_SIZE} points (ramp trail: {FFT_WINDOW_SIZE_LARGE} points)")
    print("-" * 60)

    process_directory(args.csv_dir, args.output_dir, args.wave_type)
