"""
Generate FFT plots for 2.20.2026 data using the same logic as
generate_fft_augmented_nm.py, adapted for the .txt file format.

Input format:
    Sample Frequency = 1000 Hz
    D:<displacement> N:<sample_number>
    ...

Output: FFT plots saved to MSEC/data/2.20.2026/fft/{category}/

Usage:
    python generate_fft_2.20.2026.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_BASE = BASE_DIR / "fft"

CATEGORIES = {
    "sine":   BASE_DIR / "sine",
    "Square": BASE_DIR / "Square",
    "noise":  BASE_DIR / "noise",
    "pulse":  BASE_DIR / "pulse",
    "ramp":   BASE_DIR / "ramp",
}

SEGMENT_OFFSETS = [100, 5000, 10000]
FFT_WINDOW_SIZE = 1024
FFT_WINDOW_SIZE_LARGE = 4096
LOW_FREQ_THRESHOLD = 5.0  # Hz — use large window if dominant peak below this
MIN_X_RANGE = 10


def load_txt(filepath):
    """Parse .txt file, return (fs, displacement_array)."""
    fs = 1000.0
    d_values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Sample Frequency'):
                fs = float(line.split('=')[1].strip().split()[0])
            elif line.startswith('D:'):
                d_values.append(int(line.split()[0].split(':')[1]))
    return fs, np.array(d_values, dtype=np.float64)


def get_segment(signal, fs, start_idx, n_points):
    """Extract a segment and build time_ms array."""
    if start_idx + n_points > len(signal):
        return None, None
    seg = signal[start_idx:start_idx + n_points]
    dt_ms = 1000.0 / fs
    time_ms = np.arange(n_points) * dt_ms
    return seg, time_ms


def detect_dominant_frequency(fft_freqs, amplitude):
    """Frequency of highest amplitude peak (excluding DC)."""
    peak_idx = np.argmax(amplitude[1:]) + 1
    return fft_freqs[peak_idx]


def compute_adaptive_xlim(peak_freq, nyquist):
    """5x dominant peak, clamped to [MIN_X_RANGE, nyquist], with padding."""
    x_max = max(10.0 * peak_freq, MIN_X_RANGE)
    x_max = min(x_max, nyquist)
    return -x_max * 0.02, x_max * 1.05


def generate_fft_plot(data, time_ms, base_filename, segment_label, output_dir):
    """Compute FFT and save linear amplitude plot — same logic as original script."""
    n = len(data)
    data_centered = data - np.mean(data)
    window = np.hanning(n)
    data_windowed = data_centered * window

    dt = np.mean(np.diff(time_ms)) / 1000.0
    fs = 1.0 / dt
    nyquist = fs / 2.0

    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)
    amplitude = np.abs(fft_values) * (2.0 / n) * 2.0

    amp_no_dc = amplitude[1:]
    freqs_no_dc = fft_freqs[1:]

    peak_freq = detect_dominant_frequency(fft_freqs, amplitude)
    x_min, x_max = compute_adaptive_xlim(peak_freq, nyquist)

    plt.figure(figsize=(12, 7))
    plt.plot(freqs_no_dc, amp_no_dc, linewidth=1.5, color='#0055aa')

    display_name = re.sub(r'^(sine|square|noise|pulse|ramp)[-_]', '',
                          base_filename, flags=re.IGNORECASE)
    plt.title(f'Piezo FFT Spectrum: {display_name} ({segment_label})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude (nm)', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlim(x_min, x_max)

    mask = (freqs_no_dc >= max(0, x_min)) & (freqs_no_dc <= x_max)
    visible_max = np.max(amp_no_dc[mask]) if np.any(mask) else np.max(amp_no_dc)
    plt.ylim(0, visible_max * 1.2)

    plt.tight_layout()
    out_path = output_dir / f"{base_filename}_{segment_label}_FFT.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path.name


def is_low_freq(filename, fs, signal):
    """Use large FFT window for 1Hz signals."""
    fname = filename.lower()
    if '1hz' in fname or '-1-' in fname:
        return True
    # Also auto-detect: dominant peak below threshold
    n = min(FFT_WINDOW_SIZE, len(signal))
    seg = signal[:n] - np.mean(signal[:n])
    fft_vals = np.fft.rfft(seg * np.hanning(n))
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    amplitude = np.abs(fft_vals)
    peak_freq = detect_dominant_frequency(fft_freqs, amplitude)
    return peak_freq < LOW_FREQ_THRESHOLD


def process_category(cat_name, cat_dir, output_dir):
    txt_files = sorted(glob.glob(str(cat_dir / "*.txt")))
    if not txt_files:
        print(f"  No .txt files found in {cat_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    failed = 0

    for txt_path in txt_files:
        base_filename = Path(txt_path).stem
        fs, signal = load_txt(txt_path)

        win_size = FFT_WINDOW_SIZE_LARGE if is_low_freq(base_filename, fs, signal) \
                   else FFT_WINDOW_SIZE

        for i, offset in enumerate(SEGMENT_OFFSETS):
            seg_label = f"seg{i+1}"
            try:
                data, time_ms = get_segment(signal, fs, offset, win_size)
                if data is None:
                    print(f"    Skipped {base_filename} {seg_label}: not enough data")
                    continue
                out_name = generate_fft_plot(data, time_ms, base_filename,
                                             seg_label, output_dir)
                print(f"    {out_name}")
                total += 1
            except Exception as e:
                print(f"    Error {base_filename} {seg_label}: {e}")
                failed += 1

    print(f"  {cat_name}: {total} plots generated ({failed} failed)")


def main():
    print("FFT Plot Generator for 2.20.2026 data")
    print(f"Output base: {OUTPUT_BASE}")
    print()

    for cat_name, cat_dir in CATEGORIES.items():
        print(f"Processing {cat_name}...")
        out_dir = OUTPUT_BASE / cat_name
        process_category(cat_name, cat_dir, out_dir)
        print()

    print("Done!")


if __name__ == '__main__':
    main()
