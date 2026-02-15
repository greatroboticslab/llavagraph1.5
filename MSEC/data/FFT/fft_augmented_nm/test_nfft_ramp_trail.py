"""
Test different nfft sizes on ramp trail signals to find one that separates peaks.
Generates comparison plots for Ramp-trail2_5 with nfft = 1024, 4096, 8192, 16384.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/Archive/ramp_csv/Ramp-trail2_5_absolute.csv")
OUTPUT_DIR = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/FFT/fft_augmented_nm/nfft_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NFFT_VALUES = [1024, 4096, 8192, 16384]
START = 100
WINDOW_SIZE = 1024  # actual data points


def load_data(csv_path, start, n_points):
    with open(csv_path, 'r') as f:
        f.readline()  # skip header
        lines = f.readlines()
    if start + n_points > len(lines):
        return None, None
    time_ms = np.zeros(n_points)
    disp = np.zeros(n_points)
    for i, line in enumerate(lines[start:start + n_points]):
        parts = line.strip().split(',')
        time_ms[i] = float(parts[0])
        disp[i] = float(parts[1])
    return disp, time_ms


data, time_ms = load_data(CSV_PATH, START, WINDOW_SIZE)
if data is None:
    print("Not enough data")
    exit(1)

# Detrend + window
data_centered = data - np.mean(data)
window = np.hanning(len(data))
data_windowed = data_centered * window

dt = np.mean(np.diff(time_ms)) / 1000.0
fs = 1.0 / dt
print(f"Sampling rate: {fs:.1f} Hz, data points: {len(data)}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, nfft in enumerate(NFFT_VALUES):
    # Zero-pad if nfft > data length
    if nfft > len(data_windowed):
        padded = np.zeros(nfft)
        padded[:len(data_windowed)] = data_windowed
    else:
        padded = data_windowed

    fft_values = np.fft.rfft(padded, n=nfft)
    fft_freqs = np.fft.rfftfreq(nfft, d=dt)
    amplitude = np.abs(fft_values) * (2.0 / len(data)) * 2.0  # normalize by actual data length

    # Plot 0-50 Hz to see the low-freq peaks
    mask = (fft_freqs > 0) & (fft_freqs <= 50)

    ax = axes[idx]
    ax.plot(fft_freqs[mask], amplitude[mask], linewidth=1.2, color='#0055aa')
    ax.set_title(f'nfft = {nfft} (freq resolution: {fs/nfft:.3f} Hz)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (nm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 50)

fig.suptitle('Ramp-trail2_5: FFT with different nfft (zero-padded)', fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "nfft_comparison_ramp_trail2_5.png", dpi=200)
plt.close()
print(f"Saved comparison plot to {OUTPUT_DIR / 'nfft_comparison_ramp_trail2_5.png'}")
