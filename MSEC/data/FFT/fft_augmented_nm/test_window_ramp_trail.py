"""
Test different actual window sizes (real data, not zero-padded) on ramp trail signals.
More real data = true frequency resolution improvement.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/Archive/ramp_csv/Ramp-trail2_5_absolute.csv")
OUTPUT_DIR = Path("/ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/data/FFT/fft_augmented_nm/nfft_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZES = [1024, 2048, 4096, 8192]
START = 100

# First check how much data is available
with open(CSV_PATH, 'r') as f:
    f.readline()
    total_lines = sum(1 for _ in f)
print(f"Total data points available: {total_lines}")


def load_data(csv_path, start, n_points):
    with open(csv_path, 'r') as f:
        f.readline()
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


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, win_size in enumerate(WINDOW_SIZES):
    data, time_ms = load_data(CSV_PATH, START, win_size)
    if data is None:
        print(f"Not enough data for window size {win_size}")
        axes[idx].set_title(f'Window = {win_size}: NOT ENOUGH DATA')
        continue

    n = len(data)
    data_centered = data - np.mean(data)
    window = np.hanning(n)
    data_windowed = data_centered * window

    dt = np.mean(np.diff(time_ms)) / 1000.0
    fs = 1.0 / dt

    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)
    amplitude = np.abs(fft_values) * (2.0 / n) * 2.0

    mask = (fft_freqs > 0) & (fft_freqs <= 50)

    ax = axes[idx]
    ax.plot(fft_freqs[mask], amplitude[mask], linewidth=1.2, color='#0055aa')
    freq_res = fs / n
    duration = n * dt
    ax.set_title(f'Window = {win_size} pts ({duration:.1f}s, Δf = {freq_res:.3f} Hz)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (nm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 50)
    print(f"Window {win_size}: {duration:.1f}s of data, freq resolution {freq_res:.3f} Hz")

fig.suptitle('Ramp-trail2_5: FFT with different ACTUAL window sizes', fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "window_size_comparison_ramp_trail2_5.png", dpi=200)
plt.close()
print(f"Saved to {OUTPUT_DIR / 'window_size_comparison_ramp_trail2_5.png'}")
