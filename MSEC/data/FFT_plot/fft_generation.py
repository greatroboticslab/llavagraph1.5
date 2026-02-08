import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from scipy.signal import find_peaks


def load_time_series_range(csv_path, start_idx, end_idx):
    df = pd.read_csv(csv_path)
    if len(df) < end_idx:
        end_idx = len(df)
    time_ms = df['Time_ms'].values[start_idx:end_idx]
    displacement = df['Absolute_Displacement_nm'].values[start_idx:end_idx]
    return displacement, time_ms


def plot_professional_fft(data, time_ms, base_filename, output_dir):
    n = len(data)
    if n < 2: return None

    # 1. DETRENDING (Removes the giant 0Hz spike seen in your first image)
    data_centered = data - np.mean(data)

    # 2. WINDOWING (Makes peaks sharp like your reference image)
    window = np.hanning(n)
    data_windowed = data_centered * window

    # 3. FFT CALCULATION
    dt = np.mean(np.diff(time_ms)) / 1000.0
    fs = 1.0 / dt
    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)

    # 4. AMPLITUDE NORMALIZATION
    # (2/n) converts FFT units back to actual Nanometers (nm)
    # * 2 accounts for Hanning window energy loss
    amplitude = np.abs(fft_values) * (2.0 / n) * 2.0

    # 5. PEAK DETECTION (For Classification)
    # Finds the single highest vibration peak to label it
    peaks, _ = find_peaks(amplitude[1:], height=np.max(amplitude[1:]) * 0.5)

    plt.figure(figsize=(10, 6))

    # We plot from index 1 to ignore any remaining 0Hz offset
    plt.plot(fft_freqs[1:], amplitude[1:], linewidth=1, color='#0055aa', label='Displacement Spectrum')



    plt.title(f'Piezo FFT: {base_filename}', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=10)
    plt.ylabel('Amplitude (nm)', fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Zoom into the meaningful range (Adjust 500 if your piezo vibrates faster)
    plt.xlim(0, 500)

    # Use a slight margin on Y so peaks don't touch the top
    plt.ylim(0, np.max(amplitude[1:]) * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / f"{base_filename}_Amplitude_FFT.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    CSV_PATH_PATTERN = str(Path.home() / "Desktop" / "csv" / "output_square_csv" / "*.csv")
    OUTPUT_FOLDER = Path.home() / "Desktop" / "Piezo_FFT_square"
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # IMPORTANT: Use a longer range (1024 pts) for the 'noisy floor' look
    START = 150
    END = 900

    csv_files = glob.glob(os.path.expanduser(CSV_PATH_PATTERN))
    for f in csv_files:
        data, time = load_time_series_range(f, START, END)
        plot_professional_fft(data, time, Path(f).stem, OUTPUT_FOLDER)

    print(f"Finished! Plots are in: {OUTPUT_FOLDER}")