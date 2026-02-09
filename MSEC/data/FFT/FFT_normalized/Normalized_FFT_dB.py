import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path


def load_time_series_range(csv_path, start_idx, end_idx):
    df = pd.read_csv(csv_path)
    if len(df) < end_idx:
        end_idx = len(df)
    time_ms = df['Time_ms'].values[start_idx:end_idx]
    displacement = df['Absolute_Displacement_nm'].values[start_idx:end_idx]
    return displacement, time_ms


def plot_normalized_fft_db(data, time_ms, base_filename, output_dir):
    n = len(data)
    if n < 2: return None

    # 1. DETRENDING (Removes the primary DC component from the time signal)
    data_centered = data - np.mean(data)

    # 2. WINDOWING
    window = np.hanning(n)
    data_windowed = data_centered * window

    # 3. FFT CALCULATION
    dt = np.mean(np.diff(time_ms)) / 1000.0
    fs = 1.0 / dt
    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)

    # 4. AMPLITUDE CALCULATION (Linear nm)
    amplitude_nm = np.abs(fft_values) * (2.0 / n) * 2.0

    # 5. REMOVE 0Hz (DC Peak)
    # We slice the arrays from index 1 to ignore the 0Hz component entirely
    freqs_final = fft_freqs[1:]
    amp_linear_final = amplitude_nm[1:]

    # 6. dB CONVERSION
    amplitude_db = 20 * np.log10(amp_linear_final + 1e-9)

    # 7. ZERO-CENTERING (Normalization)
    # Find the maximum dB value in the signal (excluding 0Hz)
    max_db = np.max(amplitude_db)
    normalized_db = amplitude_db - max_db

    # 8. PLOTTING
    plt.figure(figsize=(12, 7))
    plt.plot(freqs_final, normalized_db, linewidth=1.5, color='#0055aa')



    plt.title(f'Piezo FFT: {base_filename}', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Amplitude (dB)', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Set axis limits
    plt.xlim(0, 500)
    plt.ylim(np.min(normalized_db) - 5, 5)  # Provide 5dB headroom above the peak


    plt.tight_layout()

    plt.savefig(output_dir / f"{base_filename}_FFT.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Path configuration
    INPUT_PATH = Path.home() / "Desktop" / "csv" / "output_noise_csv"
    OUTPUT_FOLDER = Path.home() / "Desktop" / "Piezo_FFT_noise"
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Data range for classification
    START = 100
    END = 1024

    csv_files = glob.glob(str(INPUT_PATH / "*.csv"))
    print(f"Processing {len(csv_files)} files into zero-centered plots...")

    for f in csv_files:
        try:
            data, time = load_time_series_range(f, START, END)
            plot_normalized_fft_db(data, time, Path(f).stem, OUTPUT_FOLDER)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"Finished! Normalized plots are in: {OUTPUT_FOLDER}")
