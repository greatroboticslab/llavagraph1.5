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

def plot_professional_fft_db(data, time_ms, base_filename, output_dir):
    n = len(data)
    if n < 2: return None

    # 1. DETRENDING
    data_centered = data - np.mean(data)

    # 2. WINDOWING
    window = np.hanning(n)
    data_windowed = data_centered * window

    # 3. FFT CALCULATION
    dt = np.mean(np.diff(time_ms)) / 1000.0
    fs = 1.0 / dt
    fft_values = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, d=dt)

    # 4. AMPLITUDE NORMALIZATION (Linear nm)
    amplitude_nm = np.abs(fft_values) * (2.0 / n) * 2.0

    # 5. dB CONVERSION (Reference = 1 nm)
    # We add 1e-9 to avoid log(0) errors
    amplitude_db = 20 * np.log10(amplitude_nm + 1e-9)

    plt.figure(figsize=(10, 6))

    # 6. PLOTTING (Using your preferred blue #0055aa)
    # Plotting from index 1 to ignore DC residual
    plt.plot(fft_freqs[1:], amplitude_db[1:], linewidth=1, color='#0055aa', label='dB Spectrum')

    plt.title(f'Piezo FFT (dB Scale): {base_filename}', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=10)
    plt.ylabel('Amplitude (dB)', fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # 7. AXIS SETTINGS
    # X-axis: Still showing up to 500Hz as per your original
    plt.xlim(0, 500)

    # Y-axis: Auto-scale for dB visibility
    y_min = np.min(amplitude_db[1:])
    y_max = np.max(amplitude_db[1:])
    plt.ylim(y_max - 80, y_max + 10) # Shows 80dB dynamic range

    plt.tight_layout()
    plt.savefig(output_dir / f"{base_filename}_dB_FFT.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    CSV_PATH_PATTERN = str(Path.home() / "Desktop" / "csv" / "output_noise_csv" / "*.csv")
    # New folder for dB results
    OUTPUT_FOLDER = Path.home() / "Desktop" / "Piezo_FFT_dB_noise"
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Keeping your specific range
    START = 100
    END = 1024

    csv_files = glob.glob(os.path.expanduser(CSV_PATH_PATTERN))
    print(f"Processing {len(csv_files)} files into dB plots...")

    for f in csv_files:
        try:
            data, time = load_time_series_range(f, START, END)
            plot_professional_fft_db(data, time, Path(f).stem, OUTPUT_FOLDER)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"Finished! dB plots are in: {OUTPUT_FOLDER}")