import numpy as np
import pandas as pd
import tsaug
import matplotlib.pyplot as plt
import os
from scipy import signal

# Create output directory for augmented files
output_dir = "augmented_data"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

def load_time_series_data(csv_path):
    """Load time series data from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Extract time and displacement values
    time_ms = df['Time_ms'].values
    displacement = df['Delta_Displacement_nm'].values
    
    # Reshape for tsaug: (n_series, n_timesteps, n_features)
    X = displacement.reshape(1, -1, 1)
    
    return X, time_ms

# Load data from CSV file
csv_path = "/Users/ilminurablikim/Desktop/1/Run2_relative.csv"
X, time_ms = load_time_series_data(csv_path)

print(f"Original data shape: {X.shape}")
print(f"Time points: {len(time_ms)}")

# Extract base filename for plot titles
base_filename = os.path.basename(csv_path).replace('.csv', '')

# 1. Save original data plot
print("\n" + "="*50)
print("SAVING ORIGINAL PLOT")
print("="*50)

plt.figure(figsize=(10, 6))

# Plot using default matplotlib color (blue for single line)
plt.plot(time_ms, X[0, :, 0], linewidth=1.5)

plt.title(f'{base_filename} - Original', fontsize=14)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Displacement (nm)', fontsize=12)
plt.grid(True, alpha=0.3)

# Save plot
plot_filename = os.path.join(output_dir, "original.png")
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)
plt.close()

print(f"Saved: {plot_filename}")

# Save original data as CSV only
df_original = pd.DataFrame({
    'Time_ms': time_ms,
    'Delta_Displacement_nm': X[0, :, 0]
})
df_original.to_csv(os.path.join(output_dir, "original.csv"), index=False)

# 2. Time Warp Augmentations
print("\n" + "="*50)
print("TIME WARP AUGMENTATIONS")
print("="*50)

# Define time warp configurations
timewarp_configs = [
    ("timewarp_mild", tsaug.TimeWarp(n_speed_change=2, max_speed_ratio=1.2)),
    ("timewarp_moderate", tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=1.5)),
    ("timewarp_strong", tsaug.TimeWarp(n_speed_change=4, max_speed_ratio=1.8)),
]

for name, augmenter in timewarp_configs:
    # Apply augmentation
    X_aug = augmenter.augment(X)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot with default color (blue)
    plt.plot(time_ms, X_aug[0, :, 0], linewidth=1.5)
    
    # Format plot title
    aug_name = name.replace('_', ' ').title()
    plt.title(f'{base_filename} - {aug_name}', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Displacement (nm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = os.path.join(output_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    # Save data as CSV only
    df_aug = pd.DataFrame({
        'Time_ms': time_ms,
        'Delta_Displacement_nm': X_aug[0, :, 0]
    })
    df_aug.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    
    print(f"Saved: {plot_filename}")

# 3. Reverse Augmentation
print("\n" + "="*50)
print("REVERSE AUGMENTATION")
print("="*50)

# Apply reverse augmentation
X_reverse = tsaug.Reverse().augment(X)

# Create figure
plt.figure(figsize=(10, 6))

# Plot with default color (blue)
plt.plot(time_ms, X_reverse[0, :, 0], linewidth=1.5)

plt.title(f'{base_filename} - Reverse', fontsize=14)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Displacement (nm)', fontsize=12)
plt.grid(True, alpha=0.3)

# Save plot
plot_filename = os.path.join(output_dir, "reverse.png")
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)
plt.close()

# Save data as CSV only
df_reverse = pd.DataFrame({
    'Time_ms': time_ms,
    'Delta_Displacement_nm': X_reverse[0, :, 0]
})
df_reverse.to_csv(os.path.join(output_dir, "reverse.csv"), index=False)

print(f"Saved: {plot_filename}")

# 4. Pool Augmentations
print("\n" + "="*50)
print("POOL AUGMENTATIONS")
print("="*50)

# Define pool configurations
pool_configs = [
    ("pool_size2", tsaug.Pool(size=2)),
    ("pool_size3", tsaug.Pool(size=3)),
]

for name, augmenter in pool_configs:
    # Apply augmentation
    X_aug = augmenter.augment(X)
    pool_size = int(name.split('size')[1])
    
    # Adjust time axis for plotting (pooling reduces length)
    time_pooled = signal.resample(time_ms, len(X_aug[0, :, 0]))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot with default color (blue)
    plt.plot(time_pooled, X_aug[0, :, 0], linewidth=1.5)
    
    plt.title(f'{base_filename} - Pool Size {pool_size}', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Displacement (nm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = os.path.join(output_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    # Save data as CSV only
    df_aug = pd.DataFrame({
        'Time_ms': time_pooled,
        'Delta_Displacement_nm': X_aug[0, :, 0]
    })
    df_aug.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    
    print(f"Saved: {plot_filename}")

# Print completion message
print("\n" + "="*50)
print("AUGMENTATION COMPLETE!")
print("="*50)
print(f"\nAll files saved to: {output_dir}/")

# List generated PNG files
print(f"\nGenerated PNG files:")
print("-" * 40)
for file in sorted(os.listdir(output_dir)):
    if file.endswith('.png'):
        print(f"  {file}")