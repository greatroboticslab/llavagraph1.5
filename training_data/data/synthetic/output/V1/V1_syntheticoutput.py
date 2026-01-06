#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 11:01:24 2025

@author: ilminurablikim
"""

"""
Synthetic Piezoelectric Data Generator for Data Augmentation
Generates multiple synthetic datasets based on real sensor data statistics
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
import os

# Create output directory
os.makedirs("augmented_data", exist_ok=True)

def read_and_analyze_real_data(filename="/Users/ilminurablikim/Desktop/Run1.txt"):
    """
    Read real piezoelectric sensor data and analyze its statistical properties
    
    Args:
        filename: Path to the real data file
        
    Returns:
        df: DataFrame of real data
        stats: Dictionary of statistical properties
    """
    print("Reading and analyzing real sensor data...")
    
    data = []
    sample_freq = 1000  # Default sampling frequency
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Extract sampling frequency from header
        if 'Sample Frequency' in line:
            match = re.search(r'= (\d+) Hz', line)
            if match:
                sample_freq = int(match.group(1))
        
        # Parse data lines (format: D:727 N:1787216)
        elif line.startswith('D:'):
            parts = line.split()
            if len(parts) >= 2:
                d_value = int(parts[0].split(':')[1])
                n_value = int(parts[1].split(':')[1])
                data.append({'D': d_value, 'N': n_value})
    
    df = pd.DataFrame(data)
    
    # Calculate statistical properties
    d_counts = Counter(df['D'])
    total_samples = len(df)
    
    # Calculate probabilities
    p_727 = d_counts.get(727, 0) / total_samples
    p_728 = d_counts.get(728, 0) / total_samples
    
    # Calculate Markov transition probabilities
    transitions = []
    for i in range(1, total_samples):
        transitions.append((df['D'].iloc[i-1], df['D'].iloc[i]))
    
    # Transition probabilities
    transitions_from_727 = [t for t in transitions if t[0] == 727]
    transitions_from_728 = [t for t in transitions if t[0] == 728]
    
    p_727_to_727 = (sum(1 for f, t in transitions_from_727 if t == 727) / 
                    max(1, len(transitions_from_727)))
    p_727_to_728 = (sum(1 for f, t in transitions_from_727 if t == 728) / 
                    max(1, len(transitions_from_727)))
    p_728_to_727 = (sum(1 for f, t in transitions_from_728 if t == 727) / 
                    max(1, len(transitions_from_728)))
    p_728_to_728 = (sum(1 for f, t in transitions_from_728 if t == 728) / 
                    max(1, len(transitions_from_728)))
    
    print(f"Real data analysis complete:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - 727 frequency: {p_727*100:.2f}%")
    print(f"  - 728 frequency: {p_728*100:.2f}%")
    print(f"  - Sampling frequency: {sample_freq} Hz")
    
    return df, {
        'p_727': p_727,
        'p_728': p_728,
        'p_transitions': {
            '727_727': p_727_to_727,
            '727_728': p_727_to_728,
            '728_727': p_728_to_727,
            '728_728': p_728_to_728
        },
        'sample_freq': sample_freq,
        'last_N': df['N'].iloc[-1]
    }

def generate_continuous_dataset(stats, dataset_id, num_samples=2000):
    """
    Generate continuous synthetic data using Markov chain model
    
    Args:
        stats: Statistical properties from real data
        dataset_id: ID number for this dataset
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    print(f"  Generating continuous dataset #{dataset_id} ({num_samples} samples)...")
    
    p_trans = stats['p_transitions']
    
    # Initialize Markov chain
    synthetic_values = []
    
    # First value based on overall probability
    first_value = 727 if np.random.random() < stats['p_727'] else 728
    synthetic_values.append(first_value)
    
    # Generate remaining values using transition probabilities
    for _ in range(1, num_samples):
        current_val = synthetic_values[-1]
        
        if current_val == 727:
            # Transition from 727
            next_val = 727 if np.random.random() < p_trans['727_727'] else 728
        else:
            # Transition from 728
            next_val = 728 if np.random.random() < p_trans['728_728'] else 727
        
        synthetic_values.append(next_val)
    
    # Convert to numpy array
    synthetic_values = np.array(synthetic_values)
    
    # Generate sample numbers (unique for each dataset)
    start_N = stats['last_N'] + 100000 * dataset_id
    N_values = list(range(start_N, start_N + num_samples))
    
    return pd.DataFrame({
        'N': N_values,
        'D': synthetic_values
    })

def create_segmented_variation(df, num_segments=5):
    """
    Create a segmented variation of the data by cutting and reassembling
    
    Args:
        df: Continuous DataFrame
        num_segments: Number of segments to create
        
    Returns:
        DataFrame with segmented variation
    """
    total_samples = len(df)
    
    # Generate random segment boundaries
    min_segment_size = max(50, total_samples // (num_segments * 2))
    boundaries = [0]
    
    for i in range(num_segments - 1):
        remaining = total_samples - boundaries[-1]
        segments_left = num_segments - i
        max_boundary = total_samples - min_segment_size * segments_left
        
        if boundaries[-1] >= max_boundary:
            boundary = boundaries[-1] + min_segment_size
        else:
            # Random boundary within valid range
            boundary = np.random.randint(
                boundaries[-1] + min_segment_size,
                max_boundary + 1
            )
        boundaries.append(boundary)
    
    boundaries.append(total_samples)
    
    # Extract segments
    segments = []
    for i in range(len(boundaries) - 1):
        segment = df.iloc[boundaries[i]:boundaries[i+1]].copy()
        segments.append(segment)
    
    # Shuffle segments to create variation
    np.random.shuffle(segments)
    
    # Reassemble
    segmented_df = pd.concat(segments, ignore_index=True)
    
    # Reset N values to be sequential
    start_N = segmented_df['N'].iloc[0]
    segmented_df['N'] = range(start_N, start_N + len(segmented_df))
    
    return segmented_df, boundaries

def save_in_original_format(df, filename, sample_freq=1000):
    """
    Save data in the original file format
    
    Args:
        df: DataFrame to save
        filename: Output filename
        sample_freq: Sampling frequency
    """
    with open(filename, 'w') as f:
        f.write(f"Sample Frequency = {sample_freq} Hz\n\n")
        for _, row in df.iterrows():
            f.write(f"D:{int(row['D'])} N:{int(row['N'])}\n")

def generate_augmented_datasets(num_datasets=10, samples_per_dataset=2000):
    """
    Generate multiple augmented datasets for training
    
    Args:
        num_datasets: Number of datasets to generate
        samples_per_dataset: Samples per dataset
        
    Returns:
        summary: DataFrame with dataset statistics
    """
    print("=" * 70)
    print(f"GENERATING {num_datasets} AUGMENTED DATASETS")
    print("=" * 70)
    
    # Read and analyze real data
    real_df, stats = read_and_analyze_real_data("/Users/ilminurablikim/Desktop/Run1.txt")
    
    # Prepare summary data
    summary_data = []
    all_datasets = []
    
    for i in range(1, num_datasets + 1):
        print(f"\n{'='*40}")
        print(f"PROCESSING DATASET #{i}")
        print('='*40)
        
        # 1. Generate continuous synthetic data
        continuous_df = generate_continuous_dataset(stats, i, samples_per_dataset)
        
        # 2. Create segmented variation (different augmentation)
        segmented_df, boundaries = create_segmented_variation(continuous_df, num_segments=5)
        
        # 3. Save both versions (both are valid synthetic data)
        save_in_original_format(
            continuous_df, 
            f"augmented_data/continuous_dataset_{i:03d}.txt", 
            stats['sample_freq']
        )
        
        save_in_original_format(
            segmented_df, 
            f"augmented_data/segmented_dataset_{i:03d}.txt", 
            stats['sample_freq']
        )
        
        # 4. Record statistics
        summary_data.append({
            'Dataset_ID': i,
            'Dataset_Type': 'Continuous',
            'Filename': f'continuous_dataset_{i:03d}.txt',
            'Samples': len(continuous_df),
            'Count_727': (continuous_df['D'] == 727).sum(),
            'Count_728': (continuous_df['D'] == 728).sum(),
            'Percent_727': f"{(continuous_df['D'] == 727).sum()/len(continuous_df)*100:.2f}%",
            'Segments': 'N/A',
            'Description': 'Continuous Markov-generated data'
        })
        
        summary_data.append({
            'Dataset_ID': i,
            'Dataset_Type': 'Segmented',
            'Filename': f'segmented_dataset_{i:03d}.txt',
            'Samples': len(segmented_df),
            'Count_727': (segmented_df['D'] == 727).sum(),
            'Count_728': (segmented_df['D'] == 728).sum(),
            'Percent_727': f"{(segmented_df['D'] == 727).sum()/len(segmented_df)*100:.2f}%",
            'Segments': len(boundaries) - 1,
            'Description': 'Segmented variation of continuous data'
        })
        
        # Store for combined export
        continuous_df['Dataset_ID'] = i
        continuous_df['Dataset_Type'] = 'Continuous'
        segmented_df['Dataset_ID'] = i
        segmented_df['Dataset_Type'] = 'Segmented'
        
        all_datasets.append(continuous_df)
        all_datasets.append(segmented_df)
        
        print(f"✓ Generated: continuous_dataset_{i:03d}.txt")
        print(f"✓ Generated: segmented_dataset_{i:03d}.txt")
        print(f"  Segment boundaries: {boundaries}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Combine all datasets
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    # Save to files
    summary_df.to_excel("augmented_data/dataset_summary.xlsx", index=False)
    combined_df.to_excel("augmented_data/all_augmented_data.xlsx", index=False)
    
    # Save real data for comparison
    save_in_original_format(real_df, "augmented_data/real_data_reference.txt", stats['sample_freq'])
    real_df.to_excel("augmented_data/real_data_reference.xlsx", index=False)
    
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE!")
    print("=" * 70)
    
    # Print final statistics
    print(f"\nGenerated {num_datasets * 2} synthetic files:")
    print(f"  - {num_datasets} continuous datasets")
    print(f"  - {num_datasets} segmented datasets")
    print(f"\nTotal synthetic samples: {num_datasets * 2 * samples_per_dataset:,}")
    print(f"Location: ./augmented_data/")
    
    print(f"\nKey statistics:")
    print(f"  Average 727 percentage: {summary_df['Percent_727'].str.rstrip('%').astype(float).mean():.2f}%")
    print(f"  Target (from real data): {stats['p_727']*100:.2f}%")
    
    return summary_df, combined_df

def print_dataset_examples():
    """Print examples from generated datasets"""
    print("\n" + "=" * 70)
    print("DATASET EXAMPLES")
    print("=" * 70)
    
    # Read first few lines from example files
    for dataset_type in ['continuous', 'segmented']:
        filename = f"augmented_data/{dataset_type}_dataset_001.txt"
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()[:8]  # First 8 lines
                
            print(f"\n{dataset_type.capitalize()} dataset example ({filename}):")
            for line in lines:
                print(f"  {line.strip()}")
        except FileNotFoundError:
            print(f"\nFile not found: {filename}")

# Main execution
if __name__ == "__main__":
    # Generate augmented datasets
    # Parameters:
    #   num_datasets: Number of dataset pairs to generate
    #   samples_per_dataset: Samples in each dataset
    
    print("Piezoelectric Sensor Data Augmentation Generator")
    print("-" * 50)
    print("This script generates synthetic piezoelectric sensor data")
    print("for data augmentation in machine learning projects.")
    print("-" * 50)
    
    summary, combined = generate_augmented_datasets(
        num_datasets=5,           # Generate 5 datasets
        samples_per_dataset=2000  # 2000 samples per dataset
    )
    
    # Show examples
    print_dataset_examples()
    
    # Show summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(summary[['Dataset_ID', 'Dataset_Type', 'Samples', 'Percent_727', 'Segments']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print("In ./augmented_data/ folder:")
    print("  continuous_dataset_XXX.txt   - Continuous synthetic data")
    print("  segmented_dataset_XXX.txt    - Segmented variation")
    print("  dataset_summary.xlsx         - Statistics of all datasets")
    print("  all_augmented_data.xlsx      - All data in one file")
    print("  real_data_reference.txt      - Original real data")
    print("  real_data_reference.xlsx     - Original real data (Excel)")