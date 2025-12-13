# Time Series Augmentation

## V1 
1. Data Analysis (read_and_analyze_real_data)
Reads real piezoelectric sensor data from a text file (Run1.txt)
Parses data in format: D:727 N:1787216
Extracts statistical properties:
Frequency of values 727 and 728 (the two observed states)
Markov transition probabilities between these states
Sampling frequency (default 1000 Hz)

2. Synthetic Data Generation (generate_continuous_dataset)
Uses a Markov chain model to generate realistic sequences
First value based on overall probability distribution
Subsequent values based on transition probabilities (e.g., probability of 727→728)
Maintains realistic temporal patterns from the real data

3. Data Augmentation Variations (create_segmented_variation)
Creates segmented versions by:
Cutting continuous data into random segments
Shuffling segments
Reassembling them
This creates additional variation while maintaining statistical properties

4. Output Management
Saves data in the original format for compatibility
Creates two types of synthetic datasets per run:
Continuous: Smooth Markov-generated sequences
Segmented: Shuffled-segment variations
Generates comprehensive documentation:
Excel files with statistics
Text files in original format
Summary reports


## V2 - synthetic data

### 📊 Data prep
Before generating augmented data, raw collected data must be converted to physical units (nanometers) using the process_raw.py script.

Setup Environment:
```bash

conda activate base
python -m pip install ttkbootstrap matplotlib

```

Run Data Processing:
```bash

python process_raw.py

```

### 📊Data Augmentation
This pipeline uses the tsaug package for time series augmentation. Three augmentation methods are implemented:
1. Time Warp
Modifies temporal dynamics by applying speed changes:
Mild: max_speed_ratio=1.2
Moderate: max_speed_ratio=1.5
Strong: max_speed_ratio=1.8
2. Reverse
Reverses the time series sequence (temporal flipping).
3. Pool
Reduces data resolution through downsampling:
Size 2: Downsample by factor of 2
Size 3: Downsample by factor of 3

Installation
Prerequisites: Python 3.5 or later.
```Shell
pip install tsaug
```
References: 
https://github.com/arundo/tsaug
https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html

Run the augmentation script in the V2 folder:
```Shell
python ts_augmentation.py
```
This generates 6 augmented datasets:
Time Warp Augmentation (3 intensity levels)
Reverse Augmentation (temporal reversal)
Pool Augmentation (2 downsampling levels)

### Output Structure
The script creates an augmented_data/ directory containing:
```Shell
augmented_data/
├── original.png                    # Original data plot
├── original.csv                    # Original data (CSV format)
├── timewarp_mild.png              # Mild time warp plot
├── timewarp_mild.csv              # Mild time warp data (CSV)
├── timewarp_moderate.png          # Moderate time warp plot
├── timewarp_moderate.csv          # Moderate time warp data (CSV)
├── timewarp_strong.png            # Strong time warp plot
├── timewarp_strong.csv            # Strong time warp data (CSV)
├── reverse.png                    # Reverse augmentation plot
├── reverse.csv                    # Reverse augmentation data (CSV)
├── pool_size2.png                 # Pool size 2 plot
├── pool_size2.csv                 # Pool size 2 data (CSV)
├── pool_size3.png                 # Pool size 3 plot
└── pool_size3.csv                 # Pool size 3 data (CSV)
```
Each augmentation produces both a visualization (PNG) and the corresponding data file (CSV) for further analysis or model training.


