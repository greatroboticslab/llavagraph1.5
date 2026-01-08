# 📈 Training Data Pipeline

## Dataset Structure

```Shell
training_data/
├── data/                         # All organized dataset files
│   ├── original/                  # Original input data 
│   │   ├── input/                # Original instrument panel images (141 images)
│   │   │   ├── RandomNoise/      # 52 images with random noise signals
│   │   │   ├── SineWave/         # 36 images with sinusoidal waveforms
│   │   │   └── SquareWave/       # 53 images with square waveforms
│   │   └── output/               # Original target data
│   │   │   ├── Issac_data_Aug_27_2024/      # All original images(png + txt)
│   │   │   ├── output_file/         # Processed files(nanometer conversion）
│   │   │   ├── extract_txt_file.py      
│   │   │   ├── process_raw_8.py         # script used for processing
│   │   │   └── extract_csv.py
│   │
│   └── synthetic/                # Generated synthetic data
│       ├── input/                # Synthetic input images (2,115 images)
│       │   ├── RandomNoise/      # 705 images with random noise overlays
│       │   ├── SineWave/         # 705 images with sine wave overlays
│       │   └── SquareWave/       # 705 images with square wave overlays
│       └── output/               # Processed synthetic outputs
│           ├── V1/               # First version evaluations
│           ├── V2/               # Second version evaluations
│           └── V3/               # Final version (use for training)
│
├── scripts/                      # All processing and training scripts
│   ├── TrainingInputdata_folder.py   # input synthetic image generation
│   ├── modified_JSONData.py          # JSON annotation processing
│   └── train_lora.sh                 # LoRA training script
│
├── fullData.json                     
└── README.md

```

## Synthetic Image Generation Description_inputdata
The generation script is located at: scripts/TrainingInputdata_folder.py

To generate synthetic images, run:
```Shell
bash

python TrainingInputdata_folder.py
```

For each original image in your dataset, the script generates three types of waveform variations, each with 5 versions: Random Noise (5 variations), Sine Wave (5 variations), and Square Wave (5 variations), resulting in a total of 15 synthetic images per original image. 
Each generated synthetic image is composed of the following elements: the original instrument panel as the background, a new waveform overlay positioned at coordinates (80, 200), waveform lines in purple (color code #b43ed1), and follows a clear file naming convention: {source}_{original_filename}_{type}{number}.png.


## Synthetic Image Generation Description_outputdata
## Time Series Augmentation

### V3 - synthetic data(final version)

### 📊 Data prep
Before generating augmented data, raw collected data must be converted to physical units (nanometers) using the process_raw_8.py script.

#### Text File Extraction Process:
Run extract_txt_file.py (located in the original/output/ directory) to extract the original text files from the Issac_data_Aug_27_2024/ folder. These extracted text files serve as data for the subsequent processing pipeline.

Setup Environment:
```bash

conda activate base
python -m pip install ttkbootstrap matplotlib

```

Run Data Processing:
```bash

python process_raw_8.py

``` 

<img width="845" height="244" alt="Screenshot 2026-01-07 at 9 33 16 PM" src="https://github.com/user-attachments/assets/0db6427c-f42f-4f20-8dda-2cfb43f38cf3" />

This workflow sequentially imports three files from the path: original/output/Issac_data_Aug_27_2024/Original_only_txt. When "Absolute" mode is selected, the output original/output/output_file is generated.

For detailed implementation of the conversion code in process_raw_8.py, please refer to:
https://github.com/greatroboticslab/laserai

Note: Modified the conversion algorithm in process_raw_8.py to implement new formula:
```Shell
nm = (D - baseline) * (wavelength / 8) - correction
``` 
Since the processed output folder contains both PNG and CSV files, the extract_csv.py script located in original/output/ can be used to extract CSV files specifically for subsequent synthetic data generation.


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


Run the augmentation script in the V3 folder:
```Shell
python Augmented_final.py
```

Input: Generated CSV Files
To configure the pipeline, modify these three parameters at the bottom of the code:

```Shell
CSV_INPUT_PATH = "path/to/your/csv/*.csv"  # Path pattern for CSV files
OUTPUT_FOLDER_NAME = "OutputWave"           # Output folder name
POINTS_TO_USE = 250                         # Number of data points to use
```
The pipeline will generate 1 original visualization and 6 synthetic augmentations from each CSV file for further analysis or model training.




