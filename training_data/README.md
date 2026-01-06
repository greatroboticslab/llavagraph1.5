# 📈 Training Data Pipeline

## Dataset Structure

```Shell

training_data/
├── data/                         # All dataset files
│   ├── raw/                      # Original input data (read-only)
│   │   ├── input/                # Original instrument panel images (141 images)
│   │   │   ├── RandomNoise/      # 52 images with random noise signals
│   │   │   ├── SineWave/         # 36 images with sinusoidal waveforms
│   │   │   └── SquareWave/       # 53 images with square waveforms
│   │   └── output/               # Original target data
│   │
│   └── synthetic/                # Generated synthetic data
│       ├── input/                # Synthetic input images (2,115 images)
│       │   ├── RandomNoise/      # 705 images with random noise overlays
│       │   ├── SineWave/         # 705 images with sine wave overlays
│       │   └── SquareWave/       # 705 images with square wave overlays
│       └── output/               # Processed synthetic outputs
│           ├── V1/               # First version 
│           └── V2/               # Second version 
│           └── V3/               # Third version (use for training)
│
├── scripts/                      # All processing and training scripts
│   ├── TrainingInputdata_folder.py   # Input Synthetic image generation
│   ├── augmented_250.py              # Output Synthetic image generation
│   ├── modified_JSONData.py          # Trainning script
│   └── train_lora.sh                 # LoRA training script
│
├── fullData.json               # Generated training data
└── README.md

```

## Generate synthetic data
### Synthetic Image Generation Description_inputdata
The generation script is located at: scripts/TrainingInputdata_folder.py

To generate synthetic images, run:
```Shell
bash

python TrainingInputdata_folder.py
```

For each original image in your dataset, the script generates three types of waveform variations, each with 5 versions: Random Noise (5 variations), Sine Wave (5 variations), and Square Wave (5 variations), resulting in a total of 15 synthetic images per original image. 
Each generated synthetic image is composed of the following elements: the original instrument panel as the background, a new waveform overlay positioned at coordinates (80, 200), waveform lines in purple (color code #b43ed1), and follows a clear file naming convention: {source}_{original_filename}_{type}{number}.png.

### Synthetic Image Generation Description_outputdata(V3)

#### 📊 Data prep
Before generating augmented data, raw collected data must be converted to physical units (nanometers) using the process_raw.py script.

Setup Environment:
```bash

conda activate base
python -m pip install ttkbootstrap matplotlib

```

Run Data Processing (training_data/data/synthetic/output/V2/process_raw.py):
```bash

python process_raw.py

```

#### 📊Data Augmentation
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

Update the configuration section at the top of scripts/augmented_250.py:
```Shell
OUTPUT_DIR = "augmented_first250_output"   # or any directory you prefer
POINTS_TO_USE = 250                       # number of points per series
BASE_DIR = "/path/to/your/csv_directory"  # folder containing input CSV files

file_patterns = [
    os.path.join(BASE_DIR, "*.csv"),
    # add more patterns if needed
]
```

Then run:
```Shell
python augmented_250.py
```
The script will:
Scan all CSV files matching file_patterns
For each file, use the first POINTS_TO_USE points of columns Time_ms and Delta_Displacement_nm
Generate augmented variants and save all plots as PNG files


## 🎯 Quick Start
## Environment Setup

### 1. Virtual Environment Creation
```bash
# Create virtual environment
python -m venv llava # Linux/Mac
# Windows: venv\Scripts\activate

# Activate environment
source llava/bin/activate

# Verify activation (prompt should show (llava))
which python
```

### 2. PyTorch Installation
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Expected output:**
```
2.5.1+cu121
True
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

---

## Project Installation

### 1. Install LLaVAGraph Package
```bash
# Verify project files exist
ls pyproject.toml  # Should exist ✓

# Install project in editable mode
pip install -e .

# Install with training dependencies
pip install -e ".[train]"

# Verify installation
python -c "import llava; print('LLaVAGraph installation successful!')"
```

### 2. Optional Recommended Packages
```bash
# DeepSpeed for distributed training (optional)
pip install deepspeed

# Flash Attention for faster training (optional)
pip install flash-attn --no-build-isolation
```

---

## Data Generation

### 1. Generate Training Data

**Important:** Modify paths in `scripts/modified_JSONData.py` according to your directory structure before running.
```bash
# Run data generation pipeline
python data/modified_JSONData.py
```

**Expected Result:**
```
✅ Generated 2257 data entries saved to: /data/fullData.json
📊 Breakdown:
   - NoiseData: 758
   - SineData:  741
   - SquareData: 758
```

---

## Dependency Resolution

### Issue: PEFT Version Conflict

**Problem:** PEFT version incompatibility with Accelerate package (missing `clear_device_cache` function)

### Solution: Install Compatible Package Versions
```bash
# Remove conflicting packages
pip uninstall peft accelerate transformers torch torchvision torchaudio -y

# Install compatible version set
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install protobuf

# Reinstall project
pip install -e .
```

---

## Training Process

### Training Strategy

This approach trains the multimodal projector from scratch (fastest method for initial experiments).

### Training Command
```bash
conda activate llava

CUDA_VISIBLE_DEVICES=0 python llavagraph1.5/llava/train/train_mem.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 --version v1 \
    --data_path data/fullData_fixed.json \
    --image_folder data/stage1_input \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --image_aspect_ratio pad --fp16 True \
    --output_dir checkpoints/lora_llavagraph_no_pretrained_mm \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 --evaluation_strategy no \
    --save_steps 10 --learning_rate 1e-4 \
    --model_max_length 512 --dataloader_num_workers 0 \
    --tune_mm_mlp_adapter True
```
or can just run scipts:
```bash
bash train_lora.sh
```

### Training Configuration Comparison

| Parameter | Stable Configuration | Official Pipeline | Impact |
|-----------|---------------------|-------------------|--------|
| `lora_r` | 8 | 128 | Smaller rank → better generalization |
| `batch_size` | 1 | 16 | More stable, avoids OOM errors |
| `learning_rate` | 1e-4 | 2e-4 | More conservative, stable convergence |
| `max_length` | 512 | 2048 | Faster training, suitable for testing |
| `fp16` | ✅ fp16 | ✅ bf16 | Better hardware compatibility |
| `deepspeed` | ❌ | ✅ zero3 | Zero bugs in simple setup |
| `workers` | 0 | 4 | Stable DataLoader operation |

### Training Output

Checkpoint directory: `checkpoints/lora_llavagraph_simple_final/`

