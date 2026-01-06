# LLaVAGraph Training Data Pipeline

Complete documentation for setting up and training LLaVAGraph with LoRA fine-tuning on custom signal processing data.

## Table of Contents
- Environment Setup
- Data Generation
- Dependency Resolution
- Training Process
- Evaluation

---

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

**Important:** Modify paths in `data/JSONData.py` according to your directory structure before running.
```bash
# Run data generation pipeline
python data/JSONData.py
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

---

## Evaluation

### 1. Setup Evaluation Script
```bash
# Ensure PEFT is installed
pip install peft
```

**Note:** Optimize `evaluate_LLaVA.py` content and create `run_eval.sh` before running.

### 2. Run Evaluation
```bash
# Make script executable
chmod +x run_eval.sh

# Execute evaluation
./run_eval.sh
```

---

## Directory Structure
```
/12.14llava/
├── llava/                          # Virtual environment
├── llavagraph1.5/                  # Project source code
│   ├── llava/
│   │   └── train/
│   │       └── train_mem.py
│   └── pyproject.toml
├── data/
│   ├── fullData.json               # Generated training data 
│   ├── textData                    # used for generate json file
│   └── stage1_input/               # Image data folder
│   └── stage2_output/               # Image data folder
├── checkpoints/
│   └── lora_llavagraph_simple_final/  # Training output
└── scripts/
    ├── evaluate_LLaVA.py           # Evaluation script
    └── run_eval.sh                 # Evaluation runner
    └── modified_JSONData.py           # trainning script
```

---

## Key Configuration Details

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16
- **Target modules**: Query and Value projections

### Training Hyperparameters
- **Base model**: `lmsys/vicuna-7b-v1.5`
- **Vision encoder**: `openai/clip-vit-large-patch14-336`
- **Batch size**: 1 (per device)
- **Learning rate**: 1e-4
- **Max sequence length**: 512 tokens
- **Training epochs**: 1
- **Precision**: FP16

### Data Statistics
**Total samples**: 2,257
- **NoiseData**: 758 samples
- **SineData**: 741 samples  
- **SquareData**: 758 samples


**Total evaluate data**: 2443
- **RandomNoise**: 294 samples
- **SineWave**: 1071 samples  
- **SquareWave**: 1078 samples

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solutions:
# - Reduce per_device_train_batch_size to 1
# - Decrease model_max_length
# - Enable gradient checkpointing
```

#### 2. Package Version Conflicts
```bash
# Solution:
# Follow exact version specifications in Dependency Resolution section
# Always uninstall before reinstalling
```

#### 3. DataLoader Errors
```bash
# Solutions:
# - Set dataloader_num_workers to 0
# - Verify data path correctness
```
