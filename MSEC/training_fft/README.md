# FFT-Based Classification Pipeline

This pipeline improves waveform classification accuracy by using FFT spectrum plots instead of raw time-domain displacement graphs. It follows a two-stage LLM approach: finetuned LLaVA describes FFT plot features, then LLaMA classifies the waveform type.

## Pipeline Overview

```
Raw CSVs → FFT Plots → Train/Test Split → Gemini Training Data → LLaVA Finetune → LLaVA Eval → LLaMA Classify
```

Classification targets: **sine wave**, **square wave**, **random noise**

---

## Step 1: Generate FFT Plots

**Script**: `../data/FFT/fft_augmented_nm/generate_fft_augmented_nm.py`

Generates FFT spectrum plots from raw CSV displacement data with data augmentation.

**Key features:**
- Y-axis: Linear amplitude in nanometers (not dB) for clearer peak vs noise distinction
- Adaptive x-axis range based on dominant peak frequency (5x peak freq)
- 3 segment offsets per CSV (start indices: 100, 5000, 10000) for augmentation, each a 1024-point FFT window
- Generic plot titles (no wave type label) so the model learns from visual features, not text
- Filters to input signals <= 500Hz only

**Usage:**
```bash
# Generate for each wave type
python generate_fft_augmented_nm.py \
    --csv-dir /path/to/csv/output_sine_csv \
    --output-dir /path/to/output/sine \
    --wave-type sine

python generate_fft_augmented_nm.py \
    --csv-dir /path/to/csv/output_square_csv \
    --output-dir /path/to/output/square \
    --wave-type square

python generate_fft_augmented_nm.py \
    --csv-dir /path/to/csv/output_noise_csv \
    --output-dir /path/to/output/noise \
    --wave-type noise
```

**Input**: CSV files from `../data/output_file/` (processed displacement data in nanometers)

**Output**: `../data/FFT/fft_augmented_nm/{noise,sine,square}/` containing PNG plots

> There is also a dB version at `../data/FFT/fft_augmented_db/` using `generate_fft_augmented.py`, but the linear amplitude (nm) version was chosen for better peak distinction.

---

## Step 2: Split into Train/Test Sets

**Script**: `split_data.py`

Creates a stratified 80/20 train/test split.

**Key features:**
- Groups by CSV source file so all 3 segments from the same CSV stay together (prevents data leakage)
- Stratifies by frequency so each frequency is proportionally represented
- Excludes 500Hz sine/square (indistinguishable at Nyquist frequency)
- Deterministic split (seed=42)

**Usage:**
```bash
python split_data.py \
    --image-dir ../data/FFT/fft_augmented_nm \
    --output-dir .
```

**Output**: `train/{noise,sine,square}/` and `test/{noise,sine,square}/`

**Current counts:**

| Category | Train | Test |
|----------|-------|------|
| Noise    | 102   | 24   |
| Sine     | 165   | 45   |
| Square   | 165   | 45   |
| **Total**| **432** | **114** |

---

## Step 3: Generate Training Data with Gemini

**Script**: `gemini_train.py`

Uses Gemini 2.0 Flash to analyze each FFT plot and generate question-answer pairs for LLaVA training.

**Three questions per image (designed for peak-based classification):**

| # | Question | Classification purpose |
|---|----------|----------------------|
| Q1 | What is the approximate amplitude (in nm) of the tallest peak? Is it above or below 60 nm? | Noise detection: peak < 60nm = noise |
| Q2 | What are the approximate amplitudes (in nm) of the two tallest peaks, and at what frequencies? | Sine vs square: second/tallest ratio < 30% = sine, >= 30% = square |
| Q3 | Does the signal drop sharply or decay gradually after the dominant peak? | Disambiguates 1Hz signals: sharp = sine, gradual = square |

**Key features:**
- Resumable: saves progress every 30 images, skips already-processed entries
- Rate limiting with exponential backoff for Gemini API
- Excludes 500Hz files

**Usage:**
```bash
export GOOGLE_API_KEY='your_key'
python gemini_train.py \
    --image-dir ../data/FFT/fft_augmented_nm \
    --output trainingData_FFT.json
```

**Output**: `trainingData_FFT.json` - JSON array with LLaVA conversation format

**JSON format:**
```json
{
  "id": "sine_sine_100Hz_100Hz_10_absolute_seg1",
  "image": "train/sine/sine_100Hz_100Hz_10_absolute_seg1_FFT.png",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the approximate amplitude..."},
    {"from": "gpt", "value": "The tallest peak is approximately..."},
    {"from": "human", "value": "What are the approximate amplitudes..."},
    {"from": "gpt", "value": "The two tallest peaks are..."},
    {"from": "human", "value": "Looking at the region around..."},
    {"from": "gpt", "value": "The signal drops sharply..."}
  ]
}
```

---

## Step 4: Finetune LLaVA

**Script**: `finetune_fft.sh`
**SLURM**: `train_fft.sbatch`

Finetunes LLaVA v1.5-7B using LoRA on the FFT training data.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `models_setup/llava-v1.5-7b` |
| LoRA rank | 128 |
| LoRA alpha | 256 |
| DeepSpeed | ZeRO Stage 2 (`scripts/zero2.json`) |
| Epochs | 3 |
| Batch size | 2 (per device) |
| Gradient accumulation | 8 steps |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup ratio | 0.03 |
| Precision | bf16 |

**Usage:**
```bash
# Direct run (on GPU node)
bash finetune_fft.sh

# Via SLURM
sbatch train_fft.sbatch
```

**Training results:**
- 81 total steps (27 steps/epoch)
- Loss: 0.75 (start) -> 0.19 (final)
- Runtime: ~9 minutes on H100-80GB

**Output**: `checkpoints_FFT/` with checkpoints at epoch 1 (step 27), epoch 2 (step 54), epoch 3 (step 81)

---

## Step 5: Evaluate with LLaVA

**Script**: `evaluateLLaVA_FFT.py`
**SLURM**: `eval_fft.sbatch`

Runs the finetuned LLaVA model on test images, asking the same 3 questions used during training.

**Usage:**
```bash
# Direct run
python evaluateLLaVA_FFT.py \
    --model-path MSEC/training_fft/checkpoints_FFT \
    --model-base models_setup/llava-v1.5-7b \
    --image-folder MSEC/training_fft/test/noise \
    --output-file results/noise.json

# Via SLURM (runs all 3 categories)
sbatch eval_fft.sbatch
```

**Conda environment**: `llava` (needs the LLaVA model code)

**Output**: `results/{noise,sine,square}.json` - each contains an array of image entries with 3 Q/A pairs

**Output format:**
```json
{
  "image": "noise_10_Run2_1_absolute_seg1_FFT.png",
  "conversation": [
    {"question": "What is the approximate amplitude...", "answer": "The tallest peak..."},
    {"question": "What are the approximate amplitudes...", "answer": "The two tallest peaks..."},
    {"question": "Looking at the region around...", "answer": "The signal drops sharply..."}
  ]
}
```

---

## Step 6: Classify with LLaMA

**Script**: `categorizeLLAMA_FFT.py`
**SLURM**: `classify_fft.sbatch`

Uses LLaMA 3.2 3B Instruct to read LLaVA's answers and classify each waveform.

**Classification rules (in system prompt):**

| Rule | Condition | Classification |
|------|-----------|---------------|
| 1 | Tallest peak < 60nm | A) Random noise |
| 2 | Peak above 60nm, second/tallest ratio < 30% | B) Sine wave |
| 3 | Peak above 60nm, second/tallest ratio >= 30% | C) Square wave |
| 4 | Gradual decay across wide frequency range | C) Square wave |

**Usage:**
```bash
# Direct run
python categorizeLLAMA_FFT.py \
    --model-path models_setup/Llama-3.2-3B-Instruct \
    --conversation-file results/noise.json \
    --output-file results/noise_classified.json

# Via SLURM (runs all 3 categories and prints overall accuracy)
sbatch classify_fft.sbatch
```

**Conda environment**: `llava_infer` (requires transformers >= 4.45.0 for LLaMA 3.2 chat template support; the `llava` env has an older version that causes compatibility errors)

**Output**: `results/{noise,sine,square}_classified.json` with accuracy metrics

**Output format:**
```json
{
  "accuracy": 95.0,
  "total": 20,
  "correct": 19,
  "results": [
    {
      "image": "noise_10_Run2_1_absolute_seg1_FFT.png",
      "gt": "noise",
      "pred": "noise",
      "is_correct": true,
      "reasoning": "...Result: A"
    }
  ]
}
```

---

## Environment Setup

### `llava` env (training + LLaVA evaluation)
```bash
conda create -n llava python=3.10
conda activate llava
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install deepspeed
```

### `llava_infer` env (LLaMA classification)
```bash
conda create -n llava_infer python=3.10
conda activate llava_infer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45.0" accelerate
```

> The separate `llava_infer` env is needed because LLaVA's training code pins an older transformers version that doesn't support LLaMA 3.2's chat template format.

---

## SLURM Quick Reference

All jobs use the `GPU-shared` partition with 1x H100-80GB on Bridges-2.

```bash
# Run training (~9 min)
sbatch train_fft.sbatch

# Run LLaVA evaluation (~30 min)
sbatch eval_fft.sbatch

# Run LLaMA classification (~15 min)
sbatch classify_fft.sbatch

# Check job status
squeue -u $USER

# View output/errors
cat classify_fft_<jobid>.out
cat classify_fft_<jobid>.err
```

---

## File Structure

```
training_fft/
├── README.md                  # This file
├── split_data.py              # Step 2: Train/test split
├── gemini_train.py            # Step 3: Generate training data with Gemini
├── trainingData_FFT.json      # Training data (Gemini output)
├── finetune_fft.sh            # Step 4: LLaVA finetuning script
├── train_fft.sbatch           # SLURM wrapper for training
├── evaluateLLaVA_FFT.py       # Step 5: LLaVA evaluation
├── eval_fft.sbatch            # SLURM wrapper for evaluation
├── categorizeLLAMA_FFT.py     # Step 6: LLaMA classification
├── classify_fft.sbatch        # SLURM wrapper for classification
├── checkpoints_FFT/           # Finetuned model checkpoints
│   ├── checkpoint-27/         # Epoch 1
│   ├── checkpoint-54/         # Epoch 2
│   └── checkpoint-81/         # Epoch 3 (final)
├── train/{noise,sine,square}/ # Training images
├── test/{noise,sine,square}/  # Test images
└── results/                   # Evaluation and classification results
    ├── noise.json             # LLaVA eval output
    ├── sine.json
    ├── square.json
    ├── noise_classified.json  # LLaMA classification output
    ├── sine_classified.json
    └── square_classified.json
```

FFT plot generation script lives at:
```
../data/FFT/fft_augmented_nm/generate_fft_augmented_nm.py
```
