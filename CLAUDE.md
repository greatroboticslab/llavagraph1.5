# LLaVAGraph 1.5 - MSEC FFT Pipeline

## Project Overview
LLaVAGraph is a multimodal framework for classifying piezoelectric actuator displacement signals using finetuned LLaVA + a classifier LLM. The new work in `MSEC/` focuses on an **FFT-based approach** to improve classification accuracy over the original time-domain plots.

## Current Status: 91.2% overall accuracy (3 classes: noise, sine, square)

## Architecture (Two-Stage LLM Pipeline)
1. **LLaVA** (finetuned on FFT plots) looks at FFT spectrum images and describes peak features
2. **Qwen2.5-14B-Instruct** reads those descriptions and classifies via a decision tree prompt

## FFT Pipeline (`MSEC/training_fft/`)

### Pipeline Steps (in order)
1. **FFT Plot Generation** (`MSEC/data/FFT/fft_augmented_nm/generate_fft_augmented_nm.py`)
   - Generates FFT spectrum plots from raw CSV displacement data
   - Linear amplitude (nm) y-axis, adaptive x-axis (5x dominant peak)
   - 3 segment offsets per CSV for data augmentation (1024-point windows)
   - Filters to signals <= 500Hz only
   - Output: `MSEC/data/FFT/fft_augmented_nm/{noise,sine,square}/`

2. **Train/Test Split** (`MSEC/training_fft/split_data.py`)
   - 80/20 split, stratified by frequency
   - Groups by CSV source so all 3 segments stay together
   - Excludes 500Hz sine/square (indistinguishable at Nyquist)
   - Output: `MSEC/training_fft/train/` and `test/` with {noise,sine,square} subfolders
   - Current counts: Train (102 noise, 165 sine, 165 square) | Test (24 noise, 45 sine, 45 square)

3. **Training Data Generation** (`MSEC/training_fft/gemini_train.py`)
   - Uses Gemini 2.0 Flash to generate Q/A pairs for each FFT plot
   - 3 questions per image focused on peak-based features:
     - Q1: Tallest peak amplitude, above/below 60nm threshold
     - Q2: Two tallest peaks' amplitudes and frequencies (ratio distinguishes sine vs square)
     - Q3: Decay shape after dominant peak (sharp=sine, gradual=square)
   - Resumable (saves progress every 30 images)
   - Output: `MSEC/training_fft/trainingData_FFT.json`

4. **LLaVA Finetuning** (`MSEC/training_fft/finetune_fft.sh`, SLURM: `train_fft.sbatch`)
   - Base model: `models_setup/llava-v1.5-7b`
   - LoRA: r=128, alpha=256
   - DeepSpeed ZeRO-2 (`scripts/zero2.json`)
   - 3 epochs, batch=2, grad_accum=8, lr=2e-4, cosine schedule
   - Training completed: 81 steps, final loss ~0.19 (down from 0.75)
   - Checkpoints: `MSEC/training_fft/checkpoints_FFT/` (checkpoint-27, 54, 81)

5. **LLaVA Evaluation** (`MSEC/training_fft/evaluateLLaVA_FFT.py`, SLURM: `eval_fft.sbatch`)
   - Runs finetuned LLaVA on test images, asks same 3 questions
   - Output: `MSEC/training_fft/results/{noise,sine,square}.json`

6. **LLaMA/Qwen Classification** (`MSEC/training_fft/categorizeLLAMA_FFT.py`, SLURM: `classify_fft.sbatch`)
   - Current model: **Qwen2.5-14B-Instruct** (`models_setup/Qwen2.5-14B-Instruct`)
   - Decision tree prompt with 3 steps:
     - STEP 1: Peak amplitude < 60nm → noise (A)
     - STEP 2: Gradual decay across wide frequency range → square (C)
     - STEP 3: Ratio (second_peak / tallest_peak) < 0.30 → sine (B), >= 0.30 → square (C)
   - Output: `MSEC/training_fft/results/{noise,sine,square}_classified.json`

### Classification Results (Final: 91.2% overall)
| Category | Accuracy |
|----------|----------|
| Noise    | 24/24 = 100% |
| Sine     | 36/45 = 80% |
| Square   | 44/45 = 97.8% |
| **Overall** | **104/114 = 91.2%** |

### Model Evolution for Stage 2 Classifier
- **LLaMA 3.2 3B**: Failed — couldn't follow the decision tree prompt, produced freeform text
- **Qwen2.5-7B-Instruct**: Better instruction following, but consistently misclassified all sine as square (60.5%). The 7B model couldn't execute "skip this step" branching logic — it always output Result: C at STEP 2 regardless of the condition.
- **Qwen2.5-14B-Instruct**: Successfully follows the decision tree, achieves 91.2% overall

### Key Learnings
- Prompt restructuring alone couldn't fix the 7B model's sine misclassification — it was a model capacity issue, not a prompt issue
- The output format section at the end of prompts can confuse smaller models into evaluating all criteria even when they should stop early
- The decay shape check (STEP 2) is necessary for 1Hz signals where the ratio alone isn't sufficient
- LLaMA 3.2 8B on HuggingFace is gated and requires manual approval

## Original Pipeline (for reference, `MSEC/` root level)
- Time-domain plots with augmentation (time warp, reverse, pool)
- `gemini_train.py` -> `trainingData_Gemini.json`
- `finetune2.2.sh` -> training
- `evaluateLLaVA.sh` / `evaluateLLaVA.py` -> captions in `llava_V6/`
- `categorizeLLAMA.sh` / `categorizeLLAMA_V6.py` -> classifications in `llama_V6/`

## Environment
- HPC: PSC Bridges-2, SLURM scheduler, GPU-shared partition (V100-32GB for inference)
- Conda envs: `llava` (training, finetuning), `llava_infer` (inference with Qwen)
- Account: `cis240145p`
- Base LLaVA repo at: `/ocean/projects/cis240145p/byler/LLaVA/`
- Project dir: `/ocean/projects/cis240145p/byler/ben/llavagraph1.5/`

## Key Commands
```bash
# Submit training job
sbatch MSEC/training_fft/train_fft.sbatch

# Submit evaluation job
sbatch MSEC/training_fft/eval_fft.sbatch

# Submit classification job
sbatch MSEC/training_fft/classify_fft.sbatch
```
