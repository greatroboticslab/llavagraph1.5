# LLaVAGraph 1.5 - MSEC FFT Pipeline

## Project Overview
LLaVAGraph is a multimodal framework for classifying piezoelectric actuator displacement signals (sine, square, noise) using finetuned LLaVA + LLaMA. The new work in `MSEC/` focuses on an **FFT-based approach** to improve classification accuracy over the original time-domain plots.

## Goal
Optimize the pipeline to improve finetuned model accuracy by using FFT spectrum plots instead of raw displacement graphs.

## Architecture (Two-Stage LLM Pipeline)
1. **LLaVA** (finetuned) looks at FFT plot images and describes peak features
2. **LLaMA 3.2 3B** reads those descriptions and classifies as noise/sine/square

## New Work: FFT Pipeline (`MSEC/training_fft/`)

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
   - Evaluation completed successfully

6. **LLaMA Classification** (`MSEC/training_fft/categorizeLLAMA_FFT.py`, SLURM: `classify_fft.sbatch`)
   - Uses LLaMA 3.2 3B Instruct (`models_setup/Llama-3.2-3B-Instruct`)
   - Rule-based prompt: peak < 60nm = noise, ratio < 30% = sine, >= 30% = square
   - Output: `MSEC/training_fft/results/{noise,sine,square}_classified.json`

## Current Status
- FFT plot generation: DONE
- Train/test split: DONE
- Training data (Gemini): DONE
- LLaVA finetuning: DONE (3 epochs, loss 0.75 -> 0.19)
- LLaVA evaluation on test set: DONE (results saved)
- LLaMA classification: FAILING - `TypeError: can only concatenate str (not "dict") to str`
  - The `transformers` version in the `llava` conda env is too old for chat-template pipeline with message dicts
  - Needs either: upgrade transformers, or use `tokenizer.apply_chat_template()` manually

## Original Pipeline (for reference, `MSEC/` root level)
- Time-domain plots with augmentation (time warp, reverse, pool)
- `gemini_train.py` -> `trainingData_Gemini.json`
- `finetune2.2.sh` -> training
- `evaluateLLaVA.sh` / `evaluateLLaVA.py` -> captions in `llava_V6/`
- `categorizeLLAMA.sh` / `categorizeLLAMA_V6.py` -> classifications in `llama_V6/`

## Environment
- HPC: PSC Bridges-2, SLURM scheduler, GPU partition (H100-80GB)
- Conda env: `llava` (Python 3.10, PyTorch + CUDA, DeepSpeed)
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
