# LLaVAGraph 1.5 - MSEC FFT Pipeline

## Project Overview
LLaVAGraph is a multimodal framework for classifying piezoelectric actuator displacement signals using finetuned LLaVA + a classifier LLM. The new work in `MSEC/` focuses on an **FFT-based approach** to improve classification accuracy over the original time-domain plots.

## Current Status
- **3-class (completed)**: 91.2% overall (noise, sine, square)
- **5-class (in progress)**: noise, sine, square, pulse, ramp — LLaVA training submitted (job 37567935)

## Architecture (Two-Stage LLM Pipeline)
1. **LLaVA** (finetuned on FFT plots) looks at FFT spectrum images and describes peak features
2. **Qwen2.5-14B-Instruct** reads those descriptions and classifies via a decision tree prompt

---

## 5-Class Pipeline (`MSEC/training_fft/train_5class/`)

### Pipeline Steps (in order)

1. **FFT Plot Generation** (`MSEC/data/FFT/fft_augmented_nm/generate_fft_augmented_nm.py`)
   - Same script as 3-class, updated with adaptive window sizes
   - 1024-point window default, **4096-point for low-freq signals** (ramp trail, 1Hz sine/square)
   - Larger window gives Δf=0.244Hz (vs 0.977Hz) to resolve closely-spaced peaks
   - Output: `MSEC/data/FFT/fft_augmented_nm/{noise,sine,square,pulse,ramp}/`

2. **Train/Test Split** (`MSEC/training_fft/train_5class/split_data_5class.py`)
   - 80/20 split, stratified by frequency (sine/square) or subtype (ramp: trail vs standard)
   - Groups by CSV source so all segments stay together
   - Excludes 500Hz sine/square (indistinguishable at Nyquist)
   - Output: `train_5class/train/` and `train_5class/test/`
   - Counts: **561 train** (102 noise, 165 sine, 165 square, 67 pulse, 62 ramp) | **147 test**

3. **Training Data Generation** (`MSEC/training_fft/train_5class/gemini_train_5class.py`)
   - Uses Gemini 2.0 Flash to generate Q/A pairs for each FFT plot
   - **3 questions per image** (LLaVA only reports observations, no calculations):
     - Q1: Tallest peak amplitude (nm)
     - Q2: Peak frequency (Hz) + decay shape (sharp drop vs gradual)
     - Q3: Three tallest peaks — amplitudes (nm) and frequencies (Hz)
   - Resumable (saves every 30 images), rate-limited (4s sleep)
   - Output: `train_5class/trainingData_FFT_5class.json` (561 entries, validated against plots)

4. **LLaVA Finetuning** (`train_5class/finetune_5class.sh`, SLURM: `train_5class.sbatch`)
   - Base model: `models_setup/llava-v1.5-7b`
   - LoRA: r=128, alpha=256, DeepSpeed ZeRO-2
   - 3 epochs, batch=2, grad_accum=8, lr=2e-4, cosine schedule
   - **Status: training job submitted** (job 37567935, pending in queue)
   - Checkpoints: `train_5class/checkpoints_FFT_5class/`

5. **LLaVA Evaluation** (`train_5class/evaluateLLaVA_5class.py`, SLURM: `eval_5class.sbatch`)
   - Runs finetuned LLaVA on test images, asks same 3 questions
   - Output: `train_5class/results/{noise,sine,square,pulse,ramp}.json`

6. **Qwen Classification** (`train_5class/categorizeLLAMA_5class.py`, SLURM: `classify_5class.sbatch`)
   - Model: **Qwen2.5-14B-Instruct** (`models_setup/Qwen2.5-14B-Instruct`)
   - 5-step decision tree (output format placed before steps to avoid confusing Qwen):
     - STEP 1: Peak amplitude — >250nm → pulse, <60nm → STEP 2, 60-250nm → STEP 3
     - STEP 2: Low amplitude — peak at 1Hz or no peaks → pulse, otherwise → noise
     - STEP 3: Harmonic pattern — sort 3 peaks by frequency, check equal spacing (within 30%)
     - STEP 4: Harmonic branch — 2nd/1st ratio ≤15% → ramp, >15% → square
     - STEP 5: No harmonics — 3rd peak ≥25nm → square, <25nm → sine
   - Output: `train_5class/results/{category}_classified.json`

### Signal Characteristics
- **Noise**: Low amplitude (<60nm), no dominant frequency
- **Sine**: Single dominant peak, spectral leakage creates nearby 2nd peak (not a real harmonic)
- **Square**: Odd harmonics at f, 3f, 5f; aliasing at >100Hz may fold harmonics
- **Pulse**: Aperiodic transient, FFT shows 1/f decay with 1Hz artifact (not real frequency)
- **Ramp trail**: ~1Hz fundamental, needs 4096-pt window; harmonics at f, 2f, 3f
- **Ramp standard**: ~10Hz fundamental, clear harmonics at 3x, 5x

### Known Edge Cases
- ~5 pulse segments fall in 60-250nm range → may misclassify as ramp (accepted trade-off)
- 31/165 sine entries have 2nd peak >15% ratio due to spectral leakage → handled by decision tree using 3-peak data (leakage peaks are close together, not evenly spaced like harmonics)
- Square >100Hz may not show clean harmonic spacing → caught by STEP 5 (3rd peak ≥25nm)

---

## 3-Class Pipeline (`MSEC/training_fft/`) — Completed

### Results: 91.2% overall
| Category | Accuracy |
|----------|----------|
| Noise    | 24/24 = 100% |
| Sine     | 36/45 = 80% |
| Square   | 44/45 = 97.8% |
| **Overall** | **104/114 = 91.2%** |

### Files
- Split: `split_data.py` → `train/` and `test/` (102 noise, 165 sine, 165 square train)
- Training data: `gemini_train.py` → `trainingData_FFT.json`
- Finetuning: `finetune_fft.sh` / `train_fft.sbatch` → `checkpoints_FFT/` (81 steps, loss 0.75→0.19)
- Evaluation: `evaluateLLaVA_FFT.py` / `eval_fft.sbatch` → `results/{noise,sine,square}.json`
- Classification: `categorizeLLAMA_FFT.py` / `classify_fft.sbatch` → `results/{category}_classified.json`

### Model Evolution for Stage 2 Classifier
- **LLaMA 3.2 3B**: Failed — couldn't follow the decision tree prompt
- **Qwen2.5-7B-Instruct**: Couldn't execute branching logic (60.5%)
- **Qwen2.5-14B-Instruct**: Successfully follows the decision tree (91.2%)

### Key Learnings
- Output format section at the end of prompts confuses smaller models → place it before the steps
- Prompt restructuring alone couldn't fix 7B's failures — model capacity issue
- LLaVA should only report visual observations, never compute ratios

---

## Original Pipeline (for reference, `MSEC/` root level)
- Time-domain plots with augmentation (time warp, reverse, pool)
- `gemini_train.py` -> `trainingData_Gemini.json`
- `finetune2.2.sh` -> training
- `evaluateLLaVA.sh` / `evaluateLLaVA.py` -> captions in `llava_V6/`
- `categorizeLLAMA.sh` / `categorizeLLAMA_V6.py` -> classifications in `llama_V6/`

## Environment
- HPC: PSC Bridges-2, SLURM scheduler, GPU-shared partition
- GPUs: L40S-48GB (training), H100-80GB (eval), V100-32GB (classification)
- Conda envs: `llava` (training, finetuning), `llava_infer` (inference with Qwen)
- Account: `cis240145p`
- Base LLaVA repo at: `/ocean/projects/cis240145p/byler/LLaVA/`
- Project dir: `/ocean/projects/cis240145p/byler/ben/llavagraph1.5/`

## Key Commands
```bash
# 5-class pipeline
sbatch MSEC/training_fft/train_5class/train_5class.sbatch      # LLaVA finetuning
sbatch MSEC/training_fft/train_5class/eval_5class.sbatch       # LLaVA evaluation
sbatch MSEC/training_fft/train_5class/classify_5class.sbatch   # Qwen classification

# 3-class pipeline (completed)
sbatch MSEC/training_fft/train_fft.sbatch
sbatch MSEC/training_fft/eval_fft.sbatch
sbatch MSEC/training_fft/classify_fft.sbatch
```
