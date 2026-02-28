# LLaVAGraph 1.5 - MSEC FFT Pipeline

## Project Overview
LLaVAGraph is a multimodal framework for classifying piezoelectric actuator displacement signals using finetuned LLaVA + a classifier LLM. The new work in `MSEC/` focuses on an **FFT-based approach** to improve classification accuracy over the original time-domain plots.

## Current Status
- **3-class LLaVA+Qwen (completed)**: 91.2% overall (noise, sine, square)
- **5-class LLaVA+Qwen (completed)**: 86.4% overall (noise, sine, square, pulse, ramp)
- **5-class ViT FFT — old data only (completed)**: 98.6% overall (289/293)
- **5-class ViT FFT — combined data (completed)**: 95.95% overall (237/247)

## Architecture Comparison

### Option A: Two-Stage LLM Pipeline (LLaVA + Qwen)
1. **LLaVA** (finetuned on FFT plots) looks at FFT spectrum images and describes peak features
2. **Qwen2.5-32B-Instruct** reads those descriptions and classifies via a decision tree prompt

### Option B: ViT Direct Classification
- Fine-tuned `google/vit-base-patch16-224-in21k` directly on FFT images
- Single-stage, no LLM needed
- Significantly higher accuracy, much simpler pipeline

---

## ViT FFT Pipeline (`MSEC/training_VIT/`) — Completed

Direct ViT classification on FFT images. Two experiments: old data only, then combined old+new.

### Architecture
- Base model: `google/vit-base-patch16-224-in21k`
- Training transform: **64×64 → 224×224 double-resize** (intentional blur — must use same in eval)
- 15 epochs, lr=2e-5, batch=8, `load_best_model_at_end=True` (best by eval_loss)
- Conda env: `vit_env`

### Experiment 1: Old Data Only (`split_dataset_fft/`)
- Data: `fft_augmented_nm/` (792 images total, 500Hz excluded)
- Split: **561 train / 293 test** (80/20 approx, stratified by frequency)
- Script: `train_vit_fft.py` / `train_vit_fft.sbatch`
- **Result: 98.6% overall** (289/293, best epoch 15)

| Category | Accuracy | Freq breakdown |
|----------|----------|----------------|
| Noise    | 98.0% (50/51) | — |
| Pulse    | 100% (34/34) | — |
| Ramp     | 100% (31/31) | — |
| Sine     | 96.6% (84/87) | 1Hz: 83.3%, 100–400Hz: 100% |
| Square   | 100% (90/90) | All freqs: 100% |
| **Overall** | **98.6%** | |

- Results: `vit_output_fft/vit_freq_breakdown_fft.json`

### Experiment 2: Combined Data (`split_dataset_combined/`)
- Data: `fft_augmented_nm/` (old) + `2.20.2026/fft/` (new)
  - Old: 792 images — pulse/ramp at 10Hz input only, sine/square 1–400Hz
  - New: 477 images — all categories at 1Hz, 100Hz, 200Hz, 300Hz, 400Hz
  - New data `Square/` folder (capital S) normalized to `square`
- Split: **938 train / 247 test** (80/20, stratified by frequency)
- Script: `split_combined.py`, `train_vit_combined.py` / `train_vit_combined.sbatch`
- **Result: 95.95% overall** (237/247, best epoch 8)

| Category | Accuracy | Freq breakdown |
|----------|----------|----------------|
| Noise    | 100% (44/44) | — |
| Pulse    | 100% (36/36) | All freqs: 100% |
| Ramp     | 91.4% (32/35) | 1–300Hz: 100%, **400Hz: 25% (1/4)** |
| Sine     | 89.6% (60/67) | 100–300Hz: 100%, **1Hz: 85.7%**, **400Hz: 58.3%** |
| Square   | 100% (65/65) | All freqs: 100% |
| **Overall** | **95.95%** | |

- Results: `vit_output_combined/vit_freq_breakdown_combined.json`

### Key ViT Findings
- ViT (98.6%) >> LLaVA+Qwen (86.4%) on old data — massive improvement, simpler pipeline
- Square perfectly classified at all frequencies (LLaVA+Qwen had 82.2%)
- Combined model (95.95%) lower than old-only (98.6%): harder task (pulse/ramp now span 1–400Hz)
- Main combined failures: sine 400Hz (58.3%) and ramp 400Hz (25%) — high-freq FFT patterns converge
- **Eval note**: test set used for both validation and checkpoint selection (slight optimistic bias)

### ViT Eval Scripts
- `eval_perclass.py` — per-class accuracy on login node (fp32, uses 64→224 transform)
- `classify_vit_by_freq.py` — per-freq breakdown; regex `[-_](\d+H[zx])` handles both old (`_100Hz_`) and new (`-100Hz-`) naming

### Transform Bug (important)
Training uses 64→224 double-resize. Using direct 224 resize in eval gives ~56% accuracy. Always use:
```python
transforms.Resize((64, 64)),
transforms.Resize((224, 224)),
```

### Running Eval on Login Node (no GPU needed)
```bash
cd MSEC/training_VIT
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup /jet/home/byler/miniconda3/envs/vit_env/bin/python eval_perclass.py > out.txt 2>&1 &
```

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
   - **Status: training complete** (105 steps, loss 0.95→0.21, 22 min on V100)
   - Checkpoints: `train_5class/checkpoints_FFT_5class/` (checkpoint-35, 70, 105)

5. **LLaVA Evaluation** (`train_5class/evaluateLLaVA_5class.py`, SLURM: `eval_5class.sbatch`)
   - Runs finetuned LLaVA on test images, asks same 3 questions
   - **Status: complete** (147 test images evaluated)
   - Output: `train_5class/results/{noise,sine,square,pulse,ramp}.json`

6. **Qwen Classification** (`train_5class/categorizeLLAMA_5class.py`, SLURM: `classify_5class.sbatch`)
   - Model: **Qwen2.5-32B-Instruct** (`models_setup/Qwen2.5-32B-Instruct`) on H100-80GB
   - Upgraded from 14B after it failed basic number comparison (said "23nm > 60nm")
   - 5-step decision tree (output format before steps, 5nm noise floor threshold):
     - STEP 1: Peak amplitude — >250nm → pulse, <60nm → STEP 2, 60-250nm → STEP 3
     - STEP 2: Low amplitude — peak at 1Hz or no peaks → pulse, otherwise → noise
     - STEP 3: Harmonic pattern — discard peaks <5nm, sort by freq, check equal spacing (within 30%)
     - STEP 4: Harmonic branch — 2nd/1st ratio ≤15% → ramp, >15% → square
     - STEP 5: No harmonics — 3rd peak ≥25nm → square, <25nm → sine
   - Prompt uses fill-in-the-blank format forcing explicit comparisons (e.g., "amplitude = 23. Is 23 > 250? No. Is 23 < 60? Yes.")
   - Script loads model once for all categories (not 5 separate loads)
   - **Status: running** (job 37572616 on H100, ~10 min/category)
   - Output: `train_5class/results/{category}_classified.json`

### Classifier Model Evolution (5-class)
- **Qwen2.5-14B-Instruct**: Failed — correctly extracted amplitudes but couldn't compare numbers (said "23nm > 60nm"), routed all noise to STEP 3 instead of STEP 2, 0/24 noise accuracy
- **Qwen2.5-32B-Instruct (v1 prompt)**: Also failed number comparison — said "6nm < 5nm" when discarding peaks in STEP 3, misclassified all standard ramp as sine (5/16 ramp). Also timed out (1hr) when loading model 5 times.
- **Qwen2.5-32B-Instruct (v2 prompt)**: Explicit fill-in-the-blank comparisons fix number errors. **Overall: 127/147 = 86.4%**

### Classification Results (5-class, Qwen 32B v2 prompt)
| Category | Accuracy | Details |
|----------|----------|---------|
| Noise    | 21/24 = 87.5% | 3→square |
| Sine     | 36/45 = 80.0% | 2→ramp, 7→square |
| Square   | 37/45 = 82.2% | 8→sine |
| Pulse    | 17/17 = 100%  | — |
| Ramp     | 16/16 = 100%  | — |
| **Overall** | **127/147 = 86.4%** | |

### Error Analysis
**Noise → square (3 errors)**:
- LLaVA reports amplitudes near 60nm boundary (42-58nm actual); may overestimate, pushing into STEP 3 (60-250nm path) where template bias (120/240/360 Hz) creates fake harmonic pattern → square

**Sine → ramp (3 errors: 2x 1Hz, 1x 300Hz)**:
- 1Hz sine: LLaVA fabricates peaks at 1/3/5 Hz (22nm/9nm), both >5nm threshold → equal spacing → harmonic → ratio 11% ≤ 15% → ramp. Predicted and accepted trade-off.
- 300Hz sine: leakage peaks may have equal spacing by coincidence → harmonic pattern → low ratio → ramp

**Sine → square (6 errors, all 300Hz)**:
- Spectral leakage at 300Hz creates large secondary peaks (e.g., 50nm at ~120Hz piezo resonance), 3rd peak ≥25nm → STEP 5 routes to square

**Square → sine (8 errors: 200Hz, 300Hz, 400Hz)**:
- High-frequency square waves have aliased harmonics that fold back (e.g., 3×300=900→100Hz, 5×300=1500→500Hz), breaking the equal-spacing pattern → no harmonic detected → STEP 5 where 3rd peak <25nm (aliased peaks are weaker) → sine
- This is the fundamental limitation: aliasing makes high-freq square indistinguishable from sine in FFT

**Key insight**: Sine↔square confusion at 200-400Hz is bidirectional — their FFT patterns converge due to aliasing and spectral leakage. Pulse and ramp are perfectly classified.

### Signal Characteristics
- **Noise**: Low amplitude (<60nm), no dominant frequency
- **Sine**: Single dominant peak, spectral leakage creates nearby 2nd peak (not a real harmonic)
- **Square**: Odd harmonics at f, 3f, 5f; aliasing at >100Hz may fold harmonics
- **Pulse**: Aperiodic transient, FFT shows 1/f decay with 1Hz artifact (not real frequency)
- **Ramp trail**: ~1Hz fundamental, needs 4096-pt window; harmonics at f, 2f, 3f
- **Ramp standard**: ~10Hz fundamental, clear harmonics at 3x, 5x

### Known Edge Cases
- ~5 pulse segments fall in 60-250nm range → may misclassify as ramp (accepted trade-off)
- 2/9 of 1Hz sine test samples: LLaVA hallucinates peaks at 1/3/5 Hz (22nm, 9nm) → will misclassify as ramp (accepted, 5nm threshold fixes other 7/9)
- 31/165 sine entries have 2nd peak >15% ratio due to spectral leakage → handled by decision tree using 3-peak data (leakage peaks are close together, not evenly spaced like harmonics)
- Square >100Hz may not show clean harmonic spacing → caught by STEP 5 (3rd peak ≥25nm)

### LLaVA Evaluation Quality
- **Ramp**: A- — excellent primary peak, harmonics well-captured
- **Sine**: B+ — good primary peak, leakage peak detected but freq slightly off
- **Square**: B — good primary peak, harmonic amplitudes sometimes overestimated
- **Noise**: C+ — amplitude off 15-30%, template bias toward 120/240/360 Hz
- **Pulse**: D+ — hallucinated discrete peaks on smooth 1/f decay, inconsistent decay descriptions
- Key issues don't affect classification because noise/pulse rely on amplitude + frequency thresholds, not secondary peaks

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
# ViT combined pipeline
python MSEC/training_VIT/split_combined.py                      # combine + split data
sbatch MSEC/training_VIT/train_vit_combined.sbatch             # ViT training
# eval on login node (no GPU queue needed):
cd MSEC/training_VIT && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nohup /jet/home/byler/miniconda3/envs/vit_env/bin/python classify_vit_by_freq.py \
    --model vit_output_combined --test split_dataset_combined/test \
    --out vit_output_combined/vit_freq_breakdown_combined.json > out.txt 2>&1 &

# ViT old-data-only pipeline (completed)
sbatch MSEC/training_VIT/train_vit_fft.sbatch

# 5-class LLaVA+Qwen pipeline (completed)
sbatch MSEC/training_fft/train_5class/train_5class.sbatch      # LLaVA finetuning
sbatch MSEC/training_fft/train_5class/eval_5class.sbatch       # LLaVA evaluation
sbatch MSEC/training_fft/train_5class/classify_5class.sbatch   # Qwen classification

# 3-class pipeline (completed)
sbatch MSEC/training_fft/train_fft.sbatch
sbatch MSEC/training_fft/eval_fft.sbatch
sbatch MSEC/training_fft/classify_fft.sbatch
```
