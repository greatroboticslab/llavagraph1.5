# Vision Transformer (ViT) Classification Pipeline

This pipeline implements a waveform classification system using a Vision Transformer (ViT). Unlike LLM-based approaches, this uses a dedicated computer vision model to classify time-domain images directly.

## Pipeline Overview
```
Raw Data → Time-Domain Augmentation → Train/Test Split → ViT Training → Model Evaluation → Inference
```

Classification targets: **noise, sine, square, pulse, ramp**
---

## Step 1: Time-Domain Data Augmentation

**Script**: `augmented_vit_V2.py`

Processes raw waveform CSV data into augmented time-domain images. 

**Output**: `images_Timedomain_augmented/` folder containing the generated raw image dataset.

---

## Step 2: Split into Train/Test Sets

**Script**: `split_3.py` 

Creates a stratified train/val/test split (e.g., 60/40 (60/20/20)) while maintaining data integrity.

**Usage:**
```bash
python split_3.py
```

Output: split_dataset_3way/(train 60%;val 20%; test20%) and split_dataset_3_aug/(train 60%;val 20%; test20%) folders.

## Step 3: Train ViT Classifier

**Script**: `train_vit_classifierV2.py`

Finetunes a `vit-base-patch16-224-in21k` model on time-domain waveform images.

**Key features:**
- Architecture: Vision Transformer (ViT) with custom classification head based on detected folder names.
- Image Preprocessing: Implements a "Weak Transform" strategy:
    - Resizes image down to $64 \times 64$ pixels.
    - Resizes back up to $224 \times 224$ pixels (introducing slight blur/downsampling effect).
    - Normalizes using ImageNet mean/std from the ViT processor.
- Hardware Acceleration: Automatically detects and uses MPS (Metal Performance Shaders) for Apple Silicon Macs or CPU.
- Hyperparameters:
    - Learning Rate: 2e-5
    - Epochs: 20
    - Batch Size: 8
    - Strategy: Saves the "Best Model" based on evaluation accuracy at the end of training.

**Usage:**
```bash
python train_vit_classifierV2.py
```

## Step 4: Classification and Performance Summary

**Script**: `classify_vitV2.py`

Evaluates the finetuned model against the test set and generates a detailed performance report.

**Key features:**
- Automated Evaluation: Iterates through the split_dataset/test subfolders and compares predictions against folder labels.
- Metrics Calculation: Tracks total samples, correct predictions, and accuracy percentage for every category (Sine, Square, Noise, etc.).
- Summary Export: Produces a JSON summary including per-class accuracy.

**Usage:**
```bash
python classify_vitV2.py 
```
**Output File:** `vit_summary.json`

**Format:**
```
JSON
{
  "noise": {
    "total_samples": 19,
    "correct_predictions": 11,
    "accuracy_percent": 57.89,
    "average_confidence": 0.5863
  },
  ...
}
```

## Environment Setup
vit_env
This pipeline requires `evaluate` and `scikit-learn` in addition to standard computer vision libraries.
```bash
conda create -n vit_env python=3.10
conda activate vit_env
pip install torch torchvision torchaudio
pip install transformers datasets evaluate scikit-learn pillow
```

## File Structure
```
training_VIT/
├── readme.md
├── dataV2       # new version
├── resultV2     # new version
├── scriptsV2    # new version (updated)please use!
├── timedomain-augmentation.py  # Step 1: Time-Domain Data Augmentation(old version
├── split.py                # Step 2: Split into Train/Test Sets
├── train_vit_classifier.py # Step 3: ViT training logic
├── classify_vit.py         # Step 4: Classification and Performance Summary
├── time-domain-ori/        # Raw original time-domain images(without augmented images)
├── split_dataset/          # Split version 1 (60/40)
└── split_dataset_2/        # Split version 2 (80/20)
```
