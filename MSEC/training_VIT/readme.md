# Vision Transformer (ViT) Classification Pipeline

This pipeline implements a waveform classification system using a Vision Transformer (ViT). Unlike LLM-based approaches, this uses a dedicated computer vision model to classify time-domain images directly.

## Pipeline Overview
```
Raw Data → Time-Domain Augmentation → Train/Test Split → ViT Training → Model Evaluation → Inference
```

Classification targets: **noise, sine, square, pulse, ramp**
---

## Step 1: Time-Domain Data Augmentation

**Script**: `timedomain-augment.py`

Processes raw waveform CSV data into augmented time-domain images. 

**Key features:**
- Converts raw signal data into visual waveform plots.
- Applies data augmentation by taking different segment offsets.
- Handles multiple signal types including standard oscillators (Sine, Square, Pulse, Ramp) and Noise.

**Output**: `time-domain-ori/` folder containing the generated raw image dataset.

---

## Step 2: Split into Train/Test Sets

**Script**: `split_data.py`

Creates a stratified train/test split (e.g., 60/40) while maintaining data integrity.

**Key features:**
- **Source Consistency**: Groups by original CSV identifier to ensure that segments belonging to the same recording stay in the same set (prevents data leakage).
- **Stratification**: Ensures each frequency and wave type is proportionally represented in both sets.
- **Frequency Filtering**: Excludes 500Hz signals (optional) to maintain dataset quality where Nyquist frequency limits clarity.

**Usage:**
```bash
python split.py
```

Output: split_dataset/(train 60%;test40%) and split_dataset_2/(train 80% ;test 20%) folders.

## Step 3: Train ViT Classifier

**Script**: `train_vit_classifier.py`

Finetunes a Vision Transformer (ViT) on the augmented waveform images.

**Key features:**
- Model: Leverages google/vit-base-patch16-224 via the Hugging Face Transformers library.
- Preprocessing: Resizes images to $224 \times 224$ and applies standard ImageNet normalization.
- Optimization: Uses AdamW with a linear learning rate scheduler and early stopping.Usage:Bashpython train_vit_classifier.py

**Usage:**
```bash
python train_vit_classifier.py
```

## Step 4: Classification and Inference

**Script**: `classify_vit.py`

Runs the trained model on test data or new samples to evaluate performance and provide labels.

**Key features:**
- Loads the best-saved model checkpoint.
- Outputs classification metrics including Accuracy, F1-score, and a Confusion Matrix.
- Provides a simple interface for single-image prediction.

**Usage:**
```bash
python classify_vit.py --image_path /path/to/image.png
```

## Environment Setup
vit_env
```bash
conda create -n vit_env python=3.10
conda activate vit_env
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn matplotlib
```

## File Structure
training_VIT/
├── readme.md               
├── train_vit_classifier.py # Step 3: ViT training logic
├── classify_vit.py         # Step 4: Inference 
├── time-domain-ori/        # Raw original time-domain images(without augmented images)
├── split_dataset/          # Split version 1 (60/40)
└──  split_dataset_2/        # Split version 2 (80/20)
