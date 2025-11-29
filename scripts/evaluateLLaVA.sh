#!/usr/bin/env bash

#set -euo pipefail

#BASE_MODEL="/data/ilminur/models/llava-v1.6-vicuna-7b"
#LORA_DIR="/data/ilminur/LLaVA/checkpoints"
#IMAGE_FOLDER="/data/ilminur/LLaVA/LlaVAGraph/eval/images"
#OUTPUTDIR="/data/ilminur/LLaVA/LlaVAGraph/eval/results"
#PYROOT="/data/ilminur/LLaVA"

#mkdir -p "$OUTPUTDIR"
#export PYTHONPATH="$PYROOT:$PYTHONPATH"

# Random Noise
#python $PYROOT/llavaGraph/eval/evaluateLLaVA.py \
 # --model_name_or_path "$BASE_MODEL" \
  #--lora_weights "$LORA_DIR" \
  #--image_folder "$IMAGE_FOLDER/NoiseData" \
  #--output_file "$OUTPUTDIR/randomNoise.json" \
  #--device cuda

# Sine Wave
#python $PYROOT/llavaGraph/eval/evaluateLLaVA.py \
 # --model_name_or_path "$BASE_MODEL" \
  #--lora_weights "$LORA_DIR" \
  #--image_folder "$IMAGE_FOLDER/SineData" \
  #--output_file "$OUTPUTDIR/sineWave.json" \
  #--device cuda

# Square Wave
#python $PYROOT/llavaGraph/eval/evaluateLLaVA.py \
 # --model_name_or_path "$BASE_MODEL" \
 # --lora_weights "$LORA_DIR" \
 # --image_folder "$IMAGE_FOLDER/SquareData" \
 # --output_file "$OUTPUTDIR/squareWave.json" \
 # --device cuda

#echo "✅ Evaluation done. Results in $OUTPUTDIR"

#!/bin/bash
set -euo pipefail

# ----  EDIT THESE TO MATCH YOUR INSTALL ----
BASE_MODEL="/data/ilminur/models/llava-v1.6-vicuna-7b"   # your base HF model folder or HF repo id
CHECKPOINT="/data/ilminur/LLaVA/checkpoints"            # your LoRA/adapter checkpoint
IMAGEFOLDER="/data/ilminur/LLaVA/data"                  # folder that contains NoiseData/SineData/SquareData
EVAL_SCRIPT="/data/ilminur/LLaVA/eval/evaluateLLaVA.py" # path to the evaluateLLaVA.py script
OUTPUTDIR="/data/ilminur/LLaVA/eval/results/llava"
DEVICE="cuda"                                           # or "cpu"

mkdir -p "$OUTPUTDIR"
echo "Output will be saved to $OUTPUTDIR"

# --- Random noise
echo "==== Running Random Noise evaluation ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGEFOLDER/NoiseData" \
  --output-file "$OUTPUTDIR/randomNoise.json" \
  --device "$DEVICE"

# --- Sine waves
echo "==== Running Sine Wave evaluation ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGEFOLDER/SineData" \
  --output-file "$OUTPUTDIR/sineWave.json" \
  --device "$DEVICE"

# --- Square waves
echo "==== Running Square Wave evaluation ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGEFOLDER/SquareData" \
  --output-file "$OUTPUTDIR/squareWave.json" \
  --device "$DEVICE"

echo "All done. Results in $OUTPUTDIR"


