#!/bin/bash

# Move into your LLaVA repo
cd /data/ilminur/LLaVA

# Activate your conda environment
source /home/mtsu/miniconda3/etc/profile.d/conda.sh
conda activate llava

# Start the finetune script
bash scripts/v1_5/finetune_task_lora.sh
