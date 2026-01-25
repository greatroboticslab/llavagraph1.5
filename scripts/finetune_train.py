#!/bin/bash

# Configuration parameters
MODEL_PATH="models_setup/llava-v1.5-7b"
DATA_PATH="data/trainingData.json"
IMAGE_FOLDER="data/"
OUTPUT_DIR="checkpoints/"
DEEPSPEED_CONFIG="scripts/zero3.json"

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p scripts

# Ensure zero3.json exists
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "Error: ${DEEPSPEED_CONFIG} does not exist!"
    exit 1
fi

# Verify model path
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path ${MODEL_PATH} does not exist!"
    exit 1
fi

# Run training
deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
