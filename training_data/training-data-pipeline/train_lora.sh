#!/bin/bash

# LLaVAGraph LoRA Training Script
# Training configuration: LoRA r=8, batch_size=1, lr=1e-4

cd /data/ilminur/12.14llava/
conda activate llava

CUDA_VISIBLE_DEVICES=0 python llavagraph1.5/llava/train/train_mem.py \
>     --lora_enable True --lora_r 8 --lora_alpha 16 \
>     --model_name_or_path lmsys/vicuna-7b-v1.5 --version v1 \
>     --data_path data/fullData.json \
>     --image_folder data/stage1_input \
>     --vision_tower openai/clip-vit-large-patch14-336 \
>     --image_aspect_ratio pad \
>     --fp16 True \
>     --output_dir checkpoints/lora_llavagraph_simple_final \
>     --per_device_train_batch_size 1 \
>     --num_train_epochs 1 \
>     --evaluation_strategy no \
>     --save_strategy steps \
>     --save_steps 20 \
>     --learning_rate 1e-4 \
>     --model_max_length 512 \
>     --dataloader_num_workers 0 \
>     --lazy_preprocess False \
>     --group_by_modality_length False
