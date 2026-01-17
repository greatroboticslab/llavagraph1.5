# scripts

## finetune_train.py
Full Training Command

Base Model: LLaVA-v1.5-7B(models_setup/llava-v1.5-7b)

Vision Encoder: CLIP ViT-Large (openai/clip-vit-large-patch14-336)

Fine-tuning Method: LoRA (rank=128, alpha=256)

Training Framework: DeepSpeed ZeRO-2

Hardware: 2x NVIDIA GPUs

## zero2.json
DeepSpeed ZeRO-2 configuration for distributed training
