#!/bin/bash
#SBATCH --job-name=vit_fft
#SBATCH --output=vit_fft_%j.out
#SBATCH --error=vit_fft_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --account=cis240145p

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/training_VIT

conda run -n vit_env python train_vit_fft.py
