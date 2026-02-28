#!/bin/bash
#SBATCH --job-name=vit_fft_eval
#SBATCH --output=vit_fft_eval_%j.out
#SBATCH --error=vit_fft_eval_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=00:30:00
#SBATCH --account=cis240145p

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5/MSEC/training_VIT

conda run -n vit_env python classify_vit_fft.py
