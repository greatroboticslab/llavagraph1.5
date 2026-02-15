#!/bin/bash
#SBATCH --job-name=llava-fft-5class
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=03:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/train_5class/train_5class_%j.out
#SBATCH --error=MSEC/training_fft/train_5class/train_5class_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

# Activate conda environment
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava

bash MSEC/training_fft/train_5class/finetune_5class.sh
