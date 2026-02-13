#!/bin/bash
#SBATCH --job-name=llava-fft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=03:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/train_fft_%j.out
#SBATCH --error=MSEC/training_fft/train_fft_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

# Activate conda environment
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava

# Use the installed llava package (editable install at /ocean/projects/cis240145p/byler/LLaVA)
# which was used for V6 training successfully
bash MSEC/training_fft/finetune_fft.sh
