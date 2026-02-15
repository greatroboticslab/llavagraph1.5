#!/bin/bash
#SBATCH --job-name=llava-eval-5c
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=03:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/train_5class/eval_5class_%j.out
#SBATCH --error=MSEC/training_fft/train_5class/eval_5class_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava

MODELPATH="MSEC/training_fft/train_5class/checkpoints_FFT_5class"
MODELBASE="models_setup/llava-v1.5-7b"
TESTDIR="MSEC/training_fft/train_5class/test"
RESULTDIR="MSEC/training_fft/train_5class/results"

mkdir -p $RESULTDIR

for CAT in noise sine square pulse ramp; do
    echo "=== Evaluating $CAT ==="
    python MSEC/training_fft/train_5class/evaluateLLaVA_5class.py \
        --model-path $MODELPATH \
        --model-base $MODELBASE \
        --image-folder $TESTDIR/$CAT \
        --output-file $RESULTDIR/$CAT.json
done

echo "=== LLaVA evaluation complete ==="
