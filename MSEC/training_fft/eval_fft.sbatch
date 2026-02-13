#!/bin/bash
#SBATCH --job-name=llava-eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=03:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/eval_fft_%j.out
#SBATCH --error=MSEC/training_fft/eval_fft_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava

MODELPATH="MSEC/training_fft/checkpoints_FFT"
MODELBASE="models_setup/llava-v1.5-7b"
TESTDIR="MSEC/training_fft/test"
RESULTDIR="MSEC/training_fft/results"

mkdir -p $RESULTDIR

echo "=== Evaluating noise ==="
python MSEC/training_fft/evaluateLLaVA_FFT.py \
    --model-path $MODELPATH \
    --model-base $MODELBASE \
    --image-folder $TESTDIR/noise \
    --output-file $RESULTDIR/noise.json

echo "=== Evaluating sine ==="
python MSEC/training_fft/evaluateLLaVA_FFT.py \
    --model-path $MODELPATH \
    --model-base $MODELBASE \
    --image-folder $TESTDIR/sine \
    --output-file $RESULTDIR/sine.json

echo "=== Evaluating square ==="
python MSEC/training_fft/evaluateLLaVA_FFT.py \
    --model-path $MODELPATH \
    --model-base $MODELBASE \
    --image-folder $TESTDIR/square \
    --output-file $RESULTDIR/square.json

echo "=== LLaVA evaluation complete ==="
