#!/bin/bash
#SBATCH --job-name=llama-class
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=01:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/classify_fft_%j.out
#SBATCH --error=MSEC/training_fft/classify_fft_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava_infer

LLAMA_MODEL="models_setup/Llama-3.1-8B-Instruct"
RESULTDIR="MSEC/training_fft/results"

echo "=== Classifying noise ==="
python MSEC/training_fft/categorizeLLAMA_FFT.py \
    --model-path $LLAMA_MODEL \
    --conversation-file $RESULTDIR/noise.json \
    --output-file $RESULTDIR/noise_classified.json

echo "=== Classifying sine ==="
python MSEC/training_fft/categorizeLLAMA_FFT.py \
    --model-path $LLAMA_MODEL \
    --conversation-file $RESULTDIR/sine.json \
    --output-file $RESULTDIR/sine_classified.json

echo "=== Classifying square ==="
python MSEC/training_fft/categorizeLLAMA_FFT.py \
    --model-path $LLAMA_MODEL \
    --conversation-file $RESULTDIR/square.json \
    --output-file $RESULTDIR/square_classified.json

echo ""
echo "=== FINAL RESULTS ==="
python -c "
import json
total_correct = 0
total = 0
for cat in ['noise', 'sine', 'square']:
    with open('$RESULTDIR/' + cat + '_classified.json') as f:
        d = json.load(f)
    print(f'  {cat:6}: {d[\"correct\"]}/{d[\"total\"]} = {d[\"accuracy\"]:.1f}%')
    total_correct += d['correct']
    total += d['total']
print(f'  Overall: {total_correct}/{total} = {total_correct/total*100:.1f}%')
"
