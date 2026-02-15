#!/bin/bash
#SBATCH --job-name=qwen-class-5c
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000M
#SBATCH --time=02:00:00
#SBATCH --account=cis240145p
#SBATCH --output=MSEC/training_fft/train_5class/classify_5class_%j.out
#SBATCH --error=MSEC/training_fft/train_5class/classify_5class_%j.err

cd /ocean/projects/cis240145p/byler/ben/llavagraph1.5

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate llava_infer

LLAMA_MODEL="models_setup/Qwen2.5-32B-Instruct"
RESULTDIR="MSEC/training_fft/train_5class/results"

python MSEC/training_fft/train_5class/categorizeLLAMA_5class.py \
    --model-path $LLAMA_MODEL \
    --conversation-files $RESULTDIR/noise.json $RESULTDIR/sine.json $RESULTDIR/square.json $RESULTDIR/pulse.json $RESULTDIR/ramp.json \
    --output-dir $RESULTDIR

echo ""
echo "=== FINAL RESULTS ==="
python -c "
import json
total_correct = 0
total = 0
for cat in ['noise', 'sine', 'square', 'pulse', 'ramp']:
    path = '$RESULTDIR/' + cat + '_classified.json'
    try:
        with open(path) as f:
            d = json.load(f)
        print(f'  {cat:6}: {d[\"correct\"]}/{d[\"total\"]} = {d[\"accuracy\"]:.1f}%')
        total_correct += d['correct']
        total += d['total']
    except FileNotFoundError:
        print(f'  {cat:6}: NOT FOUND')
if total > 0:
    print(f'  Overall: {total_correct}/{total} = {total_correct/total*100:.1f}%')
"
