#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# stage1- checkpoint（output_dir）
CHECKPOINT="/data/ilminur/12.14llava/checkpoints/lora_llavagraph_simple_final"

#  base model
BASE_MODEL="lmsys/vicuna-7b-v1.5"

# Stage 2 / eval
IMAGE_ROOT="/data/ilminur/12.14llava/data/stage2_output"

# output
OUTPUTDIR="/data/ilminur/12.14llava/output"
EVAL_SCRIPT="/data/ilminur/12.14llava/llavagraph1.5/llava/eval/evaluateLLaVA.py"


mkdir -p "$OUTPUTDIR"
echo "Results will be saved to $OUTPUTDIR"

# ====== Random Noise ======
echo "==== Evaluating Random Noise ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGE_ROOT/RandomNoise" \
  --output-file "$OUTPUTDIR/randomNoise.json" \
  --load-8bit --device cuda

# ====== Sine Waves ======
echo "==== Evaluating Sine Waves ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGE_ROOT/SineWave" \
  --output-file "$OUTPUTDIR/sineWave.json" \
  --load-8bit --device cuda

# ====== Square Waves ======
echo "==== Evaluating Square Waves ===="
python "$EVAL_SCRIPT" \
  --model-path "$CHECKPOINT" \
  --model-base "$BASE_MODEL" \
  --image-folder "$IMAGE_ROOT/SquareWave" \
  --output-file "$OUTPUTDIR/squareWave.json" \
  --load-8bit --device cuda

echo "✅ Evaluation finished."
