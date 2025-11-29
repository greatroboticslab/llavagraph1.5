# define our variables

MODELPATH=/data/ilminur/models/llava-v1.6-vicuna-7b
RESULTDIR=/data/ilminur/LLaVA/eval/results/llava


# random noise
python3 categorizePhi.py \
  --model-path "$MODELPATH" \
  --conversation-file "$RESULTDIR/randomNoise.json" \
  --output-file "$RESULTDIR/final_randomNoise.json"

# sine waves
python3 categorizePhi.py \
  --model-path "$MODELPATH" \
  --conversation-file "$RESULTDIR/sineWave.json" \
  --output-file "$RESULTDIR/final_sineWave.json"

# square waves
python3 categorizePhi.py \
  --model-path "$MODELPATH" \
  --conversation-file "$RESULTDIR/squareWave.json" \
  --output-file "$RESULTDIR/final_squareWave.json"
  
echo "======================================================"
echo "Categorization completed."
