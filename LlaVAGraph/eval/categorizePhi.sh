# define our variables
MODELPATH=/data/ilminur/models/llava-v1.6-vicuna-7b
OUTPUTDIR=/data/ilminur/LLaVA/LlaVAGraph/eval/results/llava

mkdir -p $OUTPUTDIR

# random noise
python /data/ilminur/LLaVA/LlaVAGraph/eval/categorizePhi.py \
    --model-path $MODELPATH \
    --conversation-file $OUTPUTDIR/randomNoise.json
    --output-file $OUTPUTDIR/randomNoise.json

# sine waves
python /data/ilminur/LLaVA/LlaVAGraph/eval/categorizePhi.py \
    --model-path $MODELPATH \
    --conversation-file $OUTPUTDIR/sineWave.json
    --output-file $OUTPUTDIR/sineWave.json

#square waves
python /data/ilminur/LLaVA/LlaVAGraph/eval/categorizePhi.py \
    --model-path $MODELPATH \
    --conversation-file $CONVDIR/squareWave.json \
    --output-file $OUTPUTDIR/squareWave.json
    

