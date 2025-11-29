# define our variables
MODELPATH=/data/ilminur/LLaVA/eval/Llama-3.2-3B-Instruct

# random noise
python /data/ilminur/LLaVA/eval/categorizeLLAMA.py --model-path $MODELPATH --conversation-file results/llava/randomNoise.json --output-file results/randomNoise.json

# sine waves
python /data/ilminur/LLaVA/eval/categorizeLLAMA.py --model-path $MODELPATH --conversation-file results/llava/sineWave.json --output-file results/sineWave.json

#square waves
python /data/ilminur/LLaVA/eval/categorizeLLAMA.py --model-path $MODELPATH --conversation-file results/llava/squareWave.json --output-file results/squareWave.json
