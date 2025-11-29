# define our variables
MODELPATH=/data/ilminur/Llama_Instruct/Llama-3.2-3B-Instruct
IMAGEFOLDER=/data/ilminur/LLaVA/data/subset/testData

# random noise
python eval/evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/NoiseData --output-file eval/results/llava/randomNoise.json

# sine waves
python eval/evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/SineData --output-file eval/results/llava/sineWave.json

#square waves
python eval/evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/SquareData --output-file eval/results/llava/squareWave.json
