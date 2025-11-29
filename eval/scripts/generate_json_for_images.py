import os
import json

# Change this to your image folder
image_folder = "/data/ilminur/LLaVA/eval/eval image/SquareWave"
output_file = "eval/results/llava/squareWave_new.json"

data = []
for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith(".png"):
        entry = {
            "image": filename,
            "conversation": [
                {"question": "Describe the line in the graph.", "answer": ""},
                {"question": "Is the line continuous?", "answer": ""},
                {"question": "Does the graph contain random points?", "answer": ""},
                {"question": "Does the graph contain sharp corners?", "answer": ""}
            ]
        }
        data.append(entry)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated JSON file: {output_file}")
