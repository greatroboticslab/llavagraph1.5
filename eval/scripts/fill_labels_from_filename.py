import json
import os

# Files to process
files = [
    "results/llava/randomNoise_fixed.json",
    "results/llava/sineWave_fixed.json",
    "results/llava/squareWave_fixed.json"
]

# Mapping from file type to label
file_label_map = {
    "randomNoise": "A) Random noise",
    "sineWave": "B) Sine wave",
    "squareWave": "C) Square wave"
}

for file_path in files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Determine the label from the filename
    base_name = os.path.basename(file_path)
    for key, label in file_label_map.items():
        if key in base_name:
            ground_label = label
            break
    else:
        ground_label = "Unknown"

    # Load JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # Fill labels
    for entry in data:
        entry["label"] = ground_label

    # Save new file
    out_file = file_path.replace(".json", "_labeled.json")
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved labeled file: {out_file}")

