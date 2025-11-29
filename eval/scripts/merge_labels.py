import json

# Ground truth file
ground_truth_file = "eval/results/llava/ground_truth.json"

# Prediction files
prediction_files = [
    "eval/results/llava/randomNoise_fixed.json",
    "eval/results/llava/sineWave_fixed.json",
    "eval/results/llava/squareWave_fixed.json"
]

# Load ground truth
with open(ground_truth_file, "r") as f:
    ground_truth = json.load(f)

# Process each prediction file
for pred_file in prediction_files:
    with open(pred_file, "r") as f:
        predictions = json.load(f)

    for pred in predictions:
        image_id = pred.get("conversationId", pred.get("image"))
        pred["label"] = ground_truth.get(image_id, "Unknown")

    out_file = pred_file.replace("_fixed.json", "_fixed_labeled.json")
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"[INFO] Labeled predictions saved: {out_file}")
