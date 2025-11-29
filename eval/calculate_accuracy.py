import json

files = [
    "results/correct.json/randomNoise_correct.json",
    "results/correct.json/sineWave_correct.json",
    "results/correct.json/squareWave_correct.json"
]

results = {}

for input_file in files:
    with open(input_file, "r") as f:
        data = json.load(f)

    total = len(data)
    correct = 0

    for item in data:
        if item["prediction"] == item["label"]:
            correct += 1

    accuracy = correct / total * 100 if total > 0 else 0

    results[input_file] = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": round(accuracy, 2)
    }

    print(f"{input_file}:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%\n")

  #Save results to a JSON file
with open("results/llava/accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved accuracy results to results/llava/accuracy_results.json")


