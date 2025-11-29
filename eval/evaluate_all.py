import json
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def extract_label(filename):
    if "RandomNoise" in filename:
        return "A"
    elif "SineTrials" in filename:
        return "B"
    elif "SquareWave" in filename:
        return "C"
    return None

def evaluate(pred_file, output_dir):
    with open(pred_file, "r") as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    for item in data:
        true_label = extract_label(item["conversationId"])
        if true_label:
            y_true.append(true_label)
            y_pred.append(item["response"][0])

    if not y_true:
        raise ValueError("No ground truth labels found — check file naming.")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=["A","B","C"]))

    cm = confusion_matrix(y_true, y_pred, labels=["A","B","C"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["A","B","C"], yticklabels=["A","B","C"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "classification_report": classification_report(y_true, y_pred, labels=["A","B","C"], output_dict=True)
        }, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.pred_file, args.output_dir)
