import json
import os
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def extract_ground_truth(conversation_id):
    """Extract ground truth label from filename."""
    if "Sine" in conversation_id:
        return "B"
    elif "RandomNoise" in conversation_id:
        return "A"
    elif "SquareWave" in conversation_id:
        return "C"
    else:
        return "Unknown"

def evaluate(pred_file, output_dir):
    with open(pred_file, "r") as f:
        predictions = json.load(f)

    y_true = []
    y_pred = []

    for entry in predictions:
        conv_id = entry["conversationId"]
        pred_label = entry["response"].split(")")[0].strip()  # Extract "A", "B", or "C"
        y_pred.append(pred_label)
        y_true.append(extract_ground_truth(conv_id))

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    # Classification report
    report = classification_report(y_true, y_pred, labels=["A","B","C"])
    print("Classification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["A","B","C"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["A","B","C"], yticklabels=["A","B","C"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

    # Save results
    results = {
        "accuracy": acc,
        "classification_report": report
    }
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as out_f:
        json.dump(results, out_f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True, help="Path to categorized JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.pred_file, args.output_dir)
