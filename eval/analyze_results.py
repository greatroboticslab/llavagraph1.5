#!/usr/bin/env python3
import os, json, re, argparse, csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def clean_text(s):
    if s is None:
        return ""
    # remove tags like <s> etc and any non-printable
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def predict_label(text):
    t = clean_text(text)
    # rules (simple heuristics). Tune keywords if needed.
    if any(k in t for k in ["square", "sharp corner", "sharp corners", "abrupt", "flat regions", "two distinct levels", "corners"]):
        return "Square"
    if any(k in t for k in ["sine", "sinus", "smooth", "smoothly", "continuous", "gradual", "periodic", "oscillat"]):
        return "Sine"
    if any(k in t for k in ["noise", "random", "stochastic", "no structure", "random points", "chaotic"]):
        return "Noise"
    # fallback guesses based on words
    if "sharp" in t or "corner" in t:
        return "Square"
    if "smooth" in t or "sin" in t:
        return "Sine"
    if "random" in t:
        return "Noise"
    return "Unknown"

def analyze(results_dir):
    files = {
        "randomNoise.json": "Noise",
        "sineWave.json": "Sine",
        "squareWave.json": "Square"
    }
    detailed_rows = []
    per_file_counts = {}
    confusion = defaultdict(lambda: Counter())

    total = 0
    total_correct = 0

    for fname, expected_label in files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.isfile(path):
            print(f"[WARN] Missing {path}, skipping.")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        correct = 0
        n = 0
        for entry in data:
            n += 1
            # Build combined text of all answers for this image
            conv = entry.get("conversation", [])
            answers = []
            for qa in conv:
                # some json may have {"question":..., "answer": "text"} or different keys
                a = qa.get("answer") if isinstance(qa, dict) else None
                if a is None and isinstance(qa, (list, tuple)) and len(qa) > 1:
                    a = qa[-1]
                answers.append(a or "")
            combined = " ".join(answers)
            predicted = predict_label(combined)
            correct_flag = (predicted == expected_label)
            if correct_flag:
                correct += 1
                total_correct += 1
            total += 1
            confusion[expected_label][predicted] += 1
            detailed_rows.append({
                "file": fname,
                "image": entry.get("image", ""),
                "expected": expected_label,
                "predicted": predicted,
                "combined_answer": combined
            })
        per_file_counts[fname] = {"expected": expected_label, "n": n, "correct": correct, "accuracy": (correct / n if n>0 else 0)}

    overall = {"total": total, "correct": total_correct, "accuracy": (total_correct/total if total>0 else 0)}

    # Write outputs
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "detailed_predictions.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["file","image","expected","predicted","combined_answer"])
        writer.writeheader()
        for r in detailed_rows:
            writer.writerow(r)

    summary = {
        "per_file": per_file_counts,
        "overall": overall,
        "confusion": {k: dict(v) for k, v in confusion.items()}
    }
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Pretty text summary
    text_path = os.path.join(results_dir, "summary.txt")
    with open(text_path, "w") as f:
        f.write("Evaluation summary\n\n")
        f.write(f"Overall: {overall['correct']} / {overall['total']}  accuracy={overall['accuracy']:.4f}\n\n")
        for fname, info in per_file_counts.items():
            f.write(f"{fname} (expected={info['expected']}): {info['correct']} / {info['n']}  accuracy={info['accuracy']:.4f}\n")
        f.write("\nConfusion matrix (expected -> predicted counts):\n")
        for expected, ctr in confusion.items():
            f.write(f"{expected} -> {dict(ctr)}\n")

    print("Saved detailed CSV to:", csv_path)
    print("Saved summary JSON to:", summary_path)
    print("Saved summary text to:", text_path)

    # Plot confusion matrix
    labels = sorted(list(set(list(confusion.keys()) + [p for ctr in confusion.values() for p in ctr.keys()])))
    if not labels:
        print("[WARN] No labels found to plot.")
        return

    mat = []
    for e in labels:
        row = []
        for p in labels:
            row.append(confusion[e].get(p, 0))
        mat.append(row)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mat, interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", color="white" if mat[i][j]>max(max(mat))/2 else "black")
    fig.colorbar(im)
    plt.title("Confusion matrix")
    plt.tight_layout()
    png_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(png_path)
    print("Saved confusion matrix plot to:", png_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=".", help="folder containing randomNoise.json, sineWave.json, squareWave.json")
    args = parser.parse_args()
    analyze(args.results_dir)
