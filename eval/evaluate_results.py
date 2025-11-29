#!/usr/bin/env python3
"""
Evaluate categorized JSON files produced by your categorization step.
Saves:
 - evaluation_results.json (summary)
 - confusion_matrix.png
 - detailed_predictions.csv
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

LABELS = ["A", "B", "C"]  # A=random, B=sine, C=square

def extract_pred_label(response_text):
    """Extract label A/B/C from model response string, fallback to keyword heuristics."""
    if not response_text:
        return None
    s = response_text.strip()
    # Look for pattern "A)" or "A) ..." anywhere
    m = re.search(r'\b([ABC])\)', s)
    if m:
        return m.group(1)
    # fallback: first non-space char
    if len(s) > 0 and s[0] in LABELS:
        return s[0]
    s_low = s.lower()
    if "random" in s_low or "noise" in s_low or "unpredictable" in s_low or "stochastic" in s_low:
        return "A"
    if "sine" in s_low or "continuous" in s_low or "smooth" in s_low:
        return "B"
    if "square" in s_low or "sharp" in s_low or "abrupt" in s_low or "discontinuous" in s_low:
        return "C"
    return None

def expected_label_from_filename(fname):
    n = fname.lower()
    if "sine" in n:
        return "B"
    if "square" in n:
        return "C"
    if "random" in n or "noise" in n:
        return "A"
    # unknown — return None
    return None

def load_json_file(path):
    data = json.load(open(path, "r"))
    return data

def evaluate_files(file_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    per_file_stats = {}

    y_true = []
    y_pred = []

    for fp in file_paths:
        fp = Path(fp)
        if not fp.exists():
            print(f"⚠️  File not found: {fp}, skipping.")
            continue
        expected_label = expected_label_from_filename(fp.name)
        if expected_label is None:
            print(f"⚠️  Cannot infer ground-truth from filename: {fp.name}. Skipping.")
            continue

        data = load_json_file(fp)
        total = 0
        correct = 0
        for item in data:
            # conversationId or image as identifier
            idx = item.get("conversationId") or item.get("image") or item.get("id") or "<unknown>"
            resp = item.get("response") or item.get("answer") or ""
            pred = extract_pred_label(resp)
            if pred is None:
                pred = "?"  # missing indicator
            is_correct = (pred == expected_label)
            total += 1
            if is_correct:
                correct += 1
            y_true.append(expected_label)
            y_pred.append(pred)
            overall_rows.append({
                "file": fp.name,
                "conversationId": idx,
                "expected": expected_label,
                "pred": pred,
                "pred_text": resp.replace("\n"," ").strip(),
                "correct": bool(is_correct)
            })

        acc = correct / total if total>0 else 0.0
        per_file_stats[fp.name] = {"correct": correct, "total": total, "accuracy": acc}

        print(f"{fp.name}: {correct}/{total} correct  →  {acc:.2%}")

    # Save detailed CSV
    csv_out = out_dir / "detailed_predictions.csv"
    with open(csv_out, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file","conversationId","expected","pred","pred_text","correct"])
        writer.writeheader()
        for r in overall_rows:
            writer.writerow(r)
    print(f"Saved detailed predictions to {csv_out}")

    # Metrics and confusion matrix
    # Filter to only valid predictions in LABELS for confusion matrix
    filtered_pairs = [(t,p) for (t,p) in zip(y_true,y_pred) if p in LABELS and t in LABELS]
    if len(filtered_pairs) == 0:
        print("No valid predictions found (labels missing). Exiting without confusion matrix.")
        summary = {"per_file": per_file_stats, "overall": None}
        open(out_dir / "evaluation_results.json","w").write(json.dumps(summary, indent=2))
        return

    y_true_f, y_pred_f = zip(*filtered_pairs)
    cm = confusion_matrix(y_true_f, y_pred_f, labels=LABELS)
    report = classification_report(y_true_f, y_pred_f, labels=LABELS, zero_division=0, output_dict=True)
    acc_all = accuracy_score(list(y_true_f), list(y_pred_f))

    # Save summary json
    summary = {
        "per_file": per_file_stats,
        "overall": {"accuracy": acc_all, "classification_report": report, "labels": LABELS}
    }
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {out_dir / 'evaluation_results.json'}")
    print("Classification report (text):")
    print(json.dumps(report, indent=2))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

def discover_files(results_dir):
    p = Path(results_dir)
    candidates = list(p.glob("*categorized*.json"))
    # also match *_categorized_rule.json or *_categorized_processed.json
    candidates += list(p.glob("*categorized_*json"))
    # unique
    uniq = []
    seen = set()
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))
    return uniq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="Paths to specific categorized json files.")
    parser.add_argument("--results-dir", default="results/llava", help="Directory to auto-discover categorized files.")
    parser.add_argument("--out-dir", default="results/llava/evaluation_all", help="Where to write results.")
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = discover_files(args.results_dir)

    if not files:
        print("No files to evaluate. Provide --files or put categorized jsons under results-dir.")
        return

    evaluate_files(files, args.out_dir)

if __name__ == "__main__":
    main()
