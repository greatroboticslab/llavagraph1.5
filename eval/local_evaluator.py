# (same content as earlier local_evaluator.py I provided)
import json,os,sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def infer_label_from_filename(fn):
    fn = fn.lower()
    if "noisetrials" in fn or "noise" in fn:
        return "A"
    if "sinetrials" in fn or "sine" in fn:
        return "B"
    if "squaretrials" in fn or "square" in fn:
        return "C"
    return None

def evaluate(pred_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    preds = json.load(open(pred_file))
    y_true=[]
    y_pred=[]
    rows=[]
    for rec in preds:
        fname = rec.get("conversationId") or rec.get("image") or rec.get("image_file") or rec.get("image_name") or ""
        if not fname:
            fname = rec.get("image","")
        gt = infer_label_from_filename(fname)
        if gt is None:
            continue
        p = rec.get("response","").strip()
        if p=="":
            pred=None
        else:
            pred=p[0]
        if pred is None:
            continue
        y_true.append(gt)
        y_pred.append(pred)
        rows.append({"file": fname, "gt": gt, "pred": pred})
    if len(y_true)==0:
        print("No ground truth labels found. Check filename patterns and folder links.")
        return
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.2f}%".format(acc*100))
    print("\nClassification Report:")
    print(classification_report(y_true,y_pred,labels=["A","B","C"],zero_division=0))
    cm = confusion_matrix(y_true,y_pred,labels=["A","B","C"])
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["A","B","C"]); ax.set_yticklabels(["A","B","C"])
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j,i, int(cm[i,j]), ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_matrix.png"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"detailed_predictions.csv"), index=False)
    json.dump({"accuracy": acc, "confusion_matrix": cm.tolist()}, open(os.path.join(out_dir,"evaluation_summary.json"),"w"), indent=2)
    print("Saved outputs to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred-file", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    evaluate(args.pred_file, args.output_dir)
