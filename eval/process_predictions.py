#!/usr/bin/env python3
import json, argparse, re, sys

def extract_pred(response):
    if not response:
        return "Unknown"
    r = response.lower()
    # check for explicit option like 'a)', 'option a', 'A)'
    m = re.search(r'\b([abc])\)', response, flags=re.IGNORECASE)
    if m:
        opt = m.group(1).lower()
        return {"a": "Random", "b": "Sine", "c": "Square"}.get(opt, "Unknown")
    # look for words
    if "random" in r or "noise" in r:
        return "Random"
    if "sine" in r:
        return "Sine"
    if "square" in r:
        return "Square"
    # fallback single-letter (like "A) Random noise")
    m2 = re.search(r'\b([abc])\b', r)
    if m2:
        opt = m2.group(1)
        return {"a": "Random", "b": "Sine", "c": "Square"}.get(opt, "Unknown")
    return "Unknown"

def infer_label_from_filename(fname):
    f = fname.lower()
    if "noise" in f:
        return "Random"
    if "sine" in f:
        return "Sine"
    if "square" in f:
        return "Square"
    # try patterns
    if "random" in f:
        return "Random"
    return "Unknown"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    processed = []
    for item in data:
        # try common keys
        fname = item.get("conversationId") or item.get("image") or item.get("id") or ""
        resp = item.get("response") or item.get("answer") or ""
        pred = extract_pred(resp)
        label = infer_label_from_filename(fname)
        item["pred"] = pred
        item["label"] = label
        processed.append(item)

    with open(args.output, "w") as f:
        json.dump(processed, f, indent=2)
    print("Wrote:", args.output)

if __name__ == "__main__":
    main()
