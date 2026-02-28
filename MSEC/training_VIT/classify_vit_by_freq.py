"""
Per-frequency accuracy breakdown for a finetuned ViT model.
Works for both time-domain and FFT test sets.

Usage (local Mac):
    python classify_vit_by_freq.py \
        --model /path/to/vit_output_aug \
        --test  /path/to/split_dataset_aug/test \
        --out   vit_freq_breakdown.json

Usage (HPC, FFT model, after training finishes):
    python classify_vit_by_freq.py \
        --model vit_output_fft \
        --test  split_dataset_fft/test \
        --out   vit_output_fft/vit_freq_breakdown_fft.json
"""

import os
import re
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor


def extract_freq(filename):
    """Extract frequency label from filename, e.g. '100Hz', '1Hz', '1Hx'."""
    match = re.search(r'[-_](\d+H[zx])', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower().replace('hx', 'hz')
    return 'unknown'


def main(model_path, test_data_dir, output_json):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(model_path)
    id2label = model.config.id2label

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # results[category][freq] = {total, correct}
    results = {}

    for label_name in sorted(os.listdir(test_data_dir)):
        folder = os.path.join(test_data_dir, label_name)
        if not os.path.isdir(folder):
            continue

        results[label_name] = {}

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            freq = extract_freq(fname)
            if freq not in results[label_name]:
                results[label_name][freq] = {'total': 0, 'correct': 0}

            img_path = os.path.join(folder, fname)
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor).logits
                    pred_label = id2label[logits.argmax(dim=-1).item()]

                results[label_name][freq]['total'] += 1
                if pred_label.lower() == label_name.lower():
                    results[label_name][freq]['correct'] += 1
            except Exception as e:
                print(f"Error: {img_path}: {e}")

    # Build summary
    summary = {}
    print("\n" + "=" * 60)
    for cat, freq_dict in sorted(results.items()):
        summary[cat] = {}
        cat_total = cat_correct = 0
        print(f"\n{cat}:")
        for freq in sorted(freq_dict.keys(),
                           key=lambda x: int(re.sub(r'\D', '', x)) if re.search(r'\d', x) else 0):
            s = freq_dict[freq]
            acc = round(s['correct'] / s['total'] * 100, 1) if s['total'] else 0
            summary[cat][freq] = {
                'total': s['total'], 'correct': s['correct'], 'accuracy_pct': acc
            }
            print(f"  {freq:>8s}: {s['correct']:3d}/{s['total']:3d} = {acc:5.1f}%")
            cat_total += s['total']
            cat_correct += s['correct']
        cat_acc = round(cat_correct / cat_total * 100, 1) if cat_total else 0
        summary[cat]['_total'] = {
            'total': cat_total, 'correct': cat_correct, 'accuracy_pct': cat_acc
        }
        print(f"  {'TOTAL':>8s}: {cat_correct:3d}/{cat_total:3d} = {cat_acc:5.1f}%")

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to finetuned ViT model directory")
    parser.add_argument("--test",  required=True, help="Path to test data directory")
    parser.add_argument("--out",   required=True, help="Output JSON path")
    args = parser.parse_args()
    main(args.model, args.test, args.out)
