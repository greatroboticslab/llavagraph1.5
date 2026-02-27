"""
Evaluate finetuned ViT (FFT images) on test set and save accuracy summary.
Adapted from classify_vit.py for HPC paths.
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

model_path   = "vit_output_fft"
test_data_dir = "split_dataset_fft/test"
output_json  = "vit_output_fft/vit_summary_fft.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViTForImageClassification.from_pretrained(model_path).to(device)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)
id2label = model.config.id2label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

results = {}
print("Starting classification...")

for label_name in sorted(os.listdir(test_data_dir)):
    folder = os.path.join(test_data_dir, label_name)
    if not os.path.isdir(folder):
        continue

    results[label_name] = {'total': 0, 'correct': 0, 'confidence_sum': 0.0}

    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_id = probs.argmax(dim=-1).item()
                    confidence = probs[0, pred_id].item()
                    pred_label = id2label[pred_id]

                results[label_name]['total'] += 1
                results[label_name]['confidence_sum'] += confidence
                if pred_label.lower() == label_name.lower():
                    results[label_name]['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

summary = {}
total_correct = 0
total_samples = 0
for cls, stats in results.items():
    total = stats['total']
    correct = stats['correct']
    acc = round(correct / total * 100, 2) if total > 0 else 0
    avg_conf = round(stats['confidence_sum'] / total, 4) if total > 0 else 0
    summary[cls] = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": acc,
        "average_confidence": avg_conf
    }
    total_correct += correct
    total_samples += total

overall = round(total_correct / total_samples * 100, 2) if total_samples > 0 else 0
summary["overall"] = {
    "total_samples": total_samples,
    "correct_predictions": total_correct,
    "accuracy_percent": overall
}

with open(output_json, "w") as f:
    json.dump(summary, f, indent=2)

print("-" * 40)
print("Classification complete!")
for cls, v in summary.items():
    print(f"  {cls}: {v['correct_predictions']}/{v['total_samples']} = {v['accuracy_percent']}%")
print(f"\nSummary saved to: {output_json}")
