import os, torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from collections import defaultdict

model_path = "vit_output_fft"
test_dir   = "split_dataset_fft/test"

model = ViTForImageClassification.from_pretrained(model_path)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)
id2label = model.config.id2label

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

results = defaultdict(lambda: {"total": 0, "correct": 0})

for cls in sorted(os.listdir(test_dir)):
    folder = os.path.join(test_dir, cls)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = id2label[model(t).logits.argmax(-1).item()]
        results[cls]["total"] += 1
        if pred.lower() == cls.lower():
            results[cls]["correct"] += 1

total_c, total_t = 0, 0
print("Class       Correct  Total   Acc")
print("-" * 35)
for cls in sorted(results):
    c, t = results[cls]["correct"], results[cls]["total"]
    total_c += c
    total_t += t
    print("%-10s  %3d / %3d  = %5.1f%%" % (cls, c, t, c / t * 100))
print("-" * 35)
print("%-10s  %3d / %3d  = %5.1f%%" % ("overall", total_c, total_t, total_c / total_t * 100))
