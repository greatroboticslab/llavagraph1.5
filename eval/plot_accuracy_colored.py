import json
import os
import matplotlib.pyplot as plt

# === CONFIG ===
accuracy_json = "results/correct.json/accuracy_results.json"
output_png = "results/correct.json/accuracy_bar_chart_colored.png"
os.makedirs(os.path.dirname(output_png), exist_ok=True)

# === Load accuracy results ===
if not os.path.exists(accuracy_json):
    raise FileNotFoundError(f"Accuracy JSON not found: {accuracy_json}\nPlease check the path.")

with open(accuracy_json, "r") as f:
    data = json.load(f)

# === Extract dataset names and accuracy ===
labels = []
accuracies = []

for key, info in data.items():
    name = os.path.basename(key).replace("_correct.json", "").replace(".json", "")
    acc = info.get("accuracy_percent", info.get("accuracy", 0))
    labels.append(name)
    accuracies.append(float(acc))

# === Plot ===
plt.figure(figsize=(8, 6))

# Colors for each bar
colors = ['tab:blue', 'tab:orange', 'tab:green']  # you can add more if needed

bars = plt.bar(labels, accuracies, color=colors[:len(labels)])

# Add accuracy % on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig(output_png, dpi=300)
plt.show()

print(f"✅ Saved chart to {output_png}")

