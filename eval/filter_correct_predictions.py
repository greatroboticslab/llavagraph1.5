import os
import json
import shutil

# === Paths ===
input_json = "/data/ilminur/LLaVA/eval/results/llava/squareWave_fixed.json"
output_json = "/data/ilminur/LLaVA/eval/results/llava/squarewave_50.json"
image_base_dir = "/data/ilminur/LLaVA/eval/images/squareWave"
output_image_dir = "/data/ilminur/LLaVA/eval/correct_images_50_square"

os.makedirs(output_image_dir, exist_ok=True)

# === Load JSON ===
with open(input_json, "r") as f:
    data = json.load(f)

# === Filter for correct predictions ===
# Match "C) Square wave" regardless of case
correct_entries = [
    entry for entry in data
    if entry.get("prediction", "").strip().lower() == "c) square wave"
]

print(f"Found {len(correct_entries)} entries predicted as 'C) Square wave'")

# === Copy images (up to 50) ===
selected_entries = []
for entry in correct_entries:
    img_name = entry.get("image")
    src_path = os.path.join(image_base_dir, img_name)
    dst_path = os.path.join(output_image_dir, img_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        selected_entries.append(entry)

    if len(selected_entries) >= 50:
        break

# === Save the new JSON ===
with open(output_json, "w") as f:
    json.dump(selected_entries, f, indent=2)

print(f"✅ Copied {len(selected_entries)} correct images to {output_image_dir}")
print(f"✅ Updated JSON saved to {output_json}")




#import os
#import json
#import shutil

# === Paths ===
#input_json = "/data/ilminur/LLaVA/eval/results/llava/sineWave_fixed.json"  
#output_json = "/data/ilminur/LLaVA/eval/results/llava/sineWave_correct.json"
#image_base_dir = "/data/ilminur/LLaVA/eval/images/sineWave"
#output_image_dir = "/data/ilminur/LLaVA/eval/correct_imagesSine"

#os.makedirs(output_image_dir, exist_ok=True)

# === Load JSON ===
#with open(input_json, "r") as f:
    #data = json.load(f)

# === Filter for correct predictions ===
#correct_entries = [
    #entry for entry in data
    #if "sine" in entry.get("prediction", "").lower()
#]

# === Keep only entries with existing images until we have 50 ===
#selected_entries = []
#for entry in correct_entries:
    #img_name = entry.get("image")
    #
    #if os.path.exists(src_path):
        #shutil.copy(src_path, os.path.join(output_image_dir, img_name))
        #selected_entries.append(entry)
   # if len(selected_entries) == 50:
       # break

# === Save updated JSON ===
#with open(output_json, "w") as f:
    #json.dump(selected_entries, f, indent=2)

#print(f"✅ Copied {len(selected_entries)} correct images to {output_image_dir}")
#print(f"✅ Updated JSON saved to {output_json}")













#import os
#import json
#import shutil

#=== Step 1: Define paths ===
#input_json = "/data/ilminur/LLaVA/eval/results/llava/randomNoise_fixed.json"
#output_json = "/data/ilminur/LLaVA/eval/results/llava/randomnoise_correct.json"
#image_base_dir = "/data/ilminur/LLaVA/eval/images/randomNoise"
#output_image_dir = "/data/ilminur/LLaVA/eval/correct_images"

#=== Step 2: Create output image directory if not exists ===
#os.makedirs(output_image_dir, exist_ok=True)

#=== Step 3: Load the JSON file ===
#with open(input_json, "r") as f:
    #data = json.load(f)

#=== Step 4: Filter correct predictions ===
#correct_entries = [entry for entry in data if entry.get("prediction") == "A) Random noise"]

#=== Step 5: Save filtered results ===
#with open(output_json, "w") as f:
    #json.dump(correct_entries, f, indent=2)

#=== Step 6: Copy correct images to new folder ===
#for entry in correct_entries:
    #img_name = entry["image"]
   # src_path = os.path.join(image_base_dir, img_name)
    #dst_path = os.path.join(output_image_dir, img_name)
    #if os.path.exists(src_path):
       # shutil.copy(src_path, dst_path)
   # else:
        #print(f"⚠️ Image not found: {src_path}")

#print(f"✅ Done! Saved {len(correct_entries)} correct entries to {output_json}")
#print(f"✅ Copied images to {output_image_dir}")

