from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from pathlib import Path
import sys

rng = np.random.default_rng(seed=42)
    
# Configuration
NUM_OF_IMAGES_PER_TYPE = 5
BASE_DATA_FOLDER = "/Users/ilminurablikim/Desktop/LLaVA_ilminur/training_data/Old_data"
OUTPUT_BASE = "SyntheticImages"

# Define the input folders to process
INPUT_FOLDERS = ["RandomNoise", "SineWave", "SquareWave"]

# Create output directories for each type
Path(f"{OUTPUT_BASE}/RandomNoise").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_BASE}/SineWave").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_BASE}/SquareWave").mkdir(parents=True, exist_ok=True)
    
# Time array
t = np.array(range(256))
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

# Process each input folder
for input_folder in INPUT_FOLDERS:
    input_folder_path = os.path.join(BASE_DATA_FOLDER, input_folder)
    
    print(f"Processing folder: {input_folder_path}")
    
    # Check if folder exists
    if not os.path.exists(input_folder_path):
        print(f"Warning: Folder {input_folder_path} does not exist, skipping...")
        continue

    # Get all image files from the input folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    base_images = []

    try:
        all_files = os.listdir(input_folder_path)
        print(f"Found {len(all_files)} files in folder")
        
        for file in all_files:
            file_path = os.path.join(input_folder_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    base_images.append(file_path)
                    print(f"  - Image: {file}")
    except Exception as e:
        print(f"Error reading folder: {e}")
        continue

    print(f"Total images found: {len(base_images)}")

    if len(base_images) == 0:
        print("No image files found in this folder, skipping...")
        continue

    # Process each base image in the current folder
    for base_idx, base_image_path in enumerate(base_images):
        print(f"\nProcessing base image: {os.path.basename(base_image_path)} ({base_idx+1}/{len(base_images)}) from {input_folder}")
        
        try:
            baseImage = Image.open(base_image_path)
            base_name = Path(base_image_path).stem  # Get filename without extension
            
            # Generate random noise images for each base image
            for i in range(NUM_OF_IMAGES_PER_TYPE):
                fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

                randomNoise = rng.uniform(low=-1, high=1, size=256)

                plt.plot(t, randomNoise, color="#b43ed1")
                plt.axis("off")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                chart = Image.open("temp.png")
                new_image = baseImage.copy()
                new_image.paste(chart, (80, 200))
                new_image.save(f"{OUTPUT_BASE}/RandomNoise/{input_folder}_{base_name}_rand{i}.png")
                chart.close()

            print(f"  - Generated {NUM_OF_IMAGES_PER_TYPE} random noise images")

            # Generate sine wave images for each base image
            for i in range(NUM_OF_IMAGES_PER_TYPE):
                fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

                sineWave = np.sin(rng.uniform(0.05, 0.2) * t)

                plt.plot(t, sineWave, color="#b43ed1")
                plt.axis("off")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                chart = Image.open("temp.png")
                new_image = baseImage.copy()
                new_image.paste(chart, (80, 200))
                new_image.save(f"{OUTPUT_BASE}/SineWave/{input_folder}_{base_name}_sine{i}.png")
                chart.close()

            print(f"  - Generated {NUM_OF_IMAGES_PER_TYPE} sine wave images")
            
            # Generate square wave images for each base image
            for i in range(NUM_OF_IMAGES_PER_TYPE):
                fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

                squareWave = signal.square(rng.uniform(0.05, 0.2) * t)

                plt.plot(t, squareWave, color="#b43ed1")
                plt.axis("off")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                chart = Image.open("temp.png")
                new_image = baseImage.copy()
                new_image.paste(chart, (80, 200))
                new_image.save(f"{OUTPUT_BASE}/SquareWave/{input_folder}_{base_name}_square{i}.png")
                chart.close()

            print(f"  - Generated {NUM_OF_IMAGES_PER_TYPE} square wave images")
            
            baseImage.close()
                
        except Exception as e:
            print(f"Error processing image {base_image_path}: {e}")
            continue

# Clean up temporary file
if os.path.exists("temp.png"):
    os.remove("temp.png")

# Summary
print(f"\n=== PROCESSING COMPLETED ===")
print(f"Processed folders: {INPUT_FOLDERS}")
print(f"Output saved in: {OUTPUT_BASE}")
print(f"Each input image generated {NUM_OF_IMAGES_PER_TYPE} synthetic images of each type")