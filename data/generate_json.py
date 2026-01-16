#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

# Updated paths
LLAVAGRAPH_DIR = "/<your-projects-path>/llavagraph"
TRAININGDATA_DIR = f"{LLAVAGRAPH_DIR}/data"
TEXTDATA_DIR = f"{LLAVAGRAPH_DIR}/data/textData"

# Training image source
TRAIN_IMAGE_DIR = f"{TRAININGDATA_DIR}/trainData"

# Test image source
TEST_IMAGE_DIR = f"{TRAININGDATA_DIR}/testData"

# Your text answer files
TEXT_FILES = {
    "RandomNoise": {
        "continuous": f"{TEXTDATA_DIR}/random/random-continuous.txt",
        "randomness": f"{TEXTDATA_DIR}/random/random-randomness.txt",
        "corners": f"{TEXTDATA_DIR}/random/random-square.txt"
    },
    "SineWave": {
        "continuous": f"{TEXTDATA_DIR}/sine/sine-continuous.txt",
        "randomness": f"{TEXTDATA_DIR}/sine/sine-randomness.txt",
        "corners": f"{TEXTDATA_DIR}/sine/sine-square.txt"
    },
    "SquareWave": {
        "continuous": f"{TEXTDATA_DIR}/square/square-continuous.txt",
        "randomness": f"{TEXTDATA_DIR}/square/square-randomness.txt",
        "corners": f"{TEXTDATA_DIR}/square/square-square.txt"
    }
}


def load_text_answers(file_path):
    """Load text answer files"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    print(f"Warning: {file_path} does not exist, using default answer")
    return ["Default answer."]


def collect_images(image_dir, prefix=""):
    """Collect images from specified directory with subdirectories for each category"""
    all_images = []
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']

    base_path = Path(image_dir)
    if not base_path.exists():
        print(f"Warning: Directory {image_dir} does not exist")
        return all_images

    # Categories are subdirectories: RandomNoise, SineWave, SquareWave
    for category in ["RandomNoise", "SineWave", "SquareWave"]:
        category_dir = base_path / category
        if category_dir.exists():
            for ext in image_extensions:
                images = list(category_dir.glob(ext))
                for img_path in images:
                    # Store relative path from TRAININGDATA_DIR
                    rel_path = img_path.relative_to(TRAININGDATA_DIR)
                    all_images.append((category, img_path.name, str(rel_path)))
        else:
            print(f"Warning: Category directory {category_dir} does not exist")

    print(f"Found {len(all_images)} images for {prefix} in {image_dir}")
    return all_images


def generate_conversation(category):
    """Generate LLaVA conversation format"""
    answers = TEXT_FILES.get(category, TEXT_FILES["RandomNoise"])
    return [
        {"from": "human", "value": "<image>Is the line shown in the graph continuous? Describe the line."},
        {"from": "gpt", "value": random.choice(load_text_answers(answers["continuous"]))},
        {"from": "human", "value": "Does the graph contain any random points?"},
        {"from": "gpt", "value": random.choice(load_text_answers(answers["randomness"]))},
        {"from": "human", "value": "Does the graph contain sharp corners?"},
        {"from": "gpt", "value": random.choice(load_text_answers(answers["corners"]))}
    ]


def main():
    print("Generating training data from trainData directory...")

    # Training data: from trainData directory
    train_images = collect_images(TRAIN_IMAGE_DIR, "training")
    if not train_images:
        print("Error: No training images found. Please check the directory structure.")
        return

    random.shuffle(train_images)

    train_data = []
    for i, (category, img_name, rel_path) in enumerate(train_images):
        train_data.append({
            "id": i + 1,
            "image": str(rel_path),
            "conversations": generate_conversation(category)
        })

    # Test data: from testData directory
    print("Generating test data from testData directory...")
    test_images = collect_images(TEST_IMAGE_DIR, "testing")
    if not test_images:
        print("Error: No test images found. Please check the directory structure.")
        return

    random.shuffle(test_images)

    test_data = []
    for i, (category, img_name, rel_path) in enumerate(test_images):
        test_data.append({
            "id": len(train_data) + i + 1,
            "image": str(rel_path),
            "conversations": generate_conversation(category)
        })

    # Save JSON files to TRAININGDATA_DIR
    output_dir = TRAININGDATA_DIR
    with open(f"{output_dir}/trainingData.json", "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(f"{output_dir}/testData.json", "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("Complete!")
    print(f"Training data: {len(train_data)} samples -> {output_dir}/trainingData.json")
    print(f"Test data: {len(test_data)} samples -> {output_dir}/testData.json")



if __name__ == "__main__":
    main()
