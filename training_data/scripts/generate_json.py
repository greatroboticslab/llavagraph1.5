#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

# Your exact paths
LLAVAGRAPH_DIR = "/projects/ya4v/llavagraph"
TRAININGDATA_DIR = "/projects/ya4v/trainingdata"
TEXTDATA_DIR = f"{LLAVAGRAPH_DIR}/data/textData"

# Training image sources (original + synthetic input)
TRAIN_IMAGE_DIRS = [
    f"{TRAININGDATA_DIR}/data/original/input",
    f"{TRAININGDATA_DIR}/data/synthetic/input"
]

# Test image source (final processed synthetic output)
TEST_IMAGE_DIR = f"{TRAININGDATA_DIR}/data/synthetic/output/V3/final"

# Your text answer files (9 files total)
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


def collect_images(image_dirs, prefix=""):
    """Collect images from specified directories"""
    all_images = []
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']

    for base_dir in image_dirs:
        base_path = Path(base_dir)
        if base_path.exists():
            for category in ["RandomNoise", "SineWave", "SquareWave"]:
                category_dir = base_path / category
                if category_dir.exists():
                    for ext in image_extensions:
                        images = list(category_dir.glob(ext))
                        for img_path in images:
                            rel_path = img_path.relative_to(TRAININGDATA_DIR)
                            all_images.append((category, img_path.name, str(rel_path)))

    print(f"Found {len(all_images)} images for {prefix}")
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
    print("Generating training data from original/input + synthetic/input...")

    # Training data: original + synthetic input folders
    train_images = collect_images(TRAIN_IMAGE_DIRS, "training")
    random.shuffle(train_images)

    train_data = []
    for i, (category, img_name, rel_path) in enumerate(train_images):
        train_data.append({
            "id": i + 1,
            "image": str(rel_path),
            "conversations": generate_conversation(category)
        })

    # Test data: synthetic/output/V3/final folder
    print("Generating test data from synthetic/output/V3/final...")
    test_images = collect_images([TEST_IMAGE_DIR], "testing")
    random.shuffle(test_images)

    test_data = []
    for i, (category, img_name, rel_path) in enumerate(test_images):
        test_data.append({
            "id": len(train_data) + i + 1,
            "image": str(rel_path),
            "conversations": generate_conversation(category)
        })

    # Save JSON files
    with open(f"{TRAININGDATA_DIR}/trainingData.json", "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(f"{TRAININGDATA_DIR}/testData.json", "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("Complete!")
    print(f"Training data: {len(train_data)} samples -> {TRAININGDATA_DIR}/trainingData.json")
    print(f"Test data: {len(test_data)} samples -> {TRAININGDATA_DIR}/testData.json")
    print("Images remain in their original locations")


if __name__ == "__main__":
    main()
