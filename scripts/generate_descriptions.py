#!/usr/bin/env python3
"""
Complete Graph Description Generator with JSON Dataset Creation
Generates text descriptions using Gemini API and creates JSON dataset for LLaVA training
"""

import os
import json
from pathlib import Path
import time
import requests
import random


class GraphDescriptionGenerator:
    def __init__(self, api_key: str = None):
        """Initialize with Google Gemini API key"""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

    def generate_variations(self, base_sentence: str, num_variations: int = 50,
                            additional_constraints: str = "") -> list:
        """Generate N variations using direct API call"""

        prompt = f"""Rewrite this sentence {num_variations} times: {base_sentence}

Use technical vocabulary as appropriate, but do not reference statistics, linearity, or any mathematical concepts; only rewrite the sentence.
{additional_constraints}

Return ONLY the rewritten sentences, numbered 1-{num_variations}, with no additional commentary, preamble, or explanation."""

        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]

                variations = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        cleaned = line.lstrip('0123456789.-) ').strip()
                        if cleaned:
                            variations.append(cleaned)

                return variations[:num_variations]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            print(f"Error generating variations: {e}")
            return []

    def generate_all_descriptions(self, output_dir: str = "textData") -> dict:
        """Generate all 9 types of descriptions and save to files"""
        templates = {
            "random/random-continuous.txt": {
                "sentence": "While the graph is continuous, it has very sudden changes in values.",
                "constraints": ""
            },
            "random/random-randomness.txt": {
                "sentence": "The graph shows a random pattern, with no definable pattern and several random points.",
                "constraints": "Be sure to reference either the sudden drops in values or corners."
            },
            "random/random-square.txt": {
                "sentence": "While the graph does not have sharp corners, it does have significant random changes and is not smooth.",
                "constraints": "Do not reference continuous or any mathematical concepts."
            },
            "sine/sine-continuous.txt": {
                "sentence": "This graph does not have any randomness, with the line being smooth and continuous.",
                "constraints": "Be sure to mention the line is continuous and not random."
            },
            "sine/sine-randomness.txt": {
                "sentence": "This wave is periodic, smooth, and oscillates between a maximum and minimum value with equal positive and negative peaks.",
                "constraints": ""
            },
            "sine/sine-square.txt": {
                "sentence": "The graph did not have sudden drops in values, with no abrupt corners visible.",
                "constraints": "Be sure to reference either the sudden drops in values or corners."
            },
            "square/square-continuous.txt": {
                "sentence": "This graph does not have any randomness, with the line changing between two distinct levels at regular intervals.",
                "constraints": "Be sure to mention the line is continuous and not random."
            },
            "square/square-randomness.txt": {
                "sentence": "While this wave does not have random points, it is not continuous, with sudden breaks to equal positive and negative peaks.",
                "constraints": ""
            },
            "square/square-square.txt": {
                "sentence": "Although the graph is not random, it does have sudden drops in values, with abrupt corners visible.",
                "constraints": "Be sure to reference either the sudden drops in values or corners."
            }
        }

        output_path = Path(output_dir)
        for subdir in ["random", "sine", "square"]:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        results = {}
        print("Starting automated description generation with Gemini...\n")

        for filepath, template_info in templates.items():
            full_path = output_path / filepath
            print(f"Generating: {filepath}")
            print(f"  Template: {template_info['sentence']}")

            variations = self.generate_variations(
                template_info['sentence'],
                num_variations=50,
                additional_constraints=template_info['constraints']
            )

            with open(full_path, 'w', encoding='utf-8') as f:
                for variation in variations:
                    f.write(variation + '\n')

            results[filepath] = str(full_path)
            print(f"  Generated {len(variations)} variations\n")
            time.sleep(1)

        print("All descriptions generated successfully!")
        return results

    def load_text_answers(self, file_path):
        """Load text answer files"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        print(f"Warning: {file_path} does not exist, using default answer")
        return ["Default answer."]

    def collect_images(self, image_dir, text_data_dir="textData"):
        """Collect images from specified directory with subdirectories for each category"""
        all_images = []
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
        
        base_path = Path(image_dir)
        if not base_path.exists():
            print(f"Warning: Directory {image_dir} does not exist")
            return all_images

        category_mapping = {
            "RandomNoise": "random",
            "SineWave": "sine", 
            "SquareWave": "square"
        }
        
        for category, text_subdir in category_mapping.items():
            category_dir = base_path / category
            if category_dir.exists():
                for ext in image_extensions:
                    images = list(category_dir.glob(ext))
                    for img_path in images:
                        all_images.append({
                            "category": category,
                            "text_category": text_subdir,
                            "image_path": str(img_path),
                            "image_name": img_path.name
                        })
            else:
                print(f"Warning: Category directory {category_dir} does not exist")

        print(f"Found {len(all_images)} images in {image_dir}")
        return all_images

    def generate_conversation(self, text_category, text_data_dir="textData"):
        """Generate LLaVA conversation format"""
        text_files = {
            "random": {
                "continuous": f"{text_data_dir}/random/random-continuous.txt",
                "randomness": f"{text_data_dir}/random/random-randomness.txt", 
                "corners": f"{text_data_dir}/random/random-square.txt"
            },
            "sine": {
                "continuous": f"{text_data_dir}/sine/sine-continuous.txt",
                "randomness": f"{text_data_dir}/sine/sine-randomness.txt",
                "corners": f"{text_data_dir}/sine/sine-square.txt"
            },
            "square": {
                "continuous": f"{text_data_dir}/square/square-continuous.txt",
                "randomness": f"{text_data_dir}/square/square-randomness.txt",
                "corners": f"{text_data_dir}/square/square-square.txt"
            }
        }
        
        answers = text_files.get(text_category, text_files["random"])
        
        return [
            {"from": "human", "value": "<image>Is the line shown in the graph continuous? Describe the line."},
            {"from": "gpt", "value": random.choice(self.load_text_answers(answers["continuous"]))},
            {"from": "human", "value": "Does the graph contain any random points?"},
            {"from": "gpt", "value": random.choice(self.load_text_answers(answers["randomness"]))},
            {"from": "human", "value": "Does the graph contain sharp corners?"},
            {"from": "gpt", "value": random.choice(self.load_text_answers(answers["corners"]))}
        ]

    def create_json_dataset(self,
                           train_image_dir: str,
                           test_image_dir: str = None,
                           output_dir: str = ".",
                           text_data_dir: str = "textData"):
        """
        Create JSON dataset by combining images with generated descriptions
        """
        
        print("\n" + "=" * 60)
        print("Creating JSON dataset for LLaVA training")
        print("=" * 60)
        
        # Training data
        print("\nCollecting training images...")
        train_images = self.collect_images(train_image_dir, text_data_dir)
        if not train_images:
            print(f"Error: No training images found in {train_image_dir}")
            print("Please check that the directory contains RandomNoise/, SineWave/, SquareWave/ subfolders")
            return
        
        random.shuffle(train_images)
        
        train_data = []
        for i, img_info in enumerate(train_images):
            train_data.append({
                "id": i + 1,
                "image": f"trainData/{img_info['category']}/{img_info['image_name']}",
                "conversations": self.generate_conversation(img_info['text_category'], text_data_dir)
            })
        
        # Test data (if provided)
        test_data = []
        if test_image_dir and os.path.exists(test_image_dir):
            print("\nCollecting test images...")
            test_images = self.collect_images(test_image_dir, text_data_dir)
            if test_images:
                random.shuffle(test_images)
                for i, img_info in enumerate(test_images):
                    test_data.append({
                        "id": len(train_data) + i + 1,
                        "image": f"testData/{img_info['category']}/{img_info['image_name']}",
                        "conversations": self.generate_conversation(img_info['text_category'], text_data_dir)
                    })
            else:
                print(f"Warning: No test images found in {test_image_dir}")
        else:
            print("\nNo test image directory provided or directory doesn't exist")
        
        # Save JSON files
        output_path = Path(output_dir)
        with open(output_path / "trainingData.json", "w", encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved: {output_path}/trainingData.json")
        print(f"Training samples: {len(train_data)}")
        
        if test_data:
            with open(output_path / "testData.json", "w", encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print(f"Test data saved: {output_path}/testData.json")
            print(f"Test samples: {len(test_data)}")
        
        print("\n" + "=" * 60)
        print("JSON dataset creation complete!")
        print("=" * 60)
        
        return {
            "training_data": train_data,
            "test_data": test_data
        }


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Complete Graph Description & Dataset Generator")
    print("=" * 60)
    
    # Configuration - Update these paths according to your setup
    BASE_DIR = "your_project_base_directory"  # Update this to your project base directory
    TRAIN_IMAGE_DIR = f"{BASE_DIR}/data/trainData"  # Should contain RandomNoise/, SineWave/, SquareWave/
    TEST_IMAGE_DIR = f"{BASE_DIR}/data/testData"    # Optional test data directory
    OUTPUT_DIR = f"{BASE_DIR}/data"                 # Where to save JSON files
    TEXT_DATA_DIR = "textData"                      # Where text descriptions are saved
    
    print(f"Configuration:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Training images: {TRAIN_IMAGE_DIR}")
    if os.path.exists(TEST_IMAGE_DIR):
        print(f"  Test images: {TEST_IMAGE_DIR}")
    else:
        print(f"  Test images: Not found (optional)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Text data directory: {TEXT_DATA_DIR}")
    print()
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        print("\nPlease set it with:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\nOr get one at: https://makersuite.google.com/app/apikey")
        return
    
    try:
        generator = GraphDescriptionGenerator()
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return
    
    # Step 1: Generate text descriptions (if not already generated)
    print("\n" + "=" * 60)
    print("STEP 1: Generating text descriptions")
    print("=" * 60)
    
    # Check if text data already exists
    text_data_exists = os.path.exists(TEXT_DATA_DIR) and any(
        os.path.exists(f"{TEXT_DATA_DIR}/{wave}/{wave}-{aspect}.txt")
        for wave in ["random", "sine", "square"]
        for aspect in ["continuous", "randomness", "square"]
    )
    
    if text_data_exists:
        print("Text data already exists, skipping generation")
        print(f"Using existing text data from: {TEXT_DATA_DIR}/")
    else:
        print("Generating new text descriptions...")
        try:
            results = generator.generate_all_descriptions(output_dir=TEXT_DATA_DIR)
            print("Text descriptions generated successfully!")
        except Exception as e:
            print(f"Error generating text: {e}")
            return
    
    # Step 2: Create JSON dataset
    print("\n" + "=" * 60)
    print("STEP 2: Creating JSON dataset")
    print("=" * 60)
    
    # Check if training image directory exists
    if not os.path.exists(TRAIN_IMAGE_DIR):
        print(f"Error: Training image directory not found: {TRAIN_IMAGE_DIR}")
        print("\nPlease check the path and ensure it contains:")
        print("  RandomNoise/ folder with .png images")
        print("  SineWave/ folder with .png images")
        print("  SquareWave/ folder with .png images")
        return
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create JSON dataset
        dataset = generator.create_json_dataset(
            train_image_dir=TRAIN_IMAGE_DIR,
            test_image_dir=TEST_IMAGE_DIR if os.path.exists(TEST_IMAGE_DIR) else None,
            output_dir=OUTPUT_DIR,
            text_data_dir=TEXT_DATA_DIR
        )
        
        print(f"\nDataset files created in: {OUTPUT_DIR}/")
        print(f"  1. trainingData.json - {len(dataset['training_data'])} training samples")
        print(f"  2. testData.json - {len(dataset['test_data'])} test samples")
        
    except Exception as e:
        print(f"Error creating JSON dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check that trainData/ contains RandomNoise/, SineWave/, SquareWave/ subfolders")
        print("2. Check that each subfolder contains .png images")
        print("3. Verify textData/ contains all 9 text files")
        return
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETE!")
    print("=" * 60)
    print("\nSummary of generated files:")
    print(f"1. Text descriptions: {TEXT_DATA_DIR}/")
    print(f"   - 9 files with 50 variations each")
    print(f"2. JSON datasets: {OUTPUT_DIR}/")
    train_image_count = len(generator.collect_images(TRAIN_IMAGE_DIR))
    print(f"   - trainingData.json: {train_image_count} samples")
    if os.path.exists(TEST_IMAGE_DIR):
        test_image_count = len(generator.collect_images(TEST_IMAGE_DIR))
        print(f"   - testData.json: {test_image_count} samples")
    print("\nReady for LLaVA training!")


if __name__ == "__main__":
    main()
