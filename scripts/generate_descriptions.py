"""
Automated Graph Description Generator using Gemini API
This script generates varied descriptions for different graph types and characteristics.
"""

import google.generativeai as genai
import json
import os
from pathlib import Path
from typing import List, Dict
import time


class GraphDescriptionGenerator:
    def __init__(self, api_key: str = None):
        """Initialize with Google Gemini API key"""
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please provide GOOGLE_API_KEY")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def generate_variations(self, base_sentence: str, num_variations: int = 50,
                            additional_constraints: str = "") -> List[str]:
        """Generate N variations of a base sentence using Gemini"""

        prompt = f"""Rewrite this sentence {num_variations} times: {base_sentence}

Use technical vocabulary as appropriate, but do not reference statistics, linearity, or any mathematical concepts; only rewrite the sentence.
{additional_constraints}

Return ONLY the rewritten sentences, numbered 1-{num_variations}, with no additional commentary, preamble, or explanation."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Extract variations from response
            variations = []

            for line in response_text.split('\n'):
                line = line.strip()
                # Remove numbering like "1. " or "1) "
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove common prefixes
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    if cleaned:
                        variations.append(cleaned)

            # If we didn't get enough, try parsing differently
            if len(variations) < num_variations * 0.8:  # At least 80% of requested
                # Split by newlines and filter empty lines
                all_lines = [l.strip() for l in response_text.split('\n') if l.strip()]
                variations = []
                for line in all_lines:
                    # More aggressive cleaning
                    cleaned = line
                    # Remove numbering
                    if cleaned and cleaned[0].isdigit():
                        parts = cleaned.split('.', 1)
                        if len(parts) > 1:
                            cleaned = parts[1].strip()
                        else:
                            parts = cleaned.split(')', 1)
                            if len(parts) > 1:
                                cleaned = parts[1].strip()

                    if cleaned and len(cleaned) > 20:  # Reasonable sentence length
                        variations.append(cleaned)

            return variations[:num_variations]  # Ensure we don't exceed requested amount

        except Exception as e:
            print(f"Error generating variations: {e}")
            return []

    def generate_all_descriptions(self, output_dir: str = "textData") -> Dict[str, str]:
        """Generate all 9 types of descriptions and save to files"""

        # Define the 9 description templates (from the document you provided)
        templates = {
            # Random wave descriptions
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

            # Sine wave descriptions
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

            # Square wave descriptions
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

        # Create output directory structure
        output_path = Path(output_dir)
        for subdir in ["random", "sine", "square"]:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        results = {}

        print("Starting automated description generation with Gemini...\n")

        for filepath, template_info in templates.items():
            full_path = output_path / filepath
            print(f"Generating: {filepath}")
            print(f"  Template: {template_info['sentence']}")

            # Generate variations
            variations = self.generate_variations(
                template_info['sentence'],
                num_variations=50,
                additional_constraints=template_info['constraints']
            )

            # Save to file
            with open(full_path, 'w', encoding='utf-8') as f:
                for variation in variations:
                    f.write(variation + '\n')

            results[filepath] = str(full_path)
            print(f"  ✓ Generated {len(variations)} variations\n")

            # Small delay to avoid rate limiting
            time.sleep(1)

        print("All descriptions generated successfully!")
        return results

    def create_json_dataset(self,
                            image_folder: str,
                            output_file: str = "dataset.json",
                            text_data_dir: str = "textData") -> None:
        """Create JSON dataset by combining images with generated descriptions"""

        # Load all text variations
        text_data = {}
        text_path = Path(text_data_dir)

        for wave_type in ["random", "sine", "square"]:
            for aspect in ["continuous", "randomness", "square"]:
                file_path = text_path / wave_type / f"{wave_type}-{aspect}.txt"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_data[f"{wave_type}_{aspect}"] = [
                            line.strip() for line in f if line.strip()
                        ]
                else:
                    print(f"Warning: {file_path} not found")

        # Question templates
        questions = {
            "continuous": "Is the line shown in the graph continuous? Describe the line.",
            "randomness": "Does the graph contain any random points?",
            "square": "Does the graph contain sharp corners?"
        }

        # Process images and create dataset
        dataset = []
        image_path = Path(image_folder)

        if not image_path.exists():
            print(f"Warning: Image folder {image_folder} not found")
            return

        for img_file in sorted(image_path.glob("*.png")):
            # Determine wave type from filename
            wave_type = None
            filename_lower = img_file.name.lower()

            if "square" in filename_lower:
                wave_type = "square"
            elif "sine" in filename_lower:
                wave_type = "sine"
            elif "random" in filename_lower:
                wave_type = "random"
            else:
                print(f"Skipping {img_file.name} - cannot determine wave type")
                continue

            # Create conversation
            conversation = []
            for aspect in ["continuous", "randomness", "square"]:
                key = f"{wave_type}_{aspect}"
                if key in text_data and text_data[key]:
                    # Randomly select a variation
                    import random
                    answer = random.choice(text_data[key])

                    conversation.append({
                        "question": questions[aspect],
                        "answer": f"<s> {answer}</s>"
                    })

            if conversation:  # Only add if we have conversations
                dataset.append({
                    "image": img_file.name,
                    "conversation": conversation
                })

        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Dataset cråeated: {output_file}")
        print(f"  Total samples: {len(dataset)}")


def main():
    """Main execution function"""

    # Initialize generator
    # Make sure to set GOOGLE_API_KEY environment variable
    print("=" * 60)
    print("Graph Description Generator - Powered by Gemini")
    print("=" * 60)

    try:
        generator = GraphDescriptionGenerator()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\nOr get one at: https://makersuite.google.com/app/apikey")
        return

    # Step 1: Generate all text descriptions
    print("\n" + "=" * 60)
    print("STEP 1: Generating text descriptions")
    print("=" * 60)
    results = generator.generate_all_descriptions(output_dir="textData")

    # Step 2: Create JSON dataset (optional, if you have images)
    print("\n" + "=" * 60)
    print("STEP 2: Creating JSON dataset")
    print("=" * 60)

    # Uncomment and modify the following lines when you have images:
    # generator.create_json_dataset(
    #     image_folder="path/to/your/images",
    #     output_file="dataset.json",
    #     text_data_dir="textData"
    # )

    print("\n✓ To create the JSON dataset, uncomment the code in main()")
    print("  and provide the path to your image folder")

    print("\n" + "=" * 60)
    print("AUTOMATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()