"""
Automated Graph Description Generator using Ollama
100% Free, runs locally on your machine
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import subprocess
import sys


def check_ollama_status():
    """
    Check if Ollama is properly installed and running
    Returns: True if ready to use, False otherwise
    """
    print("Checking Ollama installation...\n")

    # Check if ollama command exists
    try:
        result = subprocess.run(['which', 'ollama'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama found at: {result.stdout.strip()}")
        else:
            print("✗ Ollama not found!")
            print("\nInstallation instructions:")
            print("  macOS: brew install ollama")
            print("  Or visit: https://ollama.com/download")
            return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

    # Check if ollama service is running
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("✓ Ollama service is running")

            # List available models
            models = response.json().get('models', [])
            if models:
                print(f"✓ Available models: {[m['name'] for m in models]}")
                return True
            else:
                print("\n✗ No models found!")
                print("  Please run: ollama pull llama3.2")
                return False
        else:
            print("✗ Ollama service not responding")
            return False
    except Exception as e:
        print("✗ Ollama service not running!")
        print(f"  Error: {e}")
        print("\nPlease start Ollama in another terminal:")
        print("  ollama serve")
        return False


def install_ollama_package():
    """Install the ollama Python package if not already installed"""
    try:
        import ollama
        print("✓ ollama Python package is installed\n")
        return True
    except ImportError:
        print("Installing ollama Python package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            print("✓ ollama package installed successfully\n")
            return True
        except Exception as e:
            print(f"✗ Failed to install ollama: {e}")
            print("  Please run manually: pip install ollama")
            return False


class GraphDescriptionGenerator:
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the generator with Ollama

        Args:
            model_name: Name of the Ollama model to use (default: llama3.2)
        """
        try:
            import ollama
            self.client = ollama
            self.model = model_name
            print(f"✓ Initialized with model: {model_name}\n")
        except ImportError:
            raise ImportError("Please install: pip install ollama")

    def generate_variations(self, base_sentence: str, num_variations: int = 50,
                            additional_constraints: str = "") -> List[str]:
        """
        Generate multiple variations of a base sentence using Ollama

        Args:
            base_sentence: The original sentence to rewrite
            num_variations: Number of variations to generate
            additional_constraints: Additional instructions for the model

        Returns:
            List of generated sentence variations
        """

        prompt = f"""Rewrite this sentence {num_variations} times: {base_sentence}

Use technical vocabulary as appropriate, but do not reference statistics, linearity, or any mathematical concepts; only rewrite the sentence.
{additional_constraints}

Return ONLY the rewritten sentences, numbered 1-{num_variations}, with no additional commentary."""

        try:
            print(f"   Generating variations...")
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"   ✓ Received response from Ollama")

            response_text = response['message']['content']

            # Parse the numbered variations from response
            variations = []
            for line in response_text.split('\n'):
                line = line.strip()
                # Look for numbered lines like "1. " or "1) "
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering prefix
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    if cleaned and len(cleaned) > 20:  # Filter out very short lines
                        variations.append(cleaned)

            print(f"   ✓ Parsed {len(variations)} variations")
            return variations[:num_variations]

        except Exception as e:
            print(f"   ✗ Error generating variations: {e}")
            return []

    def generate_all_descriptions(self, output_dir: str = "textData") -> Dict[str, str]:
        """
        Generate all 9 types of graph descriptions and save to text files

        Args:
            output_dir: Directory to save the generated text files

        Returns:
            Dictionary mapping file paths to their full paths
        """

        # Define the 9 description templates for different graph types
        templates = {
            "random/random-continuous.txt": {
                "sentence": "While the graph is continuous, it has very sudden changes in values.",
                "constraints": ""
            },
            "random/random-randomness.txt": {
                "sentence": "The graph shows a random pattern, with no definable pattern and several random points.",
                "constraints": ""
            },
            "random/random-square.txt": {
                "sentence": "While the graph does not have sharp corners, it does have significant random changes and is not smooth.",
                "constraints": ""
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
                "constraints": ""
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
                "constraints": ""
            }
        }

        # Create output directory structure
        output_path = Path(output_dir)
        for subdir in ["random", "sine", "square"]:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        results = {}
        print("=" * 70)
        print("Generating graph descriptions with Ollama")
        print("=" * 70)

        # Generate variations for each template
        for i, (filepath, template_info) in enumerate(templates.items(), 1):
            full_path = output_path / filepath
            print(f"\n[{i}/9] {filepath}")
            print(f"   Template: {template_info['sentence'][:60]}...")

            variations = self.generate_variations(
                template_info['sentence'],
                num_variations=50,
                additional_constraints=template_info['constraints']
            )

            # Save variations to file
            if variations:
                with open(full_path, 'w', encoding='utf-8') as f:
                    for variation in variations:
                        f.write(variation + '\n')
                results[filepath] = str(full_path)
                print(f"   ✓ Saved {len(variations)} variations to {full_path}")
            else:
                print(f"   ✗ No variations generated")

        print("\n" + "=" * 70)
        print("Generation complete!")
        print("=" * 70)
        return results

    def create_json_dataset(self,
                            image_folder: str,
                            output_file: str = "dataset.json",
                            text_data_dir: str = "textData") -> None:
        """
        Create a JSON dataset by combining graph images with generated descriptions

        Args:
            image_folder: Path to folder containing graph images
            output_file: Output JSON file name
            text_data_dir: Directory containing generated text descriptions
        """

        # Load all generated text variations
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

        # Define questions for each aspect
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

            # Create conversation for this image
            conversation = []
            for aspect in ["continuous", "randomness", "square"]:
                key = f"{wave_type}_{aspect}"
                if key in text_data and text_data[key]:
                    # Randomly select a variation from the generated descriptions
                    import random
                    answer = random.choice(text_data[key])

                    conversation.append({
                        "question": questions[aspect],
                        "answer": f"<s> {answer}</s>"
                    })

            if conversation:
                dataset.append({
                    "image": img_file.name,
                    "conversation": conversation
                })

        # Save dataset to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Dataset created: {output_file}")
        print(f"  Total samples: {len(dataset)}")


def main():
    """Main execution function"""

    print("=" * 70)
    print("Graph Description Generator - Ollama Edition")
    print("100% Free, Runs Locally")
    print("=" * 70)
    print()

    # Step 1: Check if Ollama is ready
    if not check_ollama_status():
        print("\n" + "=" * 70)
        print("Setup incomplete. Please fix the issues above and try again.")
        print("=" * 70)
        return

    # Step 2: Install Python package if needed
    if not install_ollama_package():
        return

    # Step 3: Generate descriptions
    try:
        # Initialize generator with llama3.2 model
        generator = GraphDescriptionGenerator(model_name="llama3.2")

        # Generate all text descriptions
        results = generator.generate_all_descriptions(output_dir="textData")

        print(f"\n✓ Generated {len(results)} files in textData/")

        # Step 4: Create JSON dataset (optional, uncomment when you have images)
        # generator.create_json_dataset(
        #     image_folder="path/to/your/images",
        #     output_file="dataset.json",
        #     text_data_dir="textData"
        # )

        print("\n" + "=" * 70)
        print("All done! Your graph descriptions are ready to use.")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure 'ollama serve' is running in another terminal")
        print("2. Run: ollama pull llama3.2")
        print("3. Test: ollama run llama3.2 'hello'")


if __name__ == "__main__":
    main()