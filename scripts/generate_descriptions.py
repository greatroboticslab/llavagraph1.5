#!/usr/bin/env python3
"""
Automated Graph Description Generator using Gemini API
This script generates varied descriptions for different graph types and characteristics.
"""

import os
import json
from pathlib import Path
import time
import requests

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
            print(f"  ✓ Generated {len(variations)} variations\n")
            time.sleep(1)

        print("All descriptions generated successfully!")
        return results

def main():
    print("=" * 60)
    print("Graph Description Generator - Direct API Version")
    print("=" * 60)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        print("\nPlease set it with:")
        print("  export GOOGLE_API_KEY='AIzaSyCzcLZQJeCNx_PAs9meIufYQiwdpKEqjiE'")
        return
    
    try:
        generator = GraphDescriptionGenerator()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("STEP 1: Generating text descriptions")
    print("=" * 60)
    
    try:
        results = generator.generate_all_descriptions(output_dir="textData")
        print("✓ Generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
