"""
Generate training data for FFT amplitude plots using Gemini 2.0 Flash.

Questions are designed to extract peak-based features for wave classification:
  Q1: Tallest peak height (noise < 60nm vs sine/square > 60nm)
  Q2: Amplitudes and frequencies of the two tallest peaks (ratio separates sine vs square)
  Q3: Decay shape after the dominant peak (sharp drop = sine, gradual decay = square)

Classification logic (applied by LLaMA):
  - Tallest peak < 60nm → noise
  - Second peak / tallest peak < 40% → sine
  - Second peak / tallest peak >= 50% → square
  - Q3 decay shape helps disambiguate 1Hz signals

Excludes 500Hz signals (indistinguishable sine/square at Nyquist).

Usage:
    export GOOGLE_API_KEY='your_key'
    python gemini_train.py \
        --image-dir /path/to/fft_augmented_nm \
        --output trainingData_FFT.json
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from google import genai
from google.genai import types


QUESTIONS = [
    "What is the approximate amplitude (in nm) of the tallest peak in this FFT spectrum? Is it above or below 60 nm?",
    "What are the approximate amplitudes (in nm) of the two tallest peaks in this FFT spectrum, and at what frequencies do they appear?",
    "Looking at the region around and after the dominant peak, does the signal drop sharply to near-zero within a narrow frequency range, or does it decay gradually across a wide range of frequencies?"
]


def should_exclude(filename):
    """Exclude 500Hz signals — sine and square are indistinguishable at Nyquist."""
    match = re.search(r'_500(?:hz|hx)', filename, re.IGNORECASE)
    return match is not None


class GeminiFFTAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API Key detected. "
                "Please execute: export GOOGLE_API_KEY='your_key'"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"

    def process_image(self, img_path, max_retries=5):
        """Send FFT plot to Gemini and get answers to four questions.
        Retries with exponential backoff on rate limit errors."""
        print(f"  Analyzing: {img_path.parent.name}/{img_path.name}", flush=True)

        prompt = f"""Analyze this FFT spectrum plot of a piezoelectric actuator displacement signal.
The y-axis shows amplitude in nanometers (nm) and the x-axis shows frequency in Hz.

Please provide a detailed, specific answer for each question:

1. {QUESTIONS[0]}
2. {QUESTIONS[1]}
3. {QUESTIONS[2]}

Format your response exactly as:
A1: [Your answer]
A2: [Your answer]
A3: [Your answer]
"""

        with open(img_path, "rb") as f:
            img_data = f.read()

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_data, mime_type="image/png")
                    ]
                )
                return self._parse_answers(response.text)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Extract retry delay from error if available
                    retry_match = re.search(r'retryDelay.*?(\d+)', error_str)
                    wait = int(retry_match.group(1)) + 5 if retry_match else (2 ** attempt) * 10
                    print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    Error: {e}", flush=True)
                    return None

        print(f"    Failed after {max_retries} retries", flush=True)
        return None

    def _parse_answers(self, text):
        """Split Gemini's response into three separate answers."""
        a1 = re.search(r"A1:(.*?)(?=A2:|$)", text, re.DOTALL)
        a2 = re.search(r"A2:(.*?)(?=A3:|$)", text, re.DOTALL)
        a3 = re.search(r"A3:(.*?)$", text, re.DOTALL)

        return [
            a1.group(1).strip() if a1 else "Data missing",
            a2.group(1).strip() if a2 else "Data missing",
            a3.group(1).strip() if a3 else "Data missing"
        ]

    def process_directory(self, image_dir, output_file):
        """Process all FFT plots and generate training JSON.
        Resumes from existing output file if present."""
        image_dir = Path(image_dir)
        output_file = Path(output_file)

        # Resume from existing progress
        final_data = []
        existing_ids = set()
        if output_file.exists() and output_file.stat().st_size > 2:
            with open(output_file, "r") as f:
                final_data = json.load(f)
            existing_ids = {d["id"] for d in final_data}
            print(f"Resuming: {len(final_data)} entries already processed", flush=True)

        categories = ["noise", "sine", "square"]

        for cat in categories:
            cat_path = image_dir / cat
            if not cat_path.exists():
                print(f"Skipping: Folder not found {cat_path}")
                continue

            image_files = sorted(cat_path.glob("*.png"))

            # Filter out 500Hz files for sine/square
            if cat in ("sine", "square"):
                before = len(image_files)
                image_files = [f for f in image_files if not should_exclude(f.name)]
                excluded = before - len(image_files)
                if excluded:
                    print(f"  Excluded {excluded} files at 500Hz (Nyquist)")

            print(f"\nProcessing {cat}: {len(image_files)} images", flush=True)

            for idx, img_path in enumerate(image_files):
                entry_id = f"{cat}_{img_path.stem}"
                if entry_id in existing_ids:
                    continue

                answers = self.process_image(img_path)
                if answers:
                    # Image path relative to training folder
                    rel_image = f"train/{cat}/{img_path.name}"

                    entry = {
                        "id": f"{cat}_{img_path.stem}",
                        "image": rel_image,
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{QUESTIONS[0]}"},
                            {"from": "gpt", "value": answers[0]},
                            {"from": "human", "value": QUESTIONS[1]},
                            {"from": "gpt", "value": answers[1]},
                            {"from": "human", "value": QUESTIONS[2]},
                            {"from": "gpt", "value": answers[2]}
                        ]
                    }
                    final_data.append(entry)

                # Save progress every 30 images
                if (len(final_data) % 30 == 0) and final_data:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_data, f, indent=2, ensure_ascii=False)
                    print(f"  [Progress saved: {len(final_data)} entries]", flush=True)

                # Rate limit: free API allows ~15 requests/min
                time.sleep(4)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"\nComplete! Generated {len(final_data)} training entries.", flush=True)
        print(f"Output: {output_file}", flush=True)
        return final_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FFT training data using Gemini"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory with noise/sine/square subfolders of FFT PNGs"
    )
    parser.add_argument(
        "--output", type=str, default="trainingData_FFT.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    print("FFT Training Data Generator (Gemini 2.0 Flash)")
    print(f"Image dir: {args.image_dir}")
    print(f"Output:    {args.output}")
    print(f"Questions: {len(QUESTIONS)}")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  Q{i}: {q}")
    print("-" * 60)

    analyzer = GeminiFFTAnalyzer()
    analyzer.process_directory(args.image_dir, args.output)
