"""
Classify FFT waveforms using LLaMA based on LLaVA's visual descriptions.

Takes LLaVA evaluation output (3 Q/A pairs per image) and uses LLaMA to
classify each as noise, sine, or square wave based on:
  - Peak amplitude: < 60nm = noise
  - Peak ratio: second/tallest < 30% = sine, >= 30% = square
  - Decay shape: sharp drop = sine, gradual decay = square (for 1Hz)

Usage:
    python categorizeLLAMA_FFT.py \
        --model-path models_setup/Llama-3.2-3B \
        --conversation-file results/fft/noise.json \
        --output-file results/fft/noise_classified.json
"""

import torch
from transformers import pipeline
import argparse
import json
import os
import re


SYSTEM_PROMPT = (
    "You are a signal processing engineer classifying FFT spectrum descriptions. "
    "Use the following rules strictly:\n\n"
    "CLASSIFICATION RULES:\n"
    "1. RANDOM NOISE (A): The tallest peak amplitude is BELOW 60 nm. "
    "Noise signals have low, roughly uniform peaks with no dominant frequency.\n"
    "2. SINE WAVE (B): The tallest peak is ABOVE 60 nm, AND the second tallest peak "
    "is significantly smaller than the tallest (less than 30% of the tallest peak's amplitude). "
    "Sine waves have one dominant peak with minimal harmonics.\n"
    "3. SQUARE WAVE (C): The tallest peak is ABOVE 60 nm, AND the second tallest peak "
    "is a substantial fraction of the tallest (30% or more of the tallest peak's amplitude). "
    "Square waves have multiple strong harmonic peaks. Also classify as square wave if "
    "the signal shows gradual decay across a wide frequency range (common for low-frequency square waves).\n\n"
    "DECISION PROCESS:\n"
    "1. First check: Is the tallest peak below 60 nm? If yes -> A (noise)\n"
    "2. If above 60 nm, compute the ratio: second_peak / tallest_peak\n"
    "3. If ratio < 0.30 -> B (sine wave)\n"
    "4. If ratio >= 0.30 -> C (square wave)\n"
    "5. If decay is described as 'gradual' across wide range -> C (square wave)\n"
)


def main(args):
    print(f"Loading LLaMA model from {args.model_path}...")
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if not os.path.exists(args.conversation_file):
        print(f"Error: {args.conversation_file} not found.")
        return

    with open(args.conversation_file, "r") as f:
        data = json.load(f)

    if args.subset:
        data = data[:args.subset]

    results = []
    correct_count = 0
    label_map = {"A": "noise", "B": "sine", "C": "square"}

    for entry in data:
        image_name = entry.get("image", "").lower()

        # Ground truth from filename
        if "noise" in image_name:
            ground_truth = "noise"
        elif "sine" in image_name:
            ground_truth = "sine"
        elif "square" in image_name:
            ground_truth = "square"
        else:
            ground_truth = "unknown"

        # Extract LLaVA answers
        full_description = ""
        for turn in entry.get("conversation", []):
            q = turn.get("question", "")
            a = turn.get("answer", "")
            # Clean up model artifacts
            a = a.replace("<s>", "").replace("</s>", "").strip()
            full_description += f"Q: {q}\nA: {a}\n\n"

        user_prompt = (
            f"FFT Spectrum Analysis:\n{full_description}\n"
            f"Based on the classification rules, what is this signal? "
            f"Show your reasoning (peak values and ratio calculation), "
            f"then state your answer as: Result: [A, B, or C]"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        outputs = pipe(messages, max_new_tokens=200, do_sample=False)
        model_response = outputs[0]["generated_text"][-1]["content"]

        # Extract prediction
        match = re.search(r"Result:\s*([A-C])", model_response, re.IGNORECASE)
        predicted_letter = match.group(1).upper() if match else "A"
        predicted_label = label_map.get(predicted_letter, "noise")

        is_correct = (predicted_label == ground_truth)
        if is_correct:
            correct_count += 1

        print(f"  {os.path.basename(image_name)[:35]:35} | GT: {ground_truth:6} | Pred: {predicted_label:6} | {'OK' if is_correct else 'WRONG'}")

        results.append({
            "image": entry.get("image", ""),
            "gt": ground_truth,
            "pred": predicted_label,
            "is_correct": is_correct,
            "reasoning": model_response
        })

    accuracy = (correct_count / len(data)) * 100 if len(data) > 0 else 0
    print(f"\nAccuracy: {correct_count}/{len(data)} = {accuracy:.1f}%")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w") as f:
        json.dump({"accuracy": accuracy, "total": len(data), "correct": correct_count, "results": results}, f, indent=2)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify FFT waveforms using LLaMA")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conversation-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--subset", type=int)
    args = parser.parse_args()
    main(args)
