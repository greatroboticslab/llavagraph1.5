"""
Classify FFT waveforms using Qwen based on LLaVA's visual descriptions.
5-class: noise (A), sine (B), square (C), pulse (D), ramp (E).

Decision tree:
  STEP 1: Peak amplitude — >250nm=pulse, 60-250nm=go to STEP 3, <60nm=go to STEP 2
  STEP 2: Low amplitude — peak at 1Hz or no peaks=pulse, otherwise=noise
  STEP 3: Harmonic pattern — equal spacing between top 3 peaks?
  STEP 4: Harmonic branch — 2nd/1st ratio <=15%=ramp, >15%=square
  STEP 5: No harmonics — 3rd peak >=25nm=square, <25nm=sine

Usage:
    python categorizeLLAMA_5class.py \
        --model-path models_setup/Qwen2.5-14B-Instruct \
        --conversation-file results/noise.json \
        --output-file results/noise_classified.json
"""

import torch
from transformers import pipeline
import argparse
import json
import os
import re


SYSTEM_PROMPT = (
    "You classify FFT spectrum descriptions into exactly one category:\n"
    "  A = noise, B = sine, C = square, D = pulse, E = ramp\n\n"
    "Output format: Show your work for the current step, then write Result: A, B, C, D, or E. "
    "As soon as you write a Result line, STOP. Do NOT continue to the next step.\n\n"
    "STEP 1 — Peak amplitude\n"
    "  Extract the tallest peak amplitude from the answers.\n"
    "  - If amplitude > 250 nm → Result: D (pulse). STOP.\n"
    "  - If amplitude < 60 nm → go to STEP 2.\n"
    "  - Otherwise (60–250 nm) → go to STEP 3.\n\n"
    "STEP 2 — Low-amplitude check (amplitude < 60 nm)\n"
    "  Extract the frequency of the tallest peak.\n"
    "  - If the peak frequency is near 1 Hz (≤ 2 Hz), OR the description says\n"
    "    there are no clear peaks → Result: D (pulse). STOP.\n"
    "  - Otherwise → Result: A (noise). STOP.\n\n"
    "STEP 3 — Harmonic pattern check (amplitude 60–250 nm)\n"
    "  Extract the frequencies of the three tallest peaks (f1, f2, f3) from the\n"
    "  answers, ordered by amplitude (f1 = tallest).\n"
    "  Sort f1, f2, f3 by frequency (lowest to highest) to get f_low, f_mid, f_high.\n"
    "  Compute: spacing1 = f_mid − f_low, spacing2 = f_high − f_mid.\n"
    "  If fewer than 3 peaks are reported, treat as NO harmonic pattern → go to STEP 5.\n"
    "  - If spacing1 and spacing2 are within 30% of each other\n"
    "    (i.e., |spacing1 − spacing2| / max(spacing1, spacing2) ≤ 0.30),\n"
    "    there IS a harmonic pattern → go to STEP 4.\n"
    "  - Otherwise, NO harmonic pattern → go to STEP 5.\n\n"
    "STEP 4 — Harmonic: ramp vs square\n"
    "  Compute ratio = (second tallest peak amplitude) / (tallest peak amplitude).\n"
    "  - If ratio ≤ 0.15 → Result: E (ramp). STOP.\n"
    "  - If ratio > 0.15 → Result: C (square). STOP.\n\n"
    "STEP 5 — No harmonics: sine vs square\n"
    "  Extract the amplitude of the third tallest peak.\n"
    "  If no third peak is reported, treat amplitude as 0.\n"
    "  - If third peak amplitude ≥ 25 nm → Result: C (square). STOP.\n"
    "  - If third peak amplitude < 25 nm → Result: B (sine). STOP."
)


def main(args):
    print(f"Loading model from {args.model_path}...")
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
    label_map = {"A": "noise", "B": "sine", "C": "square", "D": "pulse", "E": "ramp"}

    for entry in data:
        image_name = entry.get("image", "").lower()

        # Ground truth from filename
        if "pulse" in image_name:
            ground_truth = "pulse"
        elif "ramp" in image_name:
            ground_truth = "ramp"
        elif "noise" in image_name:
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
            a = a.replace("<s>", "").replace("</s>", "").strip()
            full_description += f"Q: {q}\nA: {a}\n\n"

        user_prompt = (
            f"FFT Spectrum Analysis:\n{full_description}\n"
            f"Based on the classification rules, what is this signal? "
            f"Show your reasoning step by step, "
            f"then state your answer as: Result: [A, B, C, D, or E]"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        outputs = pipe(messages, max_new_tokens=500, do_sample=False)
        model_response = outputs[0]["generated_text"][-1]["content"]

        # Extract prediction — find the LAST Result line (in case model writes multiple)
        matches = re.findall(r"Result:\s*([A-E])", model_response, re.IGNORECASE)
        predicted_letter = matches[-1].upper() if matches else "A"
        predicted_label = label_map.get(predicted_letter, "noise")

        is_correct = (predicted_label == ground_truth)
        if is_correct:
            correct_count += 1

        print(f"  {os.path.basename(image_name)[:40]:40} | GT: {ground_truth:6} | Pred: {predicted_label:6} | {'OK' if is_correct else 'WRONG'}")

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
    parser = argparse.ArgumentParser(description="Classify FFT waveforms (5 class)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conversation-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--subset", type=int)
    args = parser.parse_args()
    main(args)
