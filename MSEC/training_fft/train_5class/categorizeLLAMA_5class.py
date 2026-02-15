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
    "Output format: For each step, write your comparison explicitly, e.g.:\n"
    "  amplitude = 23 nm. Is 23 > 250? No. Is 23 < 60? Yes. → go to STEP 2.\n"
    "When you reach a Result, write \"Result: X\" and STOP immediately.\n\n"
    "STEP 1 — Peak amplitude\n"
    "  Write: amplitude = [value] nm.\n"
    "  Is [value] > 250? If yes → Result: D. STOP.\n"
    "  Is [value] < 60? If yes → go to STEP 2.\n"
    "  Otherwise → go to STEP 3.\n\n"
    "STEP 2 — Low-amplitude frequency check\n"
    "  Write: frequency = [value] Hz.\n"
    "  Is [value] ≤ 2, or are there no clear peaks? If yes → Result: D. STOP.\n"
    "  Otherwise → Result: A. STOP.\n\n"
    "STEP 3 — Harmonic pattern check\n"
    "  List the three tallest peaks:\n"
    "    Peak 1: [amp1] nm at [freq1] Hz\n"
    "    Peak 2: [amp2] nm at [freq2] Hz\n"
    "    Peak 3: [amp3] nm at [freq3] Hz\n"
    "  For each peak, check: is amplitude ≥ 5 nm?\n"
    "    Peak 1: [amp1] ≥ 5? [yes/no → keep/discard]\n"
    "    Peak 2: [amp2] ≥ 5? [yes/no → keep/discard]\n"
    "    Peak 3: [amp3] ≥ 5? [yes/no → keep/discard]\n"
    "  Count the kept peaks. If fewer than 3 → go to STEP 5.\n"
    "  Sort kept peaks by frequency (lowest first): f_low, f_mid, f_high.\n"
    "  spacing1 = f_mid − f_low = ?\n"
    "  spacing2 = f_high − f_mid = ?\n"
    "  difference = |spacing1 − spacing2| = ?\n"
    "  max_spacing = max(spacing1, spacing2) = ?\n"
    "  ratio = difference / max_spacing = ?\n"
    "  Is ratio ≤ 0.30? If yes → harmonic pattern found → go to STEP 4.\n"
    "  Otherwise → no harmonic pattern → go to STEP 5.\n\n"
    "STEP 4 — Harmonic: ramp vs square\n"
    "  ratio = [amp2] / [amp1] = ?\n"
    "  Is ratio ≤ 0.15? If yes → Result: E. STOP.\n"
    "  Otherwise → Result: C. STOP.\n\n"
    "STEP 5 — No harmonics: sine vs square\n"
    "  third_peak_amplitude = [amp3] nm (0 if not reported).\n"
    "  Is [amp3] ≥ 25? If yes → Result: C. STOP.\n"
    "  Otherwise → Result: B. STOP."
)


def main(args, pipe):
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
    parser.add_argument("--conversation-files", type=str, nargs='+', required=True,
                        help="One or more conversation JSON files to classify")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save classified results")
    parser.add_argument("--subset", type=int)
    args = parser.parse_args()

    # Load model once
    print(f"Loading model from {args.model_path}...")
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each file with the same model
    for conv_file in args.conversation_files:
        cat_name = os.path.splitext(os.path.basename(conv_file))[0]
        output_file = os.path.join(args.output_dir, f"{cat_name}_classified.json")
        print(f"\n=== Classifying {cat_name} ===")

        args.conversation_file = conv_file
        args.output_file = output_file
        main(args, pipe)
