#!/usr/bin/env python3
import argparse
import json
import torch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

def main(args):
    model_path = args.model_path  # local folder containing your llava model (or base model)
    print("Using model path:", model_path)

    # infer model name (helper used by the project)
    try:
        model_name = get_model_name_from_path(model_path)
    except Exception:
        model_name = "llava"

    print("Loading model (this may take a while)...")
    # Use the project's loader so custom 'llava' config/class is respected.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto",
        device=args.device
    )
    model.eval()

    # Load the conversations produced by evaluation (randomNoise/sineWave/squareWave)
    with open(args.conversation_file, "r") as f:
        conversations = json.load(f)

    if args.subset:
        conversations = conversations[: args.subset]

    results = []
    for conv in conversations:
        conversationId = conv["image"]
        print("Categorizing", conversationId)

        # Build prompt text (same instructions used before)
        prompt = (
            "Below is a description of a graph's waveform and a multiple-choice question about it. "
            "Do not generate additional questions, answers, or options. Only respond directly to the final question. "
            "Only use the criteria below for your decision; do not use any outside knowledge.\n\n"
        )
        for interaction in conv["conversation"]:
            prompt += f"Question: {interaction['question']}\nAnswer: {interaction['answer']}\n"

        prompt += (
            "\nThere are three options:\n\n"
            "A) Random noise: Random noise waves have data points distributed randomly across the graph. Random noise waves have considerable value shifts. Random noise waves have rapid value alterations. Random noise waves do not have a discernible structure.\n"
            "B) Sine wave: Sine waves have gradual transitions from one level to another. Sine waves do not have rapid value alterations. Sine waves do not have any randomly distributed datapoints. Sine waves have an easily discernible structure.\n"
            "C) Square wave: Square waves are not continuous. Square waves have sharp corners where it jumps from one value to another. If it lacks a discernible structure, it cannot be a square wave.\n\n"
            "Final Question: Based on this information, which type of graph do I have? Only select from the three options (A, B, C) provided above. Do not invent new answers or modify the options. Explain your reasoning.\n"
        )

        # Tokenize and run generation. Put input_ids on the same device as the model's parameters.
        # model may be sharded; using next(model.parameters()) to get a reference device is common.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(model_device)

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                # you can add other generation settings here
            )

        generated = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        results.append({"conversationId": conversationId, "response": generated})

    with open(args.output_file, "w") as fo:
        json.dump(results, fo, indent=2)

    print("Done. Output saved to:", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Local path to your llava/vicuna model folder (the one with config.json etc.)")
    parser.add_argument("--conversation-file", type=str, required=True,
                        help="One of the evaluate output json files: randomNoise.json, sineWave.json, or squareWave.json")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Where to save categorized output")
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional: only process first N examples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu); the loader uses device_map='auto' by default")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)

