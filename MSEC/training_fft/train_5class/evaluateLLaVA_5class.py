"""
Evaluate finetuned LLaVA on FFT test images.
5-class: noise, sine, square, pulse, ramp.

Asks 3 questions per image (matching training questions):
  Q1: Tallest peak amplitude (nm)
  Q2: Peak frequency and decay shape
  Q3: Three tallest peaks — amplitudes and frequencies

Usage:
    python evaluateLLaVA_5class.py \
        --model-path MSEC/training_fft/train_5class/checkpoints_FFT_5class \
        --model-base models_setup/llava-v1.5-7b \
        --image-folder MSEC/training_fft/train_5class/test/noise \
        --output-file results/noise.json
"""

import argparse
import torch
import os
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from transformers import TextStreamer, CLIPImageProcessor

import warnings
warnings.filterwarnings("ignore")


QUESTIONS = [
    "What is the approximate amplitude (in nm) of the tallest peak in this FFT spectrum?",
    "At what frequency (in Hz) does the tallest peak occur? After the tallest peak, does the amplitude drop sharply to near-zero, or does it decay smoothly and gradually across higher frequencies?",
    "What are the amplitudes (in nm) and frequencies (in Hz) of the three tallest peaks in this spectrum?"
]


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if 'llava' not in model_name.lower():
        model_name = 'llava-v1.5-7b-lora'
    elif 'lora' not in model_name.lower():
        model_name = model_name + '-lora'
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    if image_processor is None:
        print("[INFO] image_processor not found. Loading default CLIP processor...")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f'[WARNING] auto inferred conv mode is {conv_mode}, using {args.conv_mode}')
    else:
        args.conv_mode = conv_mode

    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = sorted([f for f in os.listdir(args.image_folder) if f.lower().endswith(valid_extensions)])

    if args.subset:
        images = images[:args.subset]

    print(f"Evaluating {len(images)} images from {args.image_folder}")

    results = []
    for testImage in images:
        conv = conv_templates[args.conv_mode].copy()
        print(f"  {testImage}")

        image_path = os.path.join(args.image_folder, testImage)
        image = load_image(image_path)
        image_size = image.size

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        result = {
            "image": testImage,
            "conversation": []
        }

        image_added = False
        for question in QUESTIONS:
            inp = question

            if not image_added:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                image_added = True

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            if outputs.endswith("</s>"):
                outputs = outputs[:-4]

            conv.messages[-1][-1] = outputs
            result["conversation"].append({"question": question, "answer": outputs})

        results.append(result)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. {len(results)} results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on FFT test images (5 class)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--subset", type=int)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
