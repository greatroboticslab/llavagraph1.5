import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

def load_lora_model(lora_path, base_model_path, device="cuda"):
    """Load LoRA fine-tuned model"""
    print(f"Loading base model: {base_model_path}")
    
    # Load base LLaVA model
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    print(f"Applying LoRA weights: {lora_path}")
    
    # Apply LoRA weights
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    # Merge weights for faster inference
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    model.eval()
    print("✅ Model loaded successfully\n")
    
    return model

def eval_model(args):
    # Load tokenizer and image processor
    print("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    # Load model
    model = load_lora_model(args.model_path, args.model_base, args.device)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(args.image_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"❌ No image files found in {args.image_folder}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    # List of questions
    questions = [
        "Is the line shown in the graph continuous? Describe the line.",
        "Does the graph contain any random points?",
        "Does the graph contain sharp corners?"
    ]
    
    results = []
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        image_path = os.path.join(args.image_folder, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"\n⚠️ Cannot load {image_file}: {e}")
            continue
        
        conversations = []
        
        for question in questions:
            # Build prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Process image
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            # Tokenize input
            input_ids = tokenizer_image_token(
                prompt, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)
            
            # Generate answer
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=512,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            conversations.append({
                "question": question,
                "answer": answer
            })
        
        results.append({
            "image": image_file,
            "conversation": conversations
        })
        
        # Print progress every 10 images
        if (idx + 1) % 10 == 0:
            print(f"\nProcessed {idx + 1}/{len(image_files)} images")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluation completed!")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--model-base", type=str, required=True, help="Path to base model")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images to evaluate")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save output JSON file")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    eval_model(args)
