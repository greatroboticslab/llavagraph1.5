from huggingface_hub import snapshot_download
import os

print("Downloading LLaVA 1.5 7B model...")
print("This may take a while (model is ~13GB)")

model_id = "liuhaotian/llava-v1.5-7b"
local_dir = "models_setup/llava-v1.5-7b"

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"\n[OK] Model downloaded successfully to {local_dir}")
except Exception as e:
    print(f"\n[ERROR] Failed to download model: {e}")
