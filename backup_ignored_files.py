import os
import shutil
from pathlib import Path

def backup_ignored_files(backup_dir="ignored_files_backup"):
    """
    Copy all ignored files to a backup folder
    """
    # Define the base directory (your project on Desktop)
    base_dir = Path.home() / "Desktop" / "LLaVA_ilminur"
    
    # List of files and directories to backup
    ignored_files = [
        "ImageNet/resnet_50.pth",
        "ImageNet/resnet_finetuned.pth", 
        "LlaVAGraph/.DS_Store",
        "LlaVAGraph/data/ChartGeneration.ipynb",
        "LlaVAGraph/data/DisplacementCalculations.ipynb",
        "LlaVAGraph/data/JSONData.ipynb",
        "LlaVAGraph/data/baseImage.png",
        "LlaVAGraph/data/laserCharts.ipynb",
        "LlaVAGraph/data/textData/",
        "LlaVAGraph/eval/Experiment-Analysis.ipynb",
        "LlaVAGraph/eval/run_eval.log",
        "LlaVAGraph/llava/.DS_Store",
        "LlaVAGraph/llava/eval/table/",
        "LlaVAGraph/llava/eval/webpage/figures/alpaca.png",
        "LlaVAGraph/llava/model/",
        "LlaVAGraph/scripts/zero2.json",
        "LlaVAGraph/scripts/zero3.json",
        "LlaVAGraph/scripts/zero3_offload.json",
        "checkpoints/",
        "llava.egg-info/",
        "models/Llama-3.2-3B/.cache/",
        "models_setup/llava-v1.6-vicuna-7b/.cache/",
        "models_setup/llava-v1.6-vicuna-7b/model-00001-of-00003.safetensors",
        "models_setup/llava-v1.6-vicuna-7b/model-00002-of-00003.safetensors", 
        "models_setup/llava-v1.6-vicuna-7b/model-00003-of-00003.safetensors",
        "models_setup/llava-v1.6-vicuna-7b/training_args.bin",
        "wandb/"
    ]
    
    # Create backup directory inside your project
    backup_path = base_dir / backup_dir
    backup_path.mkdir(exist_ok=True)
    
    copied_files = []
    skipped_files = []
    
    print(f"Backing up files from: {base_dir}")
    print(f"Backup destination: {backup_path}")
    print("-" * 50)
    
    for item in ignored_files:
        source_path = base_dir / item
        
        if not source_path.exists():
            skipped_files.append(f"{item} (does not exist)")
            continue
            
        # Destination path
        dest_path = backup_path / item
        
        try:
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_file():
                # Copy file
                shutil.copy2(source_path, dest_path)
                copied_files.append(item)
                print(f"✅ Copied file: {item}")
            elif source_path.is_dir():
                # Copy directory
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                copied_files.append(item)
                print(f"✅ Copied directory: {item}")
                
        except Exception as e:
            skipped_files.append(f"{item} (error: {str(e)})")
    
    # Print summary
    print(f"\n📊 Backup completed!")
    print(f"✅ Successfully copied: {len(copied_files)} items")
    print(f"❌ Skipped: {len(skipped_files)} items")
    
    if skipped_files:
        print("\nSkipped items:")
        for item in skipped_files:
            print(f"  - {item}")
    
    return backup_path

if __name__ == "__main__":
    backup_path = backup_ignored_files()
    print(f"\nBackup location: {backup_path}")
