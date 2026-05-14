#!/usr/bin/env python3
"""
Model download script using ModelScope API.
Optimized for network speed.
"""

# Use modelscope snapshot_download for model acquisition
from modelscope import snapshot_download
import os

# Configure model list
MODEL_LIST = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "LLM-Research/Llama-3.2-1B-Instruct",
    # "LLM-Research/Llama-3.2-3B-Instruct",
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    # "LLM-Research/Meta-Llama-3-8B-Instruct",
    # "Qwen/Qwen3-8B",
]
BASE_DIR = "/root/autodl-tmp/model"

def main():
    print("=" * 60)
    print("Downloading models via ModelScope API")
    print(f"Download directory: {os.path.abspath(BASE_DIR)}")
    print("=" * 60)

    for model_id in MODEL_LIST:
        local_dir_name = model_id.split('/')[-1]
        local_dir_path = os.path.join(BASE_DIR, local_dir_name)
        print(f"\n  Downloading: {model_id}")
        print(f"   Saving to: {local_dir_path}")

        try:
            # Use ModelScope's snapshot_download
            # Note: cache_dir specifies the download path
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=local_dir_path,
                revision='master'
            )
            print(f"  Download complete: {local_dir_name}")
            print(f"   Model cache path: {model_dir}")
        except Exception as e:
            print(f"  Download failed: {model_id}")
            print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("All download tasks completed.")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    main()
