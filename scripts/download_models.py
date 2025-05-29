#!/usr/bin/env python3
"""Download required AI models for CLaiM."""

import os
import sys
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm


# Model definitions
MODELS: List[Dict[str, str]] = [
    {
        "name": "DistilBERT Legal",
        "filename": "distilbert-legal.gguf",
        "url": "https://huggingface.co/claim-models/distilbert-legal-gguf/resolve/main/distilbert-legal.gguf",
        "size": "250MB",
        "description": "Legal-domain fine-tuned DistilBERT for document classification",
    },
    {
        "name": "Phi-3.5-mini",
        "filename": "phi-3.5-mini-Q4_K_M.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size": "2GB",
        "description": "Quantized Phi-3.5-mini for text generation",
    },
    {
        "name": "MiniLM Embeddings",
        "filename": "all-MiniLM-L6-v2.gguf",
        "url": "https://huggingface.co/claim-models/minilm-embeddings-gguf/resolve/main/all-MiniLM-L6-v2.gguf",
        "size": "90MB",
        "description": "Sentence embeddings model for semantic search",
    },
]


def download_file(url: str, filepath: Path, description: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(
                desc=description,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {description}: {e}")
        return False


def main():
    """Download all required models."""
    print("CLaiM Model Downloader")
    print("=" * 50)
    
    # Determine models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"Models directory: {models_dir}")
    print()
    
    # Check existing models
    existing_models = []
    missing_models = []
    
    for model in MODELS:
        filepath = models_dir / model["filename"]
        if filepath.exists():
            existing_models.append(model)
        else:
            missing_models.append(model)
    
    # Display status
    if existing_models:
        print("✅ Existing models:")
        for model in existing_models:
            print(f"   - {model['name']} ({model['filename']})")
        print()
    
    if not missing_models:
        print("All models are already downloaded!")
        return
    
    print("📥 Missing models to download:")
    for model in missing_models:
        print(f"   - {model['name']} ({model['size']}) - {model['description']}")
    print()
    
    # Confirm download
    response = input("Do you want to download these models? [y/N] ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print()
    print("Note: These URLs are placeholders. In production, you would:")
    print("1. Download base models from HuggingFace")
    print("2. Convert them to GGUF format using llama.cpp")
    print("3. Fine-tune on legal domain data")
    print("4. Host the fine-tuned models for distribution")
    print()
    print("For now, creating placeholder files...")
    print()
    
    # Create placeholder files for development
    for model in missing_models:
        filepath = models_dir / model["filename"]
        print(f"Creating placeholder for {model['name']}...")
        
        # Create a small placeholder file
        with open(filepath, 'w') as f:
            f.write(f"# Placeholder for {model['name']}\n")
            f.write(f"# Actual model would be {model['size']}\n")
            f.write(f"# Download from: {model['url']}\n")
        
        print(f"✅ Created {filepath}")
    
    print()
    print("✅ Model setup complete!")
    print()
    print("To use actual models:")
    print("1. Download base models from HuggingFace")
    print("2. Use scripts/convert_to_gguf.py to convert them")
    print("3. Replace placeholder files with actual GGUF files")


if __name__ == "__main__":
    main()