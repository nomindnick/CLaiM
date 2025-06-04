#!/usr/bin/env python3
"""Direct test of phi3:mini model."""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from backend.modules.llm_client.ollama_client import OllamaClient


def test_phi3_direct():
    """Test phi3:mini directly with simple prompts."""
    
    # Test 1: Very simple prompt
    logger.info("=== Test 1: Simple prompt ===")
    client = OllamaClient(model_name="phi3:mini", timeout=60)
    
    prompt = "Is page 1 a new document? Reply with just YES or NO."
    
    start = time.time()
    try:
        response = client.complete(prompt, timeout=30)
        elapsed = time.time() - start
        logger.info(f"Response in {elapsed:.2f}s: {response.strip()}")
    except Exception as e:
        logger.error(f"Failed: {e}")
    
    # Test 2: Simple JSON
    logger.info("\n=== Test 2: Simple JSON ===")
    prompt = 'Return JSON: [{"page": 1, "boundary": true}]'
    
    start = time.time()
    try:
        response = client.complete(prompt, timeout=30)
        elapsed = time.time() - start
        logger.info(f"Response in {elapsed:.2f}s: {response.strip()}")
    except Exception as e:
        logger.error(f"Failed: {e}")
    
    # Test 3: Batch analysis (simplified)
    logger.info("\n=== Test 3: Batch analysis ===")
    prompt = """Analyze these pages for document boundaries:
Page 1: Email=True, Doc=False, Len=500
Page 2: Email=False, Doc=False, Len=800
Page 3: Email=True, Doc=False, Len=400

Return JSON array with page numbers that are boundaries: [1, 3]"""
    
    start = time.time()
    try:
        response = client.complete(prompt, timeout=60)
        elapsed = time.time() - start
        logger.info(f"Response in {elapsed:.2f}s: {response.strip()}")
    except Exception as e:
        logger.error(f"Failed: {e}")


if __name__ == "__main__":
    logger.info("Testing phi3:mini model directly...")
    test_phi3_direct()