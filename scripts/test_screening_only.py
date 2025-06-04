#!/usr/bin/env python3
"""Test just the screening phase with debug output."""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger
from backend.modules.llm_client.ollama_client import OllamaClient


def test_screening():
    """Test the screening prompt directly."""
    
    # Simple test prompt similar to what two-stage detector uses
    prompt = """RETURN ONLY JSON. NO EXPLANATION.

Page 1: Email=True, Doc=False, Len=1500
Page 2: Email=False, Doc=False, Len=2000
Page 3: Email=False, Doc=True, Len=1200
Page 4: Email=True, Doc=False, Len=800
Page 5: Email=False, Doc=False, Len=1600

Output JSON array marking document boundaries:
[{"page": 0, "likely_boundary": true, "confidence": 0.8, "needs_deep_analysis": false}]

Rules: Email=True means boundary. Doc=True means boundary."""
    
    # Test with different models
    models = [
        ("phi3:mini", 60),
        ("llama3:8b-instruct-q5_K_M", 60)
    ]
    
    for model_name, timeout in models:
        logger.info(f"\n=== Testing {model_name} ===")
        client = OllamaClient(model_name=model_name, timeout=timeout)
        
        start = time.time()
        try:
            response = client.complete(prompt, timeout=timeout)
            elapsed = time.time() - start
            
            logger.info(f"Response time: {elapsed:.2f}s")
            logger.info(f"Response length: {len(response)} chars")
            logger.info(f"Response preview: {response[:200]}...")
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                logger.info(f"Extracted JSON: {json_match.group(0)}")
            else:
                logger.warning("No JSON array found in response")
                
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Failed after {elapsed:.2f}s: {e}")


if __name__ == "__main__":
    test_screening()