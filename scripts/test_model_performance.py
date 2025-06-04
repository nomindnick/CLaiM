#!/usr/bin/env python3
"""Simple test to compare model performance."""

import time
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from backend.modules.llm_client.ollama_client import OllamaClient


def test_model_speeds():
    """Test response times for different models."""
    
    # Test prompt (similar to what we use for boundary detection)
    test_prompt = """Quick screening: Identify potential document boundaries in these pages.
Look for clear document starts only (new emails, invoices, RFIs, etc).

Page 1:
- Length: 1500 chars
- Has email header: True
- Has doc header: False
- Text preview: From: John Smith...

Page 2:
- Length: 2000 chars
- Has email header: False
- Has doc header: False
- Text preview: continued discussion about...

Page 3:
- Length: 1200 chars
- Has email header: True
- Has doc header: False
- Text preview: From: Jane Doe...

Return JSON array:
[
  {
    "page": 0,
    "likely_boundary": true/false,
    "confidence": 0.0-1.0,
    "needs_deep_analysis": true/false,
    "hint": "email/invoice/rfi/etc"
  }
]

Be fast and conservative - only mark obvious boundaries."""
    
    # Test models
    models = [
        ("phi3:mini", 60, "Fast model"),
        ("llama3:8b-instruct-q5_K_M", 300, "Deep model")
    ]
    
    results = []
    
    for model_name, timeout, description in models:
        logger.info(f"\n=== Testing {description}: {model_name} ===")
        
        try:
            client = OllamaClient(model=model_name, timeout=timeout)
            
            # Warm up the model
            logger.info("Warming up model...")
            client.complete("Hello", timeout=30)
            
            # Run multiple tests
            times = []
            for i in range(3):
                start = time.time()
                response = client.complete(test_prompt, timeout=timeout)
                elapsed = time.time() - start
                times.append(elapsed)
                logger.info(f"  Test {i+1}: {elapsed:.2f}s")
                
                # Show part of response
                if i == 0:
                    logger.debug(f"Response preview: {response[:200]}...")
            
            avg_time = sum(times) / len(times)
            results.append({
                'model': model_name,
                'description': description,
                'avg_time': avg_time,
                'times': times
            })
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
    
    # Summary
    logger.info("\n=== Performance Summary ===")
    logger.info(f"{'Model':<30} {'Avg Time':<10} {'Description'}")
    logger.info("-" * 60)
    
    for result in results:
        logger.info(f"{result['model']:<30} {result['avg_time']:<10.2f}s {result['description']}")
    
    if len(results) == 2:
        speedup = results[1]['avg_time'] / results[0]['avg_time']
        logger.info(f"\nSpeedup: {speedup:.2f}x faster with {results[0]['model']}")
        
        # Estimate improvement for 36-page document
        # Assuming ~12 windows for 36 pages
        windows = 12
        original_time = windows * results[1]['avg_time']
        optimized_time = 3 * results[0]['avg_time'] + 2 * results[1]['avg_time']  # Fast screening + some deep analysis
        
        logger.info(f"\nEstimated time for 36-page document:")
        logger.info(f"  Original (all deep): {original_time:.0f}s ({original_time/60:.1f} minutes)")
        logger.info(f"  Two-stage: {optimized_time:.0f}s ({optimized_time/60:.1f} minutes)")
        logger.info(f"  Speedup: {original_time/optimized_time:.1f}x")


if __name__ == "__main__":
    test_model_speeds()