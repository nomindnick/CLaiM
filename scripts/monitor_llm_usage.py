#!/usr/bin/env python3
"""Monitor LLM usage during document processing."""

import sys
import os
import time
import subprocess
from threading import Thread
from loguru import logger

def monitor_ollama_models():
    """Monitor which Ollama models are being used."""
    logger.info("Monitoring Ollama model usage...")
    logger.info("Press Ctrl+C to stop")
    
    last_status = None
    
    try:
        while True:
            try:
                # Run ollama list to see loaded models
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        models = []
                        for line in lines[1:]:
                            parts = line.split()
                            if parts:
                                model_name = parts[0]
                                # Check if model is loaded (has SIZE info)
                                if len(parts) > 2:
                                    models.append(model_name)
                        
                        status = ", ".join(models) if models else "No models loaded"
                        
                        if status != last_status:
                            logger.info(f"Active models: {status}")
                            last_status = status
                            
                            # Check for expected models
                            if "phi3:mini" in status:
                                logger.success("✓ phi3:mini is active (fast screening)")
                            if "llama3:8b-instruct-q4_0" in status:
                                logger.success("✓ llama3:8b-instruct-q4_0 is active (deep analysis)")
                            if "llama3:8b-instruct-q5_K_M" in status:
                                logger.warning("! llama3:8b-instruct-q5_K_M is active (not two-stage)")
                
            except subprocess.TimeoutExpired:
                logger.warning("Timeout checking ollama status")
            except Exception as e:
                logger.error(f"Error checking models: {e}")
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped")

if __name__ == "__main__":
    monitor_ollama_models()