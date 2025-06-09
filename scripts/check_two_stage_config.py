#!/usr/bin/env python3
"""Check two-stage detection configuration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from backend.modules.document_processor.router import pdf_splitter
from loguru import logger

def check_configuration():
    """Check the PDF splitter configuration."""
    logger.info("Checking PDF splitter configuration...")
    
    logger.info(f"use_visual_detection: {pdf_splitter.use_visual_detection}")
    logger.info(f"use_llm_detection: {pdf_splitter.use_llm_detection}")
    logger.info(f"use_two_stage_detection: {pdf_splitter.use_two_stage_detection}")
    logger.info(f"use_hybrid_text_extraction: {pdf_splitter.use_hybrid_text_extraction}")
    
    if pdf_splitter.use_two_stage_detection:
        logger.success("✓ Two-stage detection is ENABLED")
        logger.info("  - Fast model: phi3:mini")
        logger.info("  - Deep model: llama3:8b-instruct-q4_0")
        logger.info("  - This configuration provides optimal performance")
    else:
        logger.warning("✗ Two-stage detection is DISABLED")
        
    return pdf_splitter.use_two_stage_detection

if __name__ == "__main__":
    enabled = check_configuration()
    sys.exit(0 if enabled else 1)