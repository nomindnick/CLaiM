#!/usr/bin/env python3
"""Simple test to verify two-stage detector is using phi3:mini correctly."""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.two_stage_detector import TwoStageDetector


def test_two_stage_simple():
    """Test two-stage detector on a small portion of the PDF."""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"PDF: {pdf_path.name} ({pdf_doc.page_count} pages)")
    
    # Create a smaller test PDF (first 12 pages)
    logger.info("\nCreating test PDF with first 12 pages...")
    test_doc = fitz.open()
    for i in range(min(12, pdf_doc.page_count)):
        test_doc.insert_pdf(pdf_doc, from_page=i, to_page=i)
    
    logger.info(f"Test PDF has {test_doc.page_count} pages")
    
    # Initialize two-stage detector
    logger.info("\n=== Initializing Two-Stage Detector ===")
    detector = TwoStageDetector(
        fast_model="phi3:mini",
        deep_model="llama3:8b-instruct-q5_K_M",
        window_size=3,
        confidence_threshold=0.7,
        batch_size=3,  # Small batch size
        keep_alive_minutes=10
    )
    
    # Run detection
    logger.info("\n=== Running Two-Stage Detection ===")
    start_time = time.time()
    
    try:
        boundaries = detector.detect_boundaries(test_doc)
        total_time = time.time() - start_time
        
        logger.info(f"\n=== Results ===")
        logger.info(f"Detection completed in {total_time:.2f}s")
        logger.info(f"Found {len(boundaries)} documents:")
        
        for i, (start, end) in enumerate(boundaries):
            pages_str = f"{start+1}-{end+1}" if start != end else str(start+1)
            logger.info(f"  Doc {i+1}: Pages {pages_str}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    test_doc.close()
    pdf_doc.close()


if __name__ == "__main__":
    # Set more detailed logging to see what's happening
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("Testing two-stage detector with phi3:mini fast screening...")
    test_two_stage_simple()