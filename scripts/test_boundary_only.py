#!/usr/bin/env python3
"""Test just boundary detection without full processing."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import fitz
from modules.document_processor.boundary_detector import BoundaryDetector
from modules.document_processor.ocr_handler import OCRHandler
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def test_boundaries():
    """Test boundary detection on Sample PDFs.pdf"""
    
    pdf_path = Path("/home/nick/Projects/CLaiM/tests/Sample PDFs.pdf")
    pdf_doc = fitz.open(str(pdf_path))
    
    logger.info(f"Testing boundary detection on {pdf_path.name}")
    logger.info("=" * 60)
    
    # Initialize OCR handler for boundary detection
    ocr_handler = OCRHandler(min_confidence=0.3)
    
    # Test boundary detection
    detector = BoundaryDetector(ocr_handler=ocr_handler)
    boundaries = detector.detect_boundaries(pdf_doc)
    
    logger.info(f"\nFound {len(boundaries)} documents:")
    for i, (start, end) in enumerate(boundaries):
        logger.info(f"  Document {i+1}: Pages {start+1}-{end+1} ({end-start+1} pages)")
    
    pdf_doc.close()

if __name__ == "__main__":
    test_boundaries()