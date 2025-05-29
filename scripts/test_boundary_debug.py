#!/usr/bin/env python3
"""Debug boundary detection to see what's happening."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import fitz
from modules.document_processor.ocr_handler import OCRHandler
from loguru import logger
import re

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def analyze_page_for_boundaries(text, page_num):
    """Analyze a page's text for boundary indicators."""
    
    indicators = []
    
    # Email patterns
    if re.search(r"From:\s*.*@", text[:800], re.IGNORECASE):
        indicators.append("EMAIL: From with @")
    if re.search(r"To:\s*.*@", text[:800], re.IGNORECASE):
        indicators.append("EMAIL: To with @")
    if re.search(r"Subject:", text[:800], re.IGNORECASE):
        indicators.append("EMAIL: Subject line")
    if re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\w+\s+\d{1,2},\s+\d{4}", text[:800], re.IGNORECASE):
        indicators.append("EMAIL: Date pattern")
    
    # Form patterns
    if re.search(r"SUBMITTAL\s*TRANSMITTAL", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Submittal Transmittal")
    if re.search(r"APPLICATION\s+AND\s+CERTIFICATE\s+FOR\s+PAYMENT", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Payment Application")
    if re.search(r"SCHEDULE\s+OF\s+VALUES", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Schedule of Values")
    if re.search(r"PACKING\s+SLIP", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Packing Slip")
    if re.search(r"COST\s+PROPOSAL", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Cost Proposal")
    if re.search(r"CHANGE\s+ORDER", text[:1000], re.IGNORECASE):
        indicators.append("FORM: Change Order")
    if re.search(r"REQUEST\s+FOR\s+INFORMATION", text[:1000], re.IGNORECASE):
        indicators.append("FORM: RFI")
    
    return indicators

def test_boundaries_debug():
    """Debug boundary detection on Sample PDFs.pdf"""
    
    pdf_path = Path("/home/nick/Projects/CLaiM/tests/Sample PDFs.pdf")
    pdf_doc = fitz.open(str(pdf_path))
    
    logger.info(f"Debugging boundary detection on {pdf_path.name}")
    logger.info("=" * 60)
    
    # Initialize OCR handler
    ocr_handler = OCRHandler(min_confidence=0.3)
    
    # Check each page
    for page_num in range(min(15, pdf_doc.page_count)):  # First 15 pages
        page = pdf_doc[page_num]
        text = page.get_text()
        
        # If scanned, do OCR
        if len(text.strip()) < 10:
            try:
                ocr_text, confidence = ocr_handler.process_page(page, dpi=150)
                text = ocr_text
            except:
                text = "[OCR FAILED]"
        
        # Analyze for indicators
        indicators = analyze_page_for_boundaries(text, page_num)
        
        if indicators:
            logger.info(f"\nPage {page_num + 1}: {', '.join(indicators)}")
            # Show preview
            preview = text[:150].replace('\n', ' ').strip()
            logger.info(f"  Preview: {preview}...")
    
    pdf_doc.close()

if __name__ == "__main__":
    test_boundaries_debug()