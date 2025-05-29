#!/usr/bin/env python3
"""Analyze PDF structure to understand actual document boundaries."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import fitz  # PyMuPDF
from modules.document_processor.ocr_handler import OCRHandler
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def analyze_pdf_structure():
    """Analyze the structure of Sample PDFs.pdf"""
    
    pdf_path = Path("/home/nick/Projects/CLaiM/tests/Sample PDFs.pdf")
    pdf_doc = fitz.open(str(pdf_path))
    
    logger.info(f"Analyzing {pdf_path.name} - {pdf_doc.page_count} pages")
    logger.info("=" * 80)
    
    # Initialize OCR for scanned pages
    ocr_handler = OCRHandler(min_confidence=0.3)
    
    # Analyze each page
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc[page_num]
        text = page.get_text()
        
        # Basic info
        text_length = len(text.strip())
        is_scanned = text_length < 10
        
        logger.info(f"\nPage {page_num + 1}:")
        logger.info(f"  Text length: {text_length} chars")
        logger.info(f"  Type: {'SCANNED' if is_scanned else 'TEXT'}")
        
        # If scanned, perform OCR to get content
        if is_scanned:
            try:
                ocr_text, confidence = ocr_handler.process_page(page, dpi=150)
                text = ocr_text
                logger.info(f"  OCR confidence: {confidence:.2f}")
            except Exception as e:
                logger.error(f"  OCR failed: {e}")
                continue
        
        # Look for document indicators
        if text:
            # Email indicators
            if any(marker in text[:500] for marker in ["From:", "To:", "Subject:", "Sent:"]):
                logger.info("  ✓ EMAIL detected")
            
            # RFI indicators
            if "REQUEST FOR INFORMATION" in text.upper() or "RFI" in text.upper():
                logger.info("  ✓ RFI detected")
            
            # Submittal indicators
            if "SUBMITTAL" in text.upper() or "TRANSMITTAL" in text.upper():
                logger.info("  ✓ SUBMITTAL detected")
            
            # Payment indicators
            if any(term in text.upper() for term in ["SCHEDULE OF VALUES", "APPLICATION AND CERTIFICATE", "CONTINUATION SHEET"]):
                logger.info("  ✓ PAYMENT document detected")
            
            # Packing slip indicators
            if "PACKING SLIP" in text.upper() or "SALES ORDER" in text.upper():
                logger.info("  ✓ PACKING SLIP detected")
            
            # Cost proposal indicators
            if "COST PROPOSAL" in text.upper() or "PROPOSAL #" in text.upper():
                logger.info("  ✓ COST PROPOSAL detected")
            
            # Change order indicators
            if "CHANGE ORDER" in text.upper() or "PCO" in text.upper():
                logger.info("  ✓ CHANGE ORDER detected")
            
            # Drawing indicators
            if any(term in text.upper() for term in ["SCALE:", "SHEET", "SECTION", "DETAIL"]):
                logger.info("  ✓ DRAWING detected")
            
            # Show first 100 chars as preview
            preview = text[:100].replace('\n', ' ').strip()
            if preview:
                logger.info(f"  Preview: {preview}...")
        
        # Check for images/drawings
        images = page.get_images()
        drawings = page.get_drawings()
        if images:
            logger.info(f"  Has {len(images)} images")
        if len(drawings) > 20:
            logger.info(f"  Has {len(drawings)} drawing elements (likely technical drawing)")
    
    pdf_doc.close()
    
    # Now provide analysis of likely boundaries
    logger.info("\n" + "=" * 80)
    logger.info("LIKELY DOCUMENT BOUNDARIES based on content:")
    logger.info("  1. Email Chain - Pages 1-4 (RFI responses)")
    logger.info("  2. Email - Page 5 (Exterior work schedule)")
    logger.info("  3. Email - Page 6 (HVAC update)")
    logger.info("  4. Submittal - Page 7 (Transmittal form)")
    logger.info("  5. Schedule of Values - Pages 8-12")
    logger.info("  6. Email - Page 13 (SOV request)")
    logger.info("  7. Payment Application - Pages 14-17")
    logger.info("  8. Packing Slips - Pages 18-22")
    logger.info("  9. RFI - Page 23")
    logger.info("  10. Drawing - Page 24 (Plumbing plan)")
    logger.info("  11. Photos - Page 25")
    logger.info("  12. Drawings - Pages 26-31 (Structural)")
    logger.info("  13. Cost Proposal - Page 32")
    logger.info("  14. Change Order - Page 33")
    logger.info("  15. Cost Proposals - Pages 34-35")
    logger.info("  16. Email - Page 36 (PCO review)")

if __name__ == "__main__":
    analyze_pdf_structure()