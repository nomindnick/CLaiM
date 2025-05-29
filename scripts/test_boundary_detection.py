#!/usr/bin/env python3
"""Test script to verify boundary detection and OCR improvements."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest
from loguru import logger

# Configure logger for better output
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>")

def test_sample_pdf():
    """Test the Sample PDFs.pdf with enhanced boundary detection and OCR."""
    
    pdf_path = Path("/home/nick/Projects/CLaiM/tests/Sample PDFs.pdf")
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info(f"Testing PDF: {pdf_path}")
    logger.info("=" * 80)
    
    # Create processing request with OCR enabled
    request = PDFProcessingRequest(
        file_path=pdf_path,
        split_documents=True,
        perform_ocr=True,
        ocr_language="eng",
        min_confidence=0.3,  # Lower threshold for testing
        classify_documents=True,
        extract_metadata=False,
    )
    
    # Process the PDF
    splitter = PDFSplitter()
    result = splitter.process_pdf(request)
    
    # Display results
    logger.info(f"\nProcessing complete in {result.processing_time:.2f} seconds")
    logger.info(f"Total pages: {result.total_pages}")
    logger.info(f"Documents found: {result.documents_found}")
    logger.info(f"OCR pages: {result.ocr_pages}")
    logger.info(f"Average confidence: {result.average_confidence:.2f}")
    
    if result.warnings:
        logger.warning("Warnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")
    
    if result.errors:
        logger.error("Errors:")
        for error in result.errors:
            logger.error(f"  - {error}")
    
    # Display document details
    logger.info("\nDocument Details:")
    logger.info("-" * 80)
    
    for i, doc in enumerate(result.documents):
        logger.info(f"\nDocument {i + 1}:")
        logger.info(f"  Type: {doc.type}")
        logger.info(f"  Pages: {doc.page_range[0]}-{doc.page_range[1]} ({len(doc.pages)} pages)")
        logger.info(f"  Has OCR content: {doc.has_ocr_content}")
        logger.info(f"  Average OCR confidence: {doc.average_ocr_confidence:.2f}")
        
        # Show first 200 characters of text
        preview = doc.text[:200].replace('\n', ' ').strip()
        if preview:
            logger.info(f"  Text preview: {preview}...")
        else:
            logger.info(f"  Text preview: [No text extracted]")
        
        # Show page-level details for first few pages
        for j, page in enumerate(doc.pages[:3]):
            logger.debug(f"    Page {page.page_number}: "
                        f"{'Scanned' if page.is_scanned else 'Text'}, "
                        f"Confidence: {page.confidence:.2f}, "
                        f"{'Has tables' if page.has_tables else ''} "
                        f"{'Has images' if page.has_images else ''}")

if __name__ == "__main__":
    test_sample_pdf()