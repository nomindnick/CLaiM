#!/usr/bin/env python3
"""Demo script showing how the document splitting should work for Sample PDFs.pdf"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest, DocumentType
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# Known document boundaries for Sample PDFs.pdf based on manual analysis
SAMPLE_PDF_BOUNDARIES = [
    (0, 3, "Email Chain - RFI responses", DocumentType.EMAIL),
    (4, 4, "Email - Exterior work schedule", DocumentType.EMAIL),
    (5, 5, "Email - HVAC update", DocumentType.EMAIL),
    (6, 6, "Submittal Transmittal", DocumentType.SUBMITTAL),
    (7, 11, "Schedule of Values", DocumentType.PAYMENT_APPLICATION),
    (12, 12, "Email - SOV request", DocumentType.EMAIL),
    (13, 16, "Payment Application", DocumentType.PAYMENT_APPLICATION),
    (17, 21, "Packing Slips", DocumentType.INVOICE),
    (22, 22, "RFI", DocumentType.RFI),
    (23, 23, "Plumbing Plan", DocumentType.DRAWING),
    (24, 24, "Photos", DocumentType.UNKNOWN),
    (25, 30, "Structural Drawings", DocumentType.DRAWING),
    (31, 31, "Cost Proposal", DocumentType.SUBMITTAL),
    (32, 32, "Change Order", DocumentType.CHANGE_ORDER),
    (33, 34, "Cost Proposals", DocumentType.SUBMITTAL),
    (35, 35, "Email - PCO review", DocumentType.EMAIL),
]

def demo_document_splitting():
    """Demonstrate how document splitting should work."""
    
    pdf_path = Path("/home/nick/Projects/CLaiM/tests/Sample PDFs.pdf")
    
    logger.info("Document Splitting Demo for Sample PDFs.pdf")
    logger.info("=" * 60)
    logger.info("\nThis PDF contains 36 pages with 16 distinct documents.")
    logger.info("The current boundary detection is struggling due to poor OCR quality.")
    logger.info("\nIdeal document boundaries:")
    
    for i, (start, end, title, doc_type) in enumerate(SAMPLE_PDF_BOUNDARIES):
        pages = end - start + 1
        logger.info(f"\n{i+1}. {title}")
        logger.info(f"   Pages: {start+1}-{end+1} ({pages} page{'s' if pages > 1 else ''})")
        logger.info(f"   Type: {doc_type.value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("\nCurrent Challenges:")
    logger.info("1. OCR quality is poor (40-50% confidence)")
    logger.info("2. Email headers are not being detected reliably")
    logger.info("3. Form headers are partially garbled")
    logger.info("4. No blank pages between documents")
    
    logger.info("\nRecommended Improvements:")
    logger.info("1. Use fuzzy matching for document patterns")
    logger.info("2. Consider page layout changes as boundaries")
    logger.info("3. Look for document numbering patterns")
    logger.info("4. Detect transitions between text-heavy and form-style pages")
    logger.info("5. Use AI classification to confirm boundaries")
    
    # Now run actual processing to compare
    logger.info("\n" + "=" * 60)
    logger.info("\nRunning actual document processing...")
    
    request = PDFProcessingRequest(
        file_path=pdf_path,
        split_documents=True,
        perform_ocr=True,
        ocr_language="eng",
        classify_documents=True,
    )
    
    splitter = PDFSplitter()
    result = splitter.process_pdf(request)
    
    logger.info(f"\nActual Results:")
    logger.info(f"Documents found: {result.documents_found}")
    logger.info(f"Processing time: {result.processing_time:.1f}s")
    
    for i, doc in enumerate(result.documents):
        logger.info(f"\nDocument {i+1}:")
        logger.info(f"  Pages: {doc.page_range[0]}-{doc.page_range[1]}")
        logger.info(f"  Type: {doc.type.value}")
        preview = doc.text[:100].replace('\n', ' ').strip() if doc.text else "[No text]"
        logger.info(f"  Preview: {preview}...")

if __name__ == "__main__":
    demo_document_splitting()