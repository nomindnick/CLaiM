#!/usr/bin/env python3
"""Test script for visual boundary detection."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest
from backend.modules.document_processor.hybrid_boundary_detector import (
    HybridBoundaryDetector, DetectionLevel
)
from loguru import logger
import fitz


def test_visual_detection(pdf_path: Path):
    """Test visual boundary detection on a PDF."""
    logger.info(f"Testing visual boundary detection on: {pdf_path}")
    
    # Initialize components
    splitter = PDFSplitter(use_visual_detection=True)
    
    # Create processing request
    request = PDFProcessingRequest(
        file_path=pdf_path,
        split_documents=True,
        perform_ocr=True,
        force_visual_detection=True  # Force visual detection
    )
    
    # Process PDF
    result = splitter.process_pdf(request)
    
    if result.success:
        logger.success(f"Processing successful!")
        logger.info(f"Detection level used: {result.detection_level}")
        logger.info(f"Found {result.documents_found} documents:")
        
        for i, doc in enumerate(result.documents):
            logger.info(
                f"  Document {i+1}: Pages {doc.page_range[0]}-{doc.page_range[1]} "
                f"({len(doc.pages)} pages), Type: {doc.type}"
            )
            
            # Show boundary confidence if available
            if result.boundary_confidence:
                start_page = doc.page_range[0] - 1  # Convert to 0-indexed
                confidence = result.boundary_confidence.get(start_page, 0)
                logger.info(f"    Boundary confidence: {confidence:.2f}")
    else:
        logger.error(f"Processing failed: {result.errors}")


def compare_detection_methods(pdf_path: Path):
    """Compare pattern-based vs visual detection."""
    logger.info(f"\nComparing detection methods on: {pdf_path}")
    
    pdf_doc = fitz.open(str(pdf_path))
    
    # Test pattern-based detection
    logger.info("\n1. Pattern-based detection:")
    from backend.modules.document_processor.boundary_detector import BoundaryDetector
    pattern_detector = BoundaryDetector()
    pattern_boundaries = pattern_detector.detect_boundaries(pdf_doc)
    logger.info(f"   Found {len(pattern_boundaries)} documents")
    for start, end in pattern_boundaries:
        logger.info(f"   - Pages {start+1}-{end+1}")
    
    # Test hybrid detection
    logger.info("\n2. Hybrid detection with visual analysis:")
    hybrid_detector = HybridBoundaryDetector()
    hybrid_result = hybrid_detector.detect_boundaries(
        pdf_doc,
        max_level=DetectionLevel.VISUAL,
        force_visual=True
    )
    logger.info(f"   Found {len(hybrid_result.boundaries)} documents")
    logger.info(f"   Detection level: {hybrid_result.detection_level.name.lower()}")
    
    for start, end in hybrid_result.boundaries:
        confidence = hybrid_result.confidence_scores.get(start, 0)
        logger.info(f"   - Pages {start+1}-{end+1} (confidence: {confidence:.2f})")
    
    # Get explanations
    explanations = hybrid_detector.get_boundary_explanations(hybrid_result)
    if explanations:
        logger.info("\n   Boundary explanations:")
        for page_num, reasons in explanations.items():
            logger.info(f"   Page {page_num+1}: {', '.join(reasons)}")
    
    pdf_doc.close()


def main():
    """Main test function."""
    # Test with sample PDFs
    test_pdfs = [
        Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf"),
        Path("tests/Sample PDFs.pdf"),
    ]
    
    for pdf_path in test_pdfs:
        if pdf_path.exists():
            test_visual_detection(pdf_path)
            compare_detection_methods(pdf_path)
        else:
            logger.warning(f"Test PDF not found: {pdf_path}")
    
    # Test with any PDF passed as argument
    if len(sys.argv) > 1:
        custom_pdf = Path(sys.argv[1])
        if custom_pdf.exists():
            logger.info(f"\nTesting custom PDF: {custom_pdf}")
            test_visual_detection(custom_pdf)
            compare_detection_methods(custom_pdf)
        else:
            logger.error(f"Custom PDF not found: {custom_pdf}")


if __name__ == "__main__":
    main()