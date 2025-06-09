#!/usr/bin/env python3
"""Test that two-stage detection is properly enabled."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from pathlib import Path
from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest
from loguru import logger

def test_two_stage_detection():
    """Test document processing with two-stage detection enabled."""
    # Find a test PDF
    test_pdf_dir = Path("tests/test_data")
    if not test_pdf_dir.exists():
        logger.error(f"Test data directory not found: {test_pdf_dir}")
        return False
        
    pdf_files = list(test_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in test data directory")
        return False
    
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf}")
    
    try:
        # Initialize PDF splitter with two-stage detection enabled
        splitter = PDFSplitter(use_two_stage_detection=True)
        logger.info("✓ Two-stage detection enabled")
        
        # Create processing request
        request = PDFProcessingRequest(
            file_path=test_pdf,
            max_pages=10  # Limit for faster testing
        )
        
        # Process the PDF
        logger.info("Starting PDF processing with two-stage detection...")
        result = splitter.process_pdf(request)
        
        if result.success:
            logger.success(f"Successfully processed PDF! Found {result.documents_found} documents")
            
            # Show detection level used
            if result.detection_level:
                logger.info(f"Detection level: {result.detection_level}")
                
            # Check if two-stage was actually used
            if result.detection_level == "two-stage":
                logger.success("✓ Two-stage detection was used!")
            else:
                logger.warning(f"Two-stage detection not used, got: {result.detection_level}")
                
            # Show summary
            for i, doc in enumerate(result.documents):
                logger.info(f"Document {i+1}: {doc.type} (pages {doc.page_range[0]}-{doc.page_range[1]})")
                
        else:
            logger.error(f"Processing failed: {result.errors}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_two_stage_detection()
    sys.exit(0 if success else 1)