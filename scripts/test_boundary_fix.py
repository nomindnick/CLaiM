#!/usr/bin/env python3
"""Test document ingestion after fixing BoundaryCandidate error."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from pathlib import Path
from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest
from loguru import logger

def test_ingestion():
    """Test document ingestion with a sample PDF."""
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
        # Initialize PDF splitter with visual detection enabled
        splitter = PDFSplitter(use_visual_detection=True)
        
        # Create processing request with LLM validation
        request = PDFProcessingRequest(
            file_path=test_pdf,
            force_llm_validation=True  # Force LLM validation to test the fix
        )
        
        # Process the PDF
        logger.info("Starting PDF processing with LLM validation...")
        result = splitter.process_pdf(request)
        
        if result.success:
            logger.success(f"Successfully processed PDF! Found {result.documents_found} documents")
            
            # Show summary
            for i, doc in enumerate(result.documents):
                logger.info(f"Document {i+1}: {doc.type} (pages {doc.page_range[0]}-{doc.page_range[1]})")
                
            # Show detection level used
            if result.detection_level:
                logger.info(f"Detection level: {result.detection_level}")
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
    success = test_ingestion()
    sys.exit(0 if success else 1)