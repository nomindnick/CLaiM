#!/usr/bin/env python3
"""Test visual boundary detection after dependency fix."""

import sys
from pathlib import Path
from loguru import logger

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest


def test_visual_detection():
    """Test visual boundary detection functionality."""
    logger.info("Testing visual boundary detection with fixed dependencies...")
    
    test_pdf = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        return False
    
    try:
        # Test with visual detection enabled
        logger.info("Testing with visual detection ENABLED...")
        pdf_splitter = PDFSplitter(use_visual_detection=True)
        
        request = PDFProcessingRequest(
            file_path=test_pdf,
            split_documents=True,
            perform_ocr=True
        )
        
        result = pdf_splitter.process_pdf(request)
        
        if result.success:
            logger.success(f"Visual detection found {result.documents_found} documents")
            for i, doc in enumerate(result.documents):
                logger.info(f"  Document {i+1}: {doc.type} ({doc.page_count} pages)")
        else:
            logger.error(f"Visual detection failed: {result.errors}")
            return False
            
        # Compare with pattern detection
        logger.info("\nTesting with pattern detection for comparison...")
        pdf_splitter_pattern = PDFSplitter(use_visual_detection=False)
        result_pattern = pdf_splitter_pattern.process_pdf(request)
        
        if result_pattern.success:
            logger.info(f"Pattern detection found {result_pattern.documents_found} documents")
            for i, doc in enumerate(result_pattern.documents):
                logger.info(f"  Document {i+1}: {doc.type} ({doc.page_count} pages)")
        
        # Compare results
        logger.info("\nComparison:")
        logger.info(f"Visual detection: {result.documents_found} documents")
        logger.info(f"Pattern detection: {result_pattern.documents_found} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run visual detection test."""
    success = test_visual_detection()
    
    if success:
        logger.success("\n✅ Visual boundary detection is working correctly!")
        logger.info("The dependency issue has been resolved.")
    else:
        logger.error("\n❌ Visual boundary detection test failed")
    
    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)