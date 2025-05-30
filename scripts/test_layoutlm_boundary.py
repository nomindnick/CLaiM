#!/usr/bin/env python3
"""Test LayoutLM boundary detection functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.hybrid_boundary_detector import (
    HybridBoundaryDetector, DetectionLevel
)
from loguru import logger
import fitz


def test_layoutlm_detection(pdf_path: Path):
    """Test LayoutLM detection on a PDF."""
    logger.info(f"Testing LayoutLM detection on: {pdf_path}")
    
    pdf_doc = fitz.open(str(pdf_path))
    
    # Initialize hybrid detector
    hybrid_detector = HybridBoundaryDetector()
    
    # Lower thresholds to force deep detection
    hybrid_detector.visual_confidence_threshold = 0.95  # Make it harder to satisfy
    
    try:
        # Test direct LayoutLM detector
        logger.info("\n1. Testing direct LayoutLM detector:")
        from backend.modules.document_processor.layoutlm_boundary_detector import LayoutLMBoundaryDetector
        from backend.modules.document_processor.visual_boundary_detector import PageFeatures
        
        layoutlm_detector = LayoutLMBoundaryDetector()
        
        # Create dummy page features
        page_features = [PageFeatures(page_num=i) for i in range(pdf_doc.page_count)]
        
        layoutlm_scores = layoutlm_detector.detect_boundaries(
            pdf_doc, page_features, hybrid_detector.ocr_handler
        )
        
        logger.info(f"  Found {len(layoutlm_scores)} boundary scores")
        for score in layoutlm_scores[:3]:
            logger.info(f"    Page {score.page_num+1}: score={score.total_score:.3f}, confidence={score.confidence:.3f}")
        
        # Also test hybrid with deep
        logger.info("\n2. Testing hybrid detector with deep level:")
        result = hybrid_detector.detect_boundaries(
            pdf_doc,
            max_level=DetectionLevel.DEEP,
            force_visual=True
        )
        
        logger.info(f"Detection completed!")
        logger.info(f"  Level used: {result.detection_level.name}")
        logger.info(f"  Documents found: {len(result.boundaries)}")
        logger.info(f"  Processing time: {result.processing_time:.2f}s")
        
        for i, (start, end) in enumerate(result.boundaries[:3]):
            conf = result.confidence_scores.get(start, 0)
            logger.info(f"    Doc {i+1}: Pages {start+1}-{end+1} (confidence: {conf:.2f})")
            
    except Exception as e:
        logger.error(f"LayoutLM detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    pdf_doc.close()


def main():
    """Main test function."""
    # Test with small PDF first
    test_pdf = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    
    if test_pdf.exists():
        test_layoutlm_detection(test_pdf)
    else:
        logger.error(f"Test PDF not found: {test_pdf}")


if __name__ == "__main__":
    main()