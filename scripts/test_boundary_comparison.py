#!/usr/bin/env python3
"""Quick comparison of boundary detection methods."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest
from backend.modules.document_processor.hybrid_boundary_detector import (
    HybridBoundaryDetector, DetectionLevel
)
from backend.modules.document_processor.boundary_detector import BoundaryDetector
from loguru import logger
import fitz


def compare_methods_on_pdf(pdf_path: Path):
    """Compare pattern vs visual detection on a single PDF."""
    logger.info(f"\nAnalyzing: {pdf_path.name}")
    
    pdf_doc = fitz.open(str(pdf_path))
    page_count = pdf_doc.page_count
    
    # 1. Pattern-based detection
    pattern_detector = BoundaryDetector()
    pattern_boundaries = pattern_detector.detect_boundaries(pdf_doc)
    
    logger.info(f"\nPattern-based detection:")
    logger.info(f"  Pages: {page_count}")
    logger.info(f"  Documents found: {len(pattern_boundaries)}")
    for i, (start, end) in enumerate(pattern_boundaries[:5]):  # Show first 5
        logger.info(f"    Doc {i+1}: Pages {start+1}-{end+1}")
    if len(pattern_boundaries) > 5:
        logger.info(f"    ... and {len(pattern_boundaries)-5} more")
    
    # 2. Hybrid with visual detection
    hybrid_detector = HybridBoundaryDetector()
    visual_result = hybrid_detector.detect_boundaries(
        pdf_doc,
        max_level=DetectionLevel.VISUAL,
        force_visual=True
    )
    
    logger.info(f"\nVisual detection:")
    logger.info(f"  Detection level: {visual_result.detection_level.name}")
    logger.info(f"  Documents found: {len(visual_result.boundaries)}")
    for i, (start, end) in enumerate(visual_result.boundaries[:5]):  # Show first 5
        conf = visual_result.confidence_scores.get(start, 0)
        logger.info(f"    Doc {i+1}: Pages {start+1}-{end+1} (confidence: {conf:.2f})")
    if len(visual_result.boundaries) > 5:
        logger.info(f"    ... and {len(visual_result.boundaries)-5} more")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Pattern detected: {len(pattern_boundaries)} documents")
    logger.info(f"  Visual detected: {len(visual_result.boundaries)} documents")
    logger.info(f"  Processing time: {visual_result.processing_time:.2f}s")
    
    # Calculate difference
    pattern_set = set(pattern_boundaries)
    visual_set = set(visual_result.boundaries)
    common = pattern_set & visual_set
    only_pattern = pattern_set - visual_set
    only_visual = visual_set - pattern_set
    
    logger.info(f"\nDifferences:")
    logger.info(f"  Agreed on: {len(common)} boundaries")
    logger.info(f"  Pattern-only: {len(only_pattern)} boundaries")
    logger.info(f"  Visual-only: {len(only_visual)} boundaries")
    
    pdf_doc.close()
    
    return {
        'pdf': pdf_path.name,
        'pages': page_count,
        'pattern_docs': len(pattern_boundaries),
        'visual_docs': len(visual_result.boundaries),
        'common': len(common),
        'pattern_only': len(only_pattern),
        'visual_only': len(only_visual),
        'time': visual_result.processing_time
    }


def main():
    """Main test function."""
    test_pdfs = [
        Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf"),
        Path("tests/Sample PDFs.pdf"),
    ]
    
    results = []
    
    for pdf_path in test_pdfs:
        if pdf_path.exists():
            try:
                result = compare_methods_on_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
        else:
            logger.warning(f"PDF not found: {pdf_path}")
    
    # Print overall summary
    if results:
        logger.info("\n" + "="*60)
        logger.info("OVERALL SUMMARY")
        logger.info("="*60)
        
        for r in results:
            logger.info(f"\n{r['pdf']}:")
            logger.info(f"  Pages: {r['pages']}")
            logger.info(f"  Pattern docs: {r['pattern_docs']}")
            logger.info(f"  Visual docs: {r['visual_docs']}")
            logger.info(f"  Improvement: {r['pattern_docs'] - r['visual_docs']} fewer documents")
            logger.info(f"  Time: {r['time']:.2f}s")


if __name__ == "__main__":
    main()