#!/usr/bin/env python3
"""Quick evaluation of boundary detection on single PDF."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.evaluation import (
    BoundaryEvaluator, BoundaryAnnotation
)
from backend.modules.document_processor.boundary_detector import BoundaryDetector
from backend.modules.document_processor.hybrid_boundary_detector import (
    HybridBoundaryDetector, DetectionLevel
)
from loguru import logger
import fitz


def main():
    """Quick evaluation."""
    # Single PDF test
    pdf_path = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    
    # Create evaluator with ground truth
    evaluator = BoundaryEvaluator()
    evaluator.add_annotation(
        str(pdf_path),
        [(0, 2)]  # Ground truth: one document spanning pages 1-3
    )
    
    logger.info("Testing Mixed_Document_Contract_Amendment.pdf")
    logger.info("Ground truth: 1 document (pages 1-3)")
    
    # Open PDF
    pdf_doc = fitz.open(str(pdf_path))
    
    # Pattern detection
    logger.info("\n1. Pattern-based detection:")
    pattern_detector = BoundaryDetector()
    pattern_boundaries = pattern_detector.detect_boundaries(pdf_doc)
    logger.info(f"   Found: {len(pattern_boundaries)} documents")
    for start, end in pattern_boundaries:
        logger.info(f"   - Pages {start+1}-{end+1}")
    
    # Visual detection
    logger.info("\n2. Visual detection:")
    hybrid_detector = HybridBoundaryDetector()
    visual_result = hybrid_detector.detect_boundaries(
        pdf_doc,
        max_level=DetectionLevel.VISUAL,
        force_visual=True
    )
    logger.info(f"   Found: {len(visual_result.boundaries)} documents")
    for start, end in visual_result.boundaries:
        logger.info(f"   - Pages {start+1}-{end+1}")
    
    pdf_doc.close()
    
    # Evaluate
    logger.info("\nEvaluation metrics:")
    
    # Pattern metrics
    pattern_metrics = evaluator._compute_metrics(
        [(0, 2)],  # ground truth
        pattern_boundaries,
        0.0,
        "pattern"
    )
    logger.info(f"\nPattern-based:")
    logger.info(f"  Precision: {pattern_metrics.precision:.3f}")
    logger.info(f"  Recall: {pattern_metrics.recall:.3f}")
    logger.info(f"  F1 Score: {pattern_metrics.f1_score:.3f}")
    
    # Visual metrics
    visual_metrics = evaluator._compute_metrics(
        [(0, 2)],  # ground truth
        visual_result.boundaries,
        visual_result.processing_time,
        "visual"
    )
    logger.info(f"\nVisual detection:")
    logger.info(f"  Precision: {visual_metrics.precision:.3f}")
    logger.info(f"  Recall: {visual_metrics.recall:.3f}")
    logger.info(f"  F1 Score: {visual_metrics.f1_score:.3f}")
    logger.info(f"  Processing time: {visual_result.processing_time:.2f}s")
    
    # Summary
    logger.info("\nSummary:")
    logger.info(f"  Pattern-based correctly identified: {'YES' if pattern_metrics.f1_score == 1.0 else 'NO'}")
    logger.info(f"  Visual detection correctly identified: {'YES' if visual_metrics.f1_score == 1.0 else 'NO'}")


if __name__ == "__main__":
    main()