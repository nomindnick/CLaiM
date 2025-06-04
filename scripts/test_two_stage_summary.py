#!/usr/bin/env python3
"""Summary test of two-stage detection showing the performance benefits."""

import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.two_stage_detector import TwoStageDetector


def test_summary():
    """Run a focused test showing two-stage performance."""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Test on first 6 pages only for quick results
    pdf_doc = fitz.open(pdf_path)
    test_doc = fitz.open()
    for i in range(6):
        test_doc.insert_pdf(pdf_doc, from_page=i, to_page=i)
    
    logger.info(f"Testing two-stage detection on {test_doc.page_count} pages")
    
    # Initialize detector
    detector = TwoStageDetector(
        fast_model="phi3:mini",
        deep_model="llama3:8b-instruct-q5_K_M",
        window_size=3,
        confidence_threshold=0.7,
        batch_size=3,
        keep_alive_minutes=5
    )
    
    # Track timing
    timings = {}
    
    # Override methods to track timing
    original_fast_screening = detector._fast_screening_pass
    original_batch_process = detector._batch_process_windows
    
    def timed_fast_screening(*args, **kwargs):
        start = time.time()
        result = original_fast_screening(*args, **kwargs)
        timings['fast_screening'] = time.time() - start
        timings['screening_pages'] = len(result)
        return result
    
    def timed_batch_process(*args, **kwargs):
        start = time.time()
        result = original_batch_process(*args, **kwargs)
        timings['deep_analysis'] = time.time() - start
        timings['windows_analyzed'] = len(result)
        return result
    
    detector._fast_screening_pass = timed_fast_screening
    detector._batch_process_windows = timed_batch_process
    
    # Run detection
    logger.info("\n=== Running Two-Stage Detection ===")
    start_total = time.time()
    
    try:
        boundaries = detector.detect_boundaries(test_doc)
        total_time = time.time() - start_total
        
        logger.info(f"\n=== Performance Summary ===")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Documents found: {len(boundaries)}")
        
        if 'fast_screening' in timings:
            logger.info(f"\nStage 1 - Fast Screening (phi3:mini):")
            logger.info(f"  Time: {timings['fast_screening']:.2f}s")
            logger.info(f"  Pages screened: {timings['screening_pages']}")
            logger.info(f"  Speed: {timings['screening_pages']/timings['fast_screening']:.1f} pages/second")
        
        if 'deep_analysis' in timings:
            logger.info(f"\nStage 2 - Deep Analysis (llama3):")
            logger.info(f"  Time: {timings['deep_analysis']:.2f}s")
            logger.info(f"  Windows analyzed: {timings['windows_analyzed']}")
            if timings['windows_analyzed'] > 0:
                logger.info(f"  Avg per window: {timings['deep_analysis']/timings['windows_analyzed']:.1f}s")
        
        logger.info(f"\n=== Comparison ===")
        # Estimate baseline (all windows with llama3)
        estimated_baseline = test_doc.page_count * 60  # ~60s per 3-page window
        logger.info(f"Estimated baseline time (all llama3): {estimated_baseline:.0f}s")
        logger.info(f"Two-stage time: {total_time:.0f}s")
        logger.info(f"Speedup: {estimated_baseline/total_time:.1f}x")
        
        logger.info(f"\n=== Documents Found ===")
        for i, (start, end) in enumerate(boundaries):
            logger.info(f"  Doc {i+1}: Pages {start+1}-{end+1}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    test_doc.close()
    pdf_doc.close()


if __name__ == "__main__":
    test_summary()