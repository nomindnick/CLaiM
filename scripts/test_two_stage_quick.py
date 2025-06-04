#!/usr/bin/env python3
"""Quick test of two-stage detector with correct model names."""

import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.two_stage_detector import TwoStageDetector
from backend.modules.document_processor.llm_boundary_detector import LLMBoundaryDetector
from backend.modules.llm_client.ollama_client import OllamaClient


def test_quick_comparison():
    """Quick test on first 10 pages with correct model names."""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Load ground truth
    gt_path = Path("tests/Test_PDF_Set_Ground_Truth.json")
    with open(gt_path, 'r') as f:
        data = json.load(f)
        documents = data.get("documents", [])[:5]  # First 5 documents
    
    logger.info(f"Ground truth (first 5 documents):")
    for i, doc in enumerate(documents):
        logger.info(f"  {i+1}. Pages {doc['pages']}: {doc['type']}")
    
    # Check models
    logger.info("\n=== Checking Models ===")
    
    # Test phi-3-mini
    try:
        fast_client = OllamaClient(model="phi3:mini", timeout=60)
        response = fast_client.complete("Say hello", timeout=30)
        logger.info("✓ phi3:mini is available")
    except Exception as e:
        logger.error(f"✗ phi3:mini not available: {e}")
        return
    
    # Test llama3
    try:
        deep_client = OllamaClient(model="llama3:8b-instruct-q5_K_M", timeout=300)
        response = deep_client.complete("Say hello", timeout=30)
        logger.info("✓ llama3:8b-instruct-q5_K_M is available")
    except Exception as e:
        logger.error(f"✗ llama3:8b-instruct-q5_K_M not available: {e}")
        return
    
    # Open PDF and get first 10 pages
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"\nPDF has {pdf_doc.page_count} pages, testing on first 10")
    
    # Create temporary PDF with first 10 pages
    temp_doc = fitz.open()
    for i in range(min(10, pdf_doc.page_count)):
        temp_doc.insert_pdf(pdf_doc, from_page=i, to_page=i)
    
    # Test 1: Original LLM detector (single stage)
    logger.info("\n=== Test 1: Original LLM Detector ===")
    
    llm_detector = LLMBoundaryDetector(
        llm_client=deep_client,
        window_size=3,
        overlap=1,
        confidence_threshold=0.7
    )
    
    start_time = time.time()
    original_boundaries = llm_detector.detect_boundaries(temp_doc)
    original_time = time.time() - start_time
    
    logger.info(f"Found {len(original_boundaries)} documents in {original_time:.2f}s")
    logger.info(f"Speed: {10 / original_time:.2f} pages/second")
    for i, (start, end) in enumerate(original_boundaries):
        logger.info(f"  Doc {i+1}: pages {start+1}-{end+1}")
    
    # Test 2: Two-stage detector
    logger.info("\n=== Test 2: Two-Stage Detector ===")
    
    two_stage_detector = TwoStageDetector(
        fast_model="phi3:mini",
        deep_model="llama3:8b-instruct-q5_K_M",
        window_size=3,
        confidence_threshold=0.7,
        batch_size=5,
        keep_alive_minutes=5
    )
    
    start_time = time.time()
    two_stage_boundaries = two_stage_detector.detect_boundaries(temp_doc)
    two_stage_time = time.time() - start_time
    
    logger.info(f"Found {len(two_stage_boundaries)} documents in {two_stage_time:.2f}s")
    logger.info(f"Speed: {10 / two_stage_time:.2f} pages/second")
    for i, (start, end) in enumerate(two_stage_boundaries):
        logger.info(f"  Doc {i+1}: pages {start+1}-{end+1}")
    
    # Compare results
    speedup = original_time / two_stage_time if two_stage_time > 0 else 0
    logger.info(f"\n=== Performance Comparison ===")
    logger.info(f"Original time: {original_time:.2f}s")
    logger.info(f"Two-stage time: {two_stage_time:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Compare accuracy (simple check)
    if len(original_boundaries) == len(two_stage_boundaries):
        matching = sum(1 for o, t in zip(original_boundaries, two_stage_boundaries) if o == t)
        logger.info(f"Boundary agreement: {matching}/{len(original_boundaries)} ({matching/len(original_boundaries)*100:.1f}%)")
    else:
        logger.info(f"Different number of boundaries detected: {len(original_boundaries)} vs {len(two_stage_boundaries)}")
    
    temp_doc.close()
    pdf_doc.close()


if __name__ == "__main__":
    test_quick_comparison()