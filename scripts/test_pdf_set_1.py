#!/usr/bin/env python3
"""Test LLM boundary detection on Test_PDF_Set_1.pdf"""

import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.llm_boundary_detector import LLMBoundaryDetector
from backend.modules.document_processor.hybrid_boundary_detector import HybridBoundaryDetector
from backend.modules.llm_client.ollama_client import OllamaClient


def load_ground_truth():
    """Load ground truth for Test_PDF_Set_1.pdf"""
    gt_path = Path("tests/Test_PDF_Set_Ground_Truth.json")
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            data = json.load(f)
            documents = data.get("documents", [])
            
            # Convert to boundary format
            boundaries = []
            for doc in documents:
                pages = doc["pages"]
                if "-" in pages:
                    start, end = pages.split("-")
                    boundaries.append({
                        "start": int(start),
                        "end": int(end),
                        "type": doc["type"],
                        "summary": doc["summary"]
                    })
                else:
                    page = int(pages)
                    boundaries.append({
                        "start": page,
                        "end": page,
                        "type": doc["type"],
                        "summary": doc["summary"]
                    })
            return boundaries
    return []


def test_llm_detection():
    """Test LLM boundary detection on Test_PDF_Set_1.pdf"""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth()
    logger.info(f"Ground truth: {len(ground_truth)} documents")
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"PDF has {pdf_doc.page_count} pages")
    
    # Test 1: Pattern-based detection
    logger.info("\n=== Pattern-based Detection ===")
    pattern_detector = HybridBoundaryDetector()
    start_time = time.time()
    pattern_result = pattern_detector.detect_boundaries(pdf_doc)
    pattern_boundaries = pattern_result.boundaries if hasattr(pattern_result, 'boundaries') else pattern_result
    pattern_time = time.time() - start_time
    
    logger.info(f"Found {len(pattern_boundaries)} documents in {pattern_time:.2f}s")
    for i, (start, end) in enumerate(pattern_boundaries[:10]):  # Show first 10
        logger.info(f"  Doc {i+1}: pages {start+1}-{end+1} ({end-start+1} pages)")
    
    # Test 2: LLM-based detection
    logger.info("\n=== LLM-based Detection ===")
    
    # Check if Ollama is available
    try:
        llm_client = OllamaClient(model="llama3:8b-instruct-q4_0")
        test_response = llm_client.complete("Hello")
        logger.info("Ollama is available")
    except Exception as e:
        logger.error(f"Ollama not available: {e}")
        logger.info("Skipping LLM detection")
        pdf_doc.close()
        return
    
    llm_detector = LLMBoundaryDetector(
        llm_client=llm_client,
        window_size=3,
        overlap=1,
        confidence_threshold=0.7
    )
    
    start_time = time.time()
    llm_boundaries = llm_detector.detect_boundaries(pdf_doc)
    llm_time = time.time() - start_time
    
    logger.info(f"Found {len(llm_boundaries)} documents in {llm_time:.2f}s")
    for i, (start, end) in enumerate(llm_boundaries):
        logger.info(f"  Doc {i+1}: pages {start+1}-{end+1} ({end-start+1} pages)")
    
    # Compare with ground truth
    logger.info("\n=== Comparison with Ground Truth ===")
    logger.info(f"Ground truth: {len(ground_truth)} documents")
    logger.info(f"Pattern-based: {len(pattern_boundaries)} documents")
    logger.info(f"LLM-based: {len(llm_boundaries)} documents")
    
    # Detailed comparison
    logger.info("\nGround Truth Documents:")
    for i, doc in enumerate(ground_truth):
        logger.info(f"  {i+1}. Pages {doc['start']}-{doc['end']}: {doc['type']}")
    
    # Calculate accuracy
    def calculate_accuracy(detected, ground_truth):
        if not ground_truth:
            return 0.0
        
        matches = 0
        for gt in ground_truth:
            gt_start = gt['start'] - 1  # Convert to 0-based
            gt_end = gt['end'] - 1
            
            for det_start, det_end in detected:
                # Allow 1 page tolerance
                if abs(det_start - gt_start) <= 1 and abs(det_end - gt_end) <= 1:
                    matches += 1
                    break
        
        return matches / len(ground_truth)
    
    pattern_accuracy = calculate_accuracy(pattern_boundaries, ground_truth)
    llm_accuracy = calculate_accuracy(llm_boundaries, ground_truth)
    
    logger.info(f"\nAccuracy:")
    logger.info(f"  Pattern-based: {pattern_accuracy:.2%}")
    logger.info(f"  LLM-based: {llm_accuracy:.2%}")
    
    # Show mismatches for LLM
    logger.info("\nLLM Detection Analysis:")
    for i, (start, end) in enumerate(llm_boundaries):
        # Find matching ground truth
        match = None
        for gt in ground_truth:
            gt_start = gt['start'] - 1
            gt_end = gt['end'] - 1
            if abs(start - gt_start) <= 1 and abs(end - gt_end) <= 1:
                match = gt
                break
        
        if match:
            logger.info(f"  ✓ Doc {i+1} (pages {start+1}-{end+1}): Matched {match['type']}")
        else:
            logger.info(f"  ✗ Doc {i+1} (pages {start+1}-{end+1}): No match")
    
    pdf_doc.close()


if __name__ == "__main__":
    test_llm_detection()