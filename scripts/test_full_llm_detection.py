#!/usr/bin/env python3
"""Comprehensive test of LLM boundary detection on Test_PDF_Set_1.pdf"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.llm_boundary_detector import LLMBoundaryDetector
from backend.modules.document_processor.hybrid_boundary_detector import HybridBoundaryDetector
from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest
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


def calculate_accuracy(detected: List[Tuple[int, int]], ground_truth: List[Dict]) -> Dict[str, float]:
    """Calculate detailed accuracy metrics."""
    if not ground_truth:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Check each ground truth document
    matched_gt = set()
    matched_det = set()
    
    for i, gt in enumerate(ground_truth):
        gt_start = gt['start'] - 1  # Convert to 0-based
        gt_end = gt['end'] - 1
        
        # Find best matching detection
        best_match = None
        best_overlap = 0
        
        for j, (det_start, det_end) in enumerate(detected):
            # Calculate overlap
            overlap_start = max(gt_start, det_start)
            overlap_end = min(gt_end, det_end)
            
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                total = max(gt_end - gt_start + 1, det_end - det_start + 1)
                overlap_ratio = overlap / total
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match = j
        
        # Consider it a match if overlap is > 80%
        if best_overlap > 0.8:
            true_positives += 1
            matched_gt.add(i)
            matched_det.add(best_match)
        else:
            false_negatives += 1
    
    # Count unmatched detections as false positives
    false_positives = len(detected) - len(matched_det)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": f1,  # Use F1 as overall accuracy
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def test_llm_boundary_detection():
    """Test full LLM boundary detection pipeline"""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth()
    logger.info(f"Ground truth: {len(ground_truth)} documents")
    for i, doc in enumerate(ground_truth):
        logger.info(f"  {i+1}. Pages {doc['start']}-{doc['end']}: {doc['type']}")
    
    # Check if Ollama is available
    logger.info("\n=== Checking Ollama Availability ===")
    try:
        llm_client = OllamaClient(model="llama3:8b-instruct-q4_0", timeout=300)
        response = llm_client.complete("Say 'hello'")
        logger.info("✓ Ollama is available")
    except Exception as e:
        logger.error(f"✗ Ollama not available: {e}")
        logger.info("Please ensure Ollama is running: ollama serve")
        logger.info("And model is pulled: ollama pull llama3:8b-instruct-q4_0")
        return
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"\nPDF has {pdf_doc.page_count} pages")
    
    # Test 1: Direct LLM boundary detection
    logger.info("\n=== Test 1: Direct LLM Boundary Detection ===")
    
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
    
    # Calculate metrics
    llm_metrics = calculate_accuracy(llm_boundaries, ground_truth)
    logger.info(f"\nLLM Detection Metrics:")
    logger.info(f"  Accuracy (F1): {llm_metrics['accuracy']:.2%}")
    logger.info(f"  Precision: {llm_metrics['precision']:.2%}")
    logger.info(f"  Recall: {llm_metrics['recall']:.2%}")
    logger.info(f"  True Positives: {llm_metrics['true_positives']}")
    logger.info(f"  False Positives: {llm_metrics['false_positives']}")
    logger.info(f"  False Negatives: {llm_metrics['false_negatives']}")
    
    # Test 2: PDF Splitter with LLM detection
    logger.info("\n=== Test 2: PDF Splitter with LLM Detection ===")
    
    splitter = PDFSplitter(
        use_visual_detection=False,
        use_hybrid_text_extraction=True,
        use_llm_detection=True
    )
    
    request = PDFProcessingRequest(
        file_path=pdf_path,
        split_documents=True,
        perform_ocr=True,
        ocr_language="eng"
    )
    
    start_time = time.time()
    result = splitter.process_pdf(request)
    splitter_time = time.time() - start_time
    
    if result.success:
        logger.info(f"Processing successful!")
        logger.info(f"Found {result.documents_found} documents in {splitter_time:.2f}s")
        
        # Convert to boundaries for comparison
        splitter_boundaries = [(doc.start_page - 1, doc.end_page - 1) for doc in result.documents]
        
        for i, doc in enumerate(result.documents):
            logger.info(f"  Doc {i+1}: pages {doc.start_page}-{doc.end_page} ({doc.page_count} pages) - {doc.document_type.value}")
        
        # Calculate metrics
        splitter_metrics = calculate_accuracy(splitter_boundaries, ground_truth)
        logger.info(f"\nPDF Splitter Metrics:")
        logger.info(f"  Accuracy (F1): {splitter_metrics['accuracy']:.2%}")
        logger.info(f"  Precision: {splitter_metrics['precision']:.2%}")
        logger.info(f"  Recall: {splitter_metrics['recall']:.2%}")
    else:
        logger.error(f"Processing failed: {result.error_message}")
    
    # Test 3: Compare with pattern-based detection
    logger.info("\n=== Test 3: Pattern-based Detection (Baseline) ===")
    
    hybrid_detector = HybridBoundaryDetector()
    start_time = time.time()
    pattern_result = hybrid_detector.detect_boundaries(pdf_doc)
    pattern_boundaries = pattern_result.boundaries if hasattr(pattern_result, 'boundaries') else pattern_result
    pattern_time = time.time() - start_time
    
    logger.info(f"Found {len(pattern_boundaries)} documents in {pattern_time:.2f}s")
    
    # Calculate metrics
    pattern_metrics = calculate_accuracy(pattern_boundaries, ground_truth)
    logger.info(f"\nPattern Detection Metrics:")
    logger.info(f"  Accuracy (F1): {pattern_metrics['accuracy']:.2%}")
    logger.info(f"  Precision: {pattern_metrics['precision']:.2%}")
    logger.info(f"  Recall: {pattern_metrics['recall']:.2%}")
    
    # Summary comparison
    logger.info("\n=== SUMMARY COMPARISON ===")
    logger.info(f"Ground Truth: {len(ground_truth)} documents")
    logger.info(f"\nPattern-based Detection:")
    logger.info(f"  Documents: {len(pattern_boundaries)}")
    logger.info(f"  Accuracy: {pattern_metrics['accuracy']:.2%}")
    logger.info(f"  Time: {pattern_time:.2f}s")
    logger.info(f"\nLLM-based Detection:")
    logger.info(f"  Documents: {len(llm_boundaries)}")
    logger.info(f"  Accuracy: {llm_metrics['accuracy']:.2%}")
    logger.info(f"  Time: {llm_time:.2f}s")
    logger.info(f"  Improvement: {(llm_metrics['accuracy'] - pattern_metrics['accuracy']) * 100:.1f} percentage points")
    
    pdf_doc.close()


if __name__ == "__main__":
    test_llm_boundary_detection()