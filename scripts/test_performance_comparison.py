#!/usr/bin/env python3
"""Performance comparison between original LLM detection and optimized two-stage detection."""

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
from backend.modules.document_processor.two_stage_detector import TwoStageDetector
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
    """Calculate accuracy metrics."""
    if not ground_truth:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
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
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def test_performance_comparison():
    """Compare performance between original and optimized detection."""
    
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth()
    logger.info(f"Ground truth: {len(ground_truth)} documents")
    
    # Check if models are available
    logger.info("\n=== Checking Model Availability ===")
    
    # Check phi-3-mini
    try:
        fast_client = OllamaClient(model="phi3:mini", timeout=60)
        response = fast_client.complete("Say hello", timeout=30)
        logger.info("✓ phi-3-mini is available")
    except Exception as e:
        logger.error(f"✗ phi-3-mini not available: {e}")
        logger.info("Please pull the model: ollama pull phi3:mini")
        return
    
    # Check llama3
    try:
        deep_client = OllamaClient(model="llama3:8b-instruct-q4_0", timeout=300)
        response = deep_client.complete("Say hello", timeout=30)
        logger.info("✓ llama3:8b-instruct-q4_0 is available")
    except Exception as e:
        logger.error(f"✗ llama3:8b-instruct-q4_0 not available: {e}")
        return
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    page_count = pdf_doc.page_count
    logger.info(f"\nPDF has {page_count} pages")
    
    # Test different page ranges
    test_ranges = [
        (0, 10, "First 10 pages"),
        (0, 20, "First 20 pages"),
        (0, page_count, "Full document")
    ]
    
    results = []
    
    for start_page, end_page, description in test_ranges:
        if end_page > page_count:
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {description} (pages {start_page+1}-{end_page})")
        logger.info(f"{'='*60}")
        
        # Create temporary PDF with selected pages
        temp_doc = fitz.open()
        for i in range(start_page, end_page):
            temp_doc.insert_pdf(pdf_doc, from_page=i, to_page=i)
        
        # Filter ground truth for this range
        range_gt = [
            gt for gt in ground_truth 
            if gt['start'] > start_page and gt['end'] <= end_page
        ]
        
        # Adjust ground truth page numbers
        adjusted_gt = []
        for gt in range_gt:
            adjusted_gt.append({
                'start': gt['start'] - start_page,
                'end': gt['end'] - start_page,
                'type': gt['type'],
                'summary': gt['summary']
            })
        
        logger.info(f"Ground truth documents in range: {len(adjusted_gt)}")
        
        # Test 1: Original LLM detector
        logger.info("\n--- Original LLM Detector ---")
        
        original_detector = LLMBoundaryDetector(
            llm_client=deep_client,
            window_size=3,
            overlap=1,
            confidence_threshold=0.7
        )
        
        start_time = time.time()
        original_boundaries = original_detector.detect_boundaries(temp_doc)
        original_time = time.time() - start_time
        
        original_metrics = calculate_accuracy(original_boundaries, adjusted_gt)
        
        logger.info(f"Time: {original_time:.2f}s")
        logger.info(f"Documents found: {len(original_boundaries)}")
        logger.info(f"F1 Score: {original_metrics['f1']:.2%}")
        logger.info(f"Speed: {(end_page - start_page) / original_time:.1f} pages/second")
        
        # Test 2: Two-stage detector
        logger.info("\n--- Two-Stage Detector ---")
        
        two_stage_detector = TwoStageDetector(
            fast_model="phi3:mini",
            deep_model="llama3:8b-instruct-q4_0",
            window_size=3,
            confidence_threshold=0.7,
            batch_size=5,
            keep_alive_minutes=10
        )
        
        start_time = time.time()
        two_stage_boundaries = two_stage_detector.detect_boundaries(temp_doc)
        two_stage_time = time.time() - start_time
        
        two_stage_metrics = calculate_accuracy(two_stage_boundaries, adjusted_gt)
        
        logger.info(f"Time: {two_stage_time:.2f}s")
        logger.info(f"Documents found: {len(two_stage_boundaries)}")
        logger.info(f"F1 Score: {two_stage_metrics['f1']:.2%}")
        logger.info(f"Speed: {(end_page - start_page) / two_stage_time:.1f} pages/second")
        
        # Calculate improvements
        speedup = original_time / two_stage_time if two_stage_time > 0 else 0
        accuracy_diff = two_stage_metrics['f1'] - original_metrics['f1']
        
        logger.info(f"\n--- Improvement ---")
        logger.info(f"Speedup: {speedup:.2f}x faster")
        logger.info(f"Accuracy change: {accuracy_diff:+.2%}")
        
        # Store results
        results.append({
            'description': description,
            'pages': end_page - start_page,
            'original_time': original_time,
            'original_f1': original_metrics['f1'],
            'two_stage_time': two_stage_time,
            'two_stage_f1': two_stage_metrics['f1'],
            'speedup': speedup,
            'accuracy_diff': accuracy_diff
        })
        
        temp_doc.close()
        
        # Don't run full document test if it would take too long
        if end_page == 20 and two_stage_time > 120:
            logger.info("\nSkipping full document test (would take too long)")
            break
    
    pdf_doc.close()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info("\n| Test | Pages | Original Time | Two-Stage Time | Speedup | F1 Change |")
    logger.info("|------|-------|---------------|----------------|---------|-----------|")
    
    for result in results:
        logger.info(
            f"| {result['description']:12} | {result['pages']:5} | "
            f"{result['original_time']:13.1f}s | {result['two_stage_time']:14.1f}s | "
            f"{result['speedup']:7.2f}x | {result['accuracy_diff']:+9.2%} |"
        )
    
    # Average improvements
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy_diff'] for r in results) / len(results)
        
        logger.info(f"\nAverage speedup: {avg_speedup:.2f}x")
        logger.info(f"Average accuracy change: {avg_accuracy:+.2%}")
        
        # Recommendations
        logger.info("\n=== RECOMMENDATIONS ===")
        if avg_speedup > 2.0:
            logger.info("✓ Two-stage detector provides significant speed improvements")
        if avg_accuracy >= -0.02:  # Within 2% accuracy loss
            logger.info("✓ Accuracy is maintained or improved")
        
        if avg_speedup > 1.5 and avg_accuracy >= -0.05:
            logger.info("\n✓ RECOMMENDATION: Use two-stage detector for production")
        else:
            logger.info("\n⚠ Further optimization may be needed")


if __name__ == "__main__":
    test_performance_comparison()