#!/usr/bin/env python3
"""Full test of two-stage detector on 36-page PDF with ground truth comparison."""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.modules.document_processor.two_stage_detector import TwoStageDetector
from backend.modules.document_processor.llm_boundary_detector import LLMBoundaryDetector
from backend.modules.llm_client.ollama_client import OllamaClient


def load_ground_truth(gt_path: Path) -> List[Dict]:
    """Load ground truth data."""
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    # Convert ground truth format to our expected format
    documents = []
    for i, doc in enumerate(data.get("documents", [])):
        # Parse page range (e.g., "1-4" or "13")
        pages_str = doc["pages"]
        if "-" in pages_str:
            start, end = map(int, pages_str.split("-"))
            pages = list(range(start, end + 1))
        else:
            pages = [int(pages_str)]
        
        documents.append({
            "id": i + 1,
            "pages": pages,
            "type": doc["type"],
            "summary": doc.get("summary", "")
        })
    
    return documents


def boundaries_to_documents(boundaries: List[Tuple[int, int]]) -> List[Dict]:
    """Convert boundary tuples to document format for comparison."""
    documents = []
    for i, (start, end) in enumerate(boundaries):
        documents.append({
            "id": i + 1,
            "pages": list(range(start + 1, end + 2)),  # Convert to 1-indexed
            "type": "unknown"  # Type will be determined later
        })
    return documents


def calculate_overlap(pages1: List[int], pages2: List[int]) -> float:
    """Calculate overlap ratio between two page lists."""
    set1 = set(pages1)
    set2 = set(pages2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def match_documents(detected: List[Dict], ground_truth: List[Dict], threshold: float = 0.5) -> Dict:
    """Match detected documents to ground truth based on page overlap."""
    matches = []
    unmatched_detected = list(range(len(detected)))
    unmatched_gt = list(range(len(ground_truth)))
    
    # Find best matches
    for i, det_doc in enumerate(detected):
        best_match = None
        best_overlap = 0.0
        
        for j, gt_doc in enumerate(ground_truth):
            if j not in unmatched_gt:
                continue
                
            overlap = calculate_overlap(det_doc["pages"], gt_doc["pages"])
            if overlap > best_overlap and overlap >= threshold:
                best_overlap = overlap
                best_match = j
        
        if best_match is not None:
            matches.append((i, best_match, best_overlap))
            unmatched_detected.remove(i)
            unmatched_gt.remove(best_match)
    
    return {
        "matches": matches,
        "unmatched_detected": unmatched_detected,
        "unmatched_gt": unmatched_gt
    }


def calculate_metrics(detected: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 score."""
    matching = match_documents(detected, ground_truth)
    
    true_positives = len(matching["matches"])
    false_positives = len(matching["unmatched_detected"])
    false_negatives = len(matching["unmatched_gt"])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matching_details": matching
    }


def test_full_document():
    """Test two-stage detector on full 36-page document."""
    
    # Paths
    pdf_path = Path("tests/Test_PDF_Set_1.pdf")
    gt_path = Path("tests/Test_PDF_Set_Ground_Truth.json")
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    if not gt_path.exists():
        logger.error(f"Ground truth not found: {gt_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth(gt_path)
    logger.info(f"\n=== Ground Truth ===")
    logger.info(f"Total documents: {len(ground_truth)}")
    for doc in ground_truth:
        pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
        logger.info(f"  Doc {doc['id']}: Pages {pages_str} ({len(doc['pages'])} pages) - {doc['type']}")
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"\nPDF: {pdf_path.name} ({pdf_doc.page_count} pages)")
    
    # Test 1: Original LLM detector (baseline)
    logger.info("\n=== Test 1: Original LLM Detector (Baseline) ===")
    
    llm_client = OllamaClient(
        model_name="llama3:8b-instruct-q5_K_M",
        timeout=300,
        keep_alive='10m'
    )
    
    llm_detector = LLMBoundaryDetector(
        llm_client=llm_client,
        window_size=3,
        overlap=1,
        confidence_threshold=0.7
    )
    
    start_time = time.time()
    try:
        original_boundaries = llm_detector.detect_boundaries(pdf_doc)
        original_time = time.time() - start_time
        original_success = True
    except Exception as e:
        logger.error(f"Original detector failed: {e}")
        original_boundaries = []
        original_time = time.time() - start_time
        original_success = False
    
    if original_success:
        logger.info(f"Found {len(original_boundaries)} documents in {original_time:.2f}s ({original_time/60:.1f} minutes)")
        logger.info(f"Speed: {pdf_doc.page_count / original_time:.2f} pages/second")
        
        # Convert to document format and calculate metrics
        original_docs = boundaries_to_documents(original_boundaries)
        original_metrics = calculate_metrics(original_docs, ground_truth)
        
        logger.info(f"\nAccuracy Metrics:")
        logger.info(f"  Precision: {original_metrics['precision']:.2%}")
        logger.info(f"  Recall: {original_metrics['recall']:.2%}")
        logger.info(f"  F1 Score: {original_metrics['f1']:.2%}")
    
    # Test 2: Two-stage detector
    logger.info("\n=== Test 2: Two-Stage Detector (Optimized) ===")
    
    two_stage_detector = TwoStageDetector(
        fast_model="phi3:mini",
        deep_model="llama3:8b-instruct-q5_K_M",
        window_size=3,
        confidence_threshold=0.7,
        batch_size=5,
        keep_alive_minutes=10
    )
    
    start_time = time.time()
    try:
        two_stage_boundaries = two_stage_detector.detect_boundaries(pdf_doc)
        two_stage_time = time.time() - start_time
        two_stage_success = True
    except Exception as e:
        logger.error(f"Two-stage detector failed: {e}")
        two_stage_boundaries = []
        two_stage_time = time.time() - start_time
        two_stage_success = False
    
    if two_stage_success:
        logger.info(f"Found {len(two_stage_boundaries)} documents in {two_stage_time:.2f}s ({two_stage_time/60:.1f} minutes)")
        logger.info(f"Speed: {pdf_doc.page_count / two_stage_time:.2f} pages/second")
        
        # Convert to document format and calculate metrics
        two_stage_docs = boundaries_to_documents(two_stage_boundaries)
        two_stage_metrics = calculate_metrics(two_stage_docs, ground_truth)
        
        logger.info(f"\nAccuracy Metrics:")
        logger.info(f"  Precision: {two_stage_metrics['precision']:.2%}")
        logger.info(f"  Recall: {two_stage_metrics['recall']:.2%}")
        logger.info(f"  F1 Score: {two_stage_metrics['f1']:.2%}")
    
    # Performance comparison
    logger.info("\n=== Performance Comparison ===")
    
    if original_success and two_stage_success:
        speedup = original_time / two_stage_time if two_stage_time > 0 else 0
        logger.info(f"Original time: {original_time:.2f}s ({original_time/60:.1f} minutes)")
        logger.info(f"Two-stage time: {two_stage_time:.2f}s ({two_stage_time/60:.1f} minutes)")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Accuracy comparison
        logger.info(f"\nAccuracy Comparison:")
        logger.info(f"{'Metric':<20} {'Original':<15} {'Two-Stage':<15} {'Difference'}")
        logger.info("-" * 65)
        
        for metric in ['precision', 'recall', 'f1']:
            orig_val = original_metrics[metric]
            two_val = two_stage_metrics[metric]
            diff = two_val - orig_val
            sign = "+" if diff >= 0 else ""
            logger.info(f"{metric.capitalize():<20} {orig_val:<15.2%} {two_val:<15.2%} {sign}{diff:.2%}")
    
    # Detailed analysis
    if two_stage_success:
        logger.info("\n=== Detailed Results (Two-Stage) ===")
        
        # Show detected documents
        logger.info("\nDetected Documents:")
        for i, (start, end) in enumerate(two_stage_boundaries):
            pages_str = f"{start+1}-{end+1}" if start != end else str(start+1)
            logger.info(f"  Doc {i+1}: Pages {pages_str} ({end-start+1} pages)")
        
        # Show matching details
        matching = two_stage_metrics["matching_details"]
        
        if matching["matches"]:
            logger.info("\nMatched Documents:")
            for det_idx, gt_idx, overlap in matching["matches"]:
                det_doc = two_stage_docs[det_idx]
                gt_doc = ground_truth[gt_idx]
                pages_str = f"{det_doc['pages'][0]}-{det_doc['pages'][-1]}" if len(det_doc['pages']) > 1 else str(det_doc['pages'][0])
                logger.info(f"  Detected Doc {det_idx+1} (pages {pages_str}) → Ground Truth Doc {gt_doc['id']} ({gt_doc['type']}) - {overlap:.0%} overlap")
        
        if matching["unmatched_detected"]:
            logger.info("\nFalse Positives (detected but not in ground truth):")
            for idx in matching["unmatched_detected"]:
                doc = two_stage_docs[idx]
                pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
                logger.info(f"  Doc {idx+1}: Pages {pages_str}")
        
        if matching["unmatched_gt"]:
            logger.info("\nFalse Negatives (in ground truth but not detected):")
            for idx in matching["unmatched_gt"]:
                doc = ground_truth[idx]
                pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
                logger.info(f"  Doc {doc['id']}: Pages {pages_str} - {doc['type']}")
    
    # Summary
    logger.info("\n=== Summary ===")
    
    if two_stage_success:
        logger.info(f"✓ Two-stage detection completed successfully")
        logger.info(f"  - Processing time: {two_stage_time:.1f}s ({two_stage_time/60:.1f} minutes)")
        logger.info(f"  - Documents found: {len(two_stage_boundaries)}/{len(ground_truth)}")
        logger.info(f"  - F1 Score: {two_stage_metrics['f1']:.2%}")
        
        if original_success:
            logger.info(f"  - Speedup: {speedup:.1f}x faster than baseline")
            f1_diff = two_stage_metrics['f1'] - original_metrics['f1']
            if abs(f1_diff) < 0.05:
                logger.info(f"  - Accuracy: Maintained (within 5%)")
            elif f1_diff > 0:
                logger.info(f"  - Accuracy: Improved by {f1_diff:.1%}")
            else:
                logger.info(f"  - Accuracy: Decreased by {abs(f1_diff):.1%}")
    
    pdf_doc.close()


if __name__ == "__main__":
    logger.info("Starting full PDF test with two-stage detection...")
    test_full_document()