#!/usr/bin/env python3
"""Test only the two-stage detector on full PDF."""

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
            "type": "unknown"
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


def test_two_stage_only():
    """Test only the two-stage detector on full 36-page document."""
    
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
    for doc in ground_truth[:5]:  # Show first 5
        pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
        logger.info(f"  Doc {doc['id']}: Pages {pages_str} ({len(doc['pages'])} pages) - {doc['type']}")
    if len(ground_truth) > 5:
        logger.info(f"  ... and {len(ground_truth) - 5} more documents")
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"\nPDF: {pdf_path.name} ({pdf_doc.page_count} pages)")
    
    # Test Two-stage detector
    logger.info("\n=== Testing Two-Stage Detector ===")
    
    two_stage_detector = TwoStageDetector(
        fast_model="phi3:mini",
        deep_model="llama3:8b-instruct-q5_K_M",
        window_size=3,
        confidence_threshold=0.7,
        batch_size=5,
        keep_alive_minutes=10
    )
    
    # Pre-load models
    logger.info("\nPre-loading models...")
    try:
        two_stage_detector.preload_models()
        logger.info("✓ Models pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")
    
    start_time = time.time()
    logger.info("\nStarting boundary detection...")
    
    try:
        two_stage_boundaries = two_stage_detector.detect_boundaries(pdf_doc)
        two_stage_time = time.time() - start_time
        
        logger.info(f"\n=== Results ===")
        logger.info(f"Found {len(two_stage_boundaries)} documents in {two_stage_time:.2f}s ({two_stage_time/60:.1f} minutes)")
        logger.info(f"Speed: {pdf_doc.page_count / two_stage_time:.2f} pages/second")
        
        # Show detected boundaries
        logger.info("\nDetected Documents:")
        for i, (start, end) in enumerate(two_stage_boundaries[:10]):  # Show first 10
            pages_str = f"{start+1}-{end+1}" if start != end else str(start+1)
            logger.info(f"  Doc {i+1}: Pages {pages_str} ({end-start+1} pages)")
        if len(two_stage_boundaries) > 10:
            logger.info(f"  ... and {len(two_stage_boundaries) - 10} more documents")
        
        # Calculate metrics
        two_stage_docs = boundaries_to_documents(two_stage_boundaries)
        metrics = calculate_metrics(two_stage_docs, ground_truth)
        
        logger.info(f"\n=== Accuracy Metrics ===")
        logger.info(f"Precision: {metrics['precision']:.2%}")
        logger.info(f"Recall: {metrics['recall']:.2%}")
        logger.info(f"F1 Score: {metrics['f1']:.2%}")
        logger.info(f"  - True Positives: {metrics['true_positives']}")
        logger.info(f"  - False Positives: {metrics['false_positives']}")
        logger.info(f"  - False Negatives: {metrics['false_negatives']}")
        
        # Show matching details
        matching = metrics["matching_details"]
        
        if matching["unmatched_detected"]:
            logger.info(f"\nFalse Positives (detected but not in ground truth): {len(matching['unmatched_detected'])}")
            for idx in matching["unmatched_detected"][:3]:
                doc = two_stage_docs[idx]
                pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
                logger.info(f"  - Pages {pages_str}")
        
        if matching["unmatched_gt"]:
            logger.info(f"\nFalse Negatives (missed from ground truth): {len(matching['unmatched_gt'])}")
            for idx in matching["unmatched_gt"][:3]:
                doc = ground_truth[idx]
                pages_str = f"{doc['pages'][0]}-{doc['pages'][-1]}" if len(doc['pages']) > 1 else str(doc['pages'][0])
                logger.info(f"  - Pages {pages_str} ({doc['type']})")
        
        # Performance summary
        logger.info(f"\n=== Performance Summary ===")
        logger.info(f"✓ Processing completed successfully")
        logger.info(f"  - Time: {two_stage_time:.1f}s ({two_stage_time/60:.1f} minutes)")
        logger.info(f"  - Speed: {pdf_doc.page_count / two_stage_time:.1f} pages/second")
        logger.info(f"  - Documents: {len(two_stage_boundaries)} detected / {len(ground_truth)} ground truth")
        logger.info(f"  - F1 Score: {metrics['f1']:.2%}")
        
        # Based on earlier estimates
        baseline_time_estimate = pdf_doc.page_count * 100 / 3  # ~100s per 3-page window
        speedup_estimate = baseline_time_estimate / two_stage_time
        logger.info(f"  - Estimated speedup vs baseline: {speedup_estimate:.1f}x")
        
    except Exception as e:
        logger.error(f"Two-stage detector failed: {e}")
        import traceback
        traceback.print_exc()
    
    pdf_doc.close()


if __name__ == "__main__":
    logger.info("Starting two-stage detection test on full PDF...")
    test_two_stage_only()