#!/usr/bin/env python3
"""Test improved boundary detection against ground truth."""

import json
import sys
from pathlib import Path

import fitz
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.boundary_detector import BoundaryDetector
from modules.document_processor.improved_boundary_detector import ImprovedBoundaryDetector
from modules.document_processor.refined_boundary_detector import RefinedBoundaryDetector
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler


def parse_page_range(page_range: str) -> tuple:
    """Parse page range like '1-4' to (0, 3) for 0-indexed pages."""
    if '-' in page_range:
        start, end = page_range.split('-')
        return int(start) - 1, int(end) - 1
    else:
        # Single page
        page = int(page_range) - 1
        return page, page


def load_ground_truth(path: str) -> list:
    """Load ground truth boundaries."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    boundaries = []
    for doc in data['documents']:
        start, end = parse_page_range(doc['pages'])
        boundaries.append((start, end))
    
    return boundaries


def compare_boundaries(detected: list, ground_truth: list) -> dict:
    """Compare detected boundaries with ground truth."""
    # Convert to sets of pages for each document
    detected_docs = []
    for start, end in detected:
        detected_docs.append(set(range(start, end + 1)))
    
    truth_docs = []
    for start, end in ground_truth:
        truth_docs.append(set(range(start, end + 1)))
    
    # Calculate metrics
    correct_docs = 0
    partial_matches = 0
    false_positives = 0
    false_negatives = 0
    
    # Track which truth docs were matched
    matched_truth = set()
    
    # Check each detected document
    for i, detected_pages in enumerate(detected_docs):
        best_match = None
        best_overlap = 0
        
        # Find best matching truth document
        for j, truth_pages in enumerate(truth_docs):
            overlap = len(detected_pages & truth_pages)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = j
        
        if best_match is not None and best_overlap > 0:
            truth_pages = truth_docs[best_match]
            if detected_pages == truth_pages:
                correct_docs += 1
                logger.info(f"✓ Detected doc {i+1} matches truth doc {best_match+1} exactly")
            else:
                partial_matches += 1
                precision = best_overlap / len(detected_pages)
                recall = best_overlap / len(truth_pages)
                logger.warning(f"~ Detected doc {i+1} partially matches truth doc {best_match+1} "
                             f"(precision: {precision:.2f}, recall: {recall:.2f})")
            matched_truth.add(best_match)
        else:
            false_positives += 1
            pages_str = f"{min(detected_pages)+1}-{max(detected_pages)+1}"
            logger.error(f"✗ Detected doc {i+1} (pages {pages_str}) has no match in ground truth")
    
    # Check for missed truth documents
    for j in range(len(truth_docs)):
        if j not in matched_truth:
            false_negatives += 1
            truth_pages = truth_docs[j]
            pages_str = f"{min(truth_pages)+1}-{max(truth_pages)+1}"
            logger.error(f"✗ Truth doc {j+1} (pages {pages_str}) was not detected")
    
    # Calculate overall metrics
    total_detected = len(detected_docs)
    total_truth = len(truth_docs)
    
    precision = correct_docs / total_detected if total_detected > 0 else 0
    recall = correct_docs / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'correct_docs': correct_docs,
        'partial_matches': partial_matches,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_detected': total_detected,
        'total_truth': total_truth,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    """Test boundary detection improvements."""
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Paths
    pdf_path = Path(__file__).parent.parent / "tests" / "Test_PDF_Set_1.pdf"
    ground_truth_path = Path(__file__).parent.parent / "tests" / "Test_PDF_Set_Ground_Truth.json"
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    if not ground_truth_path.exists():
        logger.error(f"Ground truth not found: {ground_truth_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth(str(ground_truth_path))
    logger.info(f"Ground truth contains {len(ground_truth)} documents")
    
    # Open PDF
    pdf_doc = fitz.open(str(pdf_path))
    logger.info(f"PDF has {pdf_doc.page_count} pages")
    
    # Initialize OCR handler
    ocr_handler = ImprovedOCRHandler()
    
    # Test original detector
    logger.info("\n" + "="*60)
    logger.info("Testing ORIGINAL boundary detector")
    logger.info("="*60)
    
    original_detector = BoundaryDetector(ocr_handler=ocr_handler)
    original_boundaries = original_detector.detect_boundaries(pdf_doc)
    
    logger.info(f"\nOriginal detector found {len(original_boundaries)} documents:")
    for i, (start, end) in enumerate(original_boundaries):
        logger.info(f"  Document {i+1}: pages {start+1}-{end+1}")
    
    original_metrics = compare_boundaries(original_boundaries, ground_truth)
    
    # Test improved detector
    logger.info("\n" + "="*60)
    logger.info("Testing IMPROVED boundary detector")
    logger.info("="*60)
    
    improved_detector = ImprovedBoundaryDetector(ocr_handler=ocr_handler)
    improved_boundaries = improved_detector.detect_boundaries(pdf_doc)
    
    logger.info(f"\nImproved detector found {len(improved_boundaries)} documents:")
    for i, (start, end) in enumerate(improved_boundaries):
        logger.info(f"  Document {i+1}: pages {start+1}-{end+1}")
    
    improved_metrics = compare_boundaries(improved_boundaries, ground_truth)
    
    # Test refined detector
    logger.info("\n" + "="*60)
    logger.info("Testing REFINED boundary detector")
    logger.info("="*60)
    
    refined_detector = RefinedBoundaryDetector(ocr_handler=ocr_handler)
    refined_boundaries = refined_detector.detect_boundaries(pdf_doc)
    
    logger.info(f"\nRefined detector found {len(refined_boundaries)} documents:")
    for i, (start, end) in enumerate(refined_boundaries):
        logger.info(f"  Document {i+1}: pages {start+1}-{end+1}")
    
    refined_metrics = compare_boundaries(refined_boundaries, ground_truth)
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    logger.info("\nOriginal Detector:")
    logger.info(f"  Documents: {original_metrics['total_detected']} detected, {original_metrics['total_truth']} expected")
    logger.info(f"  Correct: {original_metrics['correct_docs']}")
    logger.info(f"  Partial: {original_metrics['partial_matches']}")
    logger.info(f"  False Positives: {original_metrics['false_positives']}")
    logger.info(f"  False Negatives: {original_metrics['false_negatives']}")
    logger.info(f"  Precision: {original_metrics['precision']:.2%}")
    logger.info(f"  Recall: {original_metrics['recall']:.2%}")
    logger.info(f"  F1 Score: {original_metrics['f1_score']:.2%}")
    
    logger.info("\nImproved Detector:")
    logger.info(f"  Documents: {improved_metrics['total_detected']} detected, {improved_metrics['total_truth']} expected")
    logger.info(f"  Correct: {improved_metrics['correct_docs']}")
    logger.info(f"  Partial: {improved_metrics['partial_matches']}")
    logger.info(f"  False Positives: {improved_metrics['false_positives']}")
    logger.info(f"  False Negatives: {improved_metrics['false_negatives']}")
    logger.info(f"  Precision: {improved_metrics['precision']:.2%}")
    logger.info(f"  Recall: {improved_metrics['recall']:.2%}")
    logger.info(f"  F1 Score: {improved_metrics['f1_score']:.2%}")
    
    logger.info("\nRefined Detector:")
    logger.info(f"  Documents: {refined_metrics['total_detected']} detected, {refined_metrics['total_truth']} expected")
    logger.info(f"  Correct: {refined_metrics['correct_docs']}")
    logger.info(f"  Partial: {refined_metrics['partial_matches']}")
    logger.info(f"  False Positives: {refined_metrics['false_positives']}")
    logger.info(f"  False Negatives: {refined_metrics['false_negatives']}")
    logger.info(f"  Precision: {refined_metrics['precision']:.2%}")
    logger.info(f"  Recall: {refined_metrics['recall']:.2%}")
    logger.info(f"  F1 Score: {refined_metrics['f1_score']:.2%}")
    
    # Show improvement
    logger.info("\nImprovement Summary:")
    logger.info("Original → Improved → Refined:")
    logger.info(f"  Precision: {original_metrics['precision']:.2%} → {improved_metrics['precision']:.2%} → {refined_metrics['precision']:.2%}")
    logger.info(f"  Recall: {original_metrics['recall']:.2%} → {improved_metrics['recall']:.2%} → {refined_metrics['recall']:.2%}")
    logger.info(f"  F1 Score: {original_metrics['f1_score']:.2%} → {improved_metrics['f1_score']:.2%} → {refined_metrics['f1_score']:.2%}")


if __name__ == "__main__":
    main()