#!/usr/bin/env python3
"""
Test only boundary detection without classification.
"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

from modules.document_processor.improved_boundary_detector import ImprovedBoundaryDetector
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
import fitz

# Test files
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF_1 = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"
TEST_PDF_2 = PROJECT_ROOT / "tests" / "Test_PDF_Set_2.pdf"
GROUND_TRUTH = PROJECT_ROOT / "tests" / "Test_PDF_Set_Ground_Truth.json"

def test_boundary_detection(pdf_path: Path):
    """Test boundary detection on a PDF."""
    print(f"\nTesting: {pdf_path.name}")
    print("="*60)
    
    # Load ground truth
    with open(GROUND_TRUTH, 'r') as f:
        ground_truth = json.load(f)['documents']
    
    expected_boundaries = []
    for doc in ground_truth:
        pages = doc['pages']
        if '-' in pages:
            start, end = pages.split('-')
            expected_boundaries.append((int(start), int(end)))
        else:
            page = int(pages)
            expected_boundaries.append((page, page))
    
    # Open PDF
    pdf_doc = fitz.open(str(pdf_path))
    
    # Initialize detectors
    ocr_handler = ImprovedOCRHandler(min_confidence=0.4)
    detector = ImprovedBoundaryDetector(ocr_handler)
    
    # Detect boundaries
    boundaries = detector.detect_boundaries(pdf_doc)
    
    print(f"Expected: {len(expected_boundaries)} documents")
    print(f"Found: {len(boundaries)} documents")
    
    # Show detected boundaries
    print("\nDetected boundaries:")
    for i, (start, end) in enumerate(boundaries):
        print(f"  Document {i+1}: pages {start+1}-{end+1}")
    
    # Compare to expected
    print("\nComparison to ground truth:")
    correct = 0
    for i, (exp_start, exp_end) in enumerate(expected_boundaries):
        found = False
        for (det_start, det_end) in boundaries:
            if det_start+1 == exp_start and det_end+1 == exp_end:
                found = True
                correct += 1
                break
        status = "✓" if found else "✗"
        print(f"{status} Document {i+1}: expected pages {exp_start}-{exp_end}")
    
    accuracy = (correct / len(expected_boundaries)) * 100
    print(f"\nAccuracy: {correct}/{len(expected_boundaries)} ({accuracy:.1f}%)")
    
    pdf_doc.close()
    return boundaries, expected_boundaries

def main():
    """Test boundary detection on both PDFs."""
    print("Boundary Detection Test")
    print("="*80)
    
    # Test both PDFs
    for pdf_path in [TEST_PDF_1, TEST_PDF_2]:
        if pdf_path.exists():
            test_boundary_detection(pdf_path)
        else:
            print(f"ERROR: {pdf_path} not found")

if __name__ == "__main__":
    main()