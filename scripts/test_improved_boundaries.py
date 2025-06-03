#!/usr/bin/env python3
"""
Quick test of improved boundary detection.
"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest

# Test files
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF_1 = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"  # Non-searchable
GROUND_TRUTH = PROJECT_ROOT / "tests" / "Test_PDF_Set_Ground_Truth.json"

def test_boundaries():
    """Test boundary detection on first PDF."""
    print("Testing improved boundary detection")
    print("="*80)
    
    # Load ground truth
    with open(GROUND_TRUTH, 'r') as f:
        ground_truth = json.load(f)['documents']
    
    print(f"Ground truth: {len(ground_truth)} documents expected")
    expected_boundaries = []
    for doc in ground_truth:
        pages = doc['pages']
        if '-' in pages:
            start, end = pages.split('-')
            expected_boundaries.append((int(start), int(end)))
        else:
            page = int(pages)
            expected_boundaries.append((page, page))
    
    print(f"Expected boundaries: {expected_boundaries}")
    
    # Test with pattern detection only
    print("\nTesting pattern-based boundary detection...")
    splitter = PDFSplitter(use_visual_detection=False)
    
    request = PDFProcessingRequest(
        file_path=TEST_PDF_1,
        perform_ocr=True,
        privacy_mode="local",
        split_documents=True  # Enable document splitting
    )
    
    # Process to get boundaries
    result = splitter.process_pdf(request)
    
    if result.success:
        print(f"Processing successful")
        print(f"Total pages: {result.total_pages}")
        print(f"Documents found: {result.documents_found}")
        
        # Show actual boundaries detected
        print("\nDocuments detected:")
        for i, doc in enumerate(result.documents):
            start, end = doc.page_range
            print(f"Document {i+1}: pages {start}-{end} ({end-start+1} pages)")
            print(f"  Type: {doc.type.value}")
            print(f"  First 100 chars: {doc.text[:100].replace(chr(10), ' ')}")
        
        # Compare to expected
        print("\nComparison to ground truth:")
        for i, expected in enumerate(expected_boundaries):
            exp_start, exp_end = expected
            found = False
            for doc in result.documents:
                doc_start, doc_end = doc.page_range
                if doc_start == exp_start and doc_end == exp_end:
                    found = True
                    break
            status = "✓" if found else "✗"
            print(f"{status} Document {i+1}: expected pages {exp_start}-{exp_end}")
    else:
        print(f"Processing failed: {result.error}")

if __name__ == "__main__":
    test_boundaries()