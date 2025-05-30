#!/usr/bin/env python3
"""
Test the fixed boundary detection logic.

This tests whether the improved merge logic correctly preserves
high-confidence pattern detection results.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest
import fitz

# Ground truth data
GROUND_TRUTH = {
    "documents": [
        {"pages": "1-4", "type": "Email Chain"},
        {"pages": "5-6", "type": "Email Chain"},
        {"pages": "7-8", "type": "Submittal"},
        {"pages": "9-12", "type": "Schedule of Values"},
        {"pages": "13", "type": "Email"},
        {"pages": "14-17", "type": "Application for Payment"},
        {"pages": "18-19", "type": "Invoice"},
        {"pages": "20-22", "type": "Invoice"},
        {"pages": "23-25", "type": "Request for Information"},
        {"pages": "26-31", "type": "Plans and Specifications"},
        {"pages": "32-33", "type": "Cost Proposal"},
        {"pages": "34", "type": "Cost Proposal"},
        {"pages": "35", "type": "Cost Proposal"},
        {"pages": "36", "type": "Email"}
    ]
}


def parse_page_range(page_range: str) -> tuple:
    """Parse page range string like '1-4' into (start, end) tuple."""
    if '-' in page_range:
        start, end = page_range.split('-')
        return (int(start), int(end))
    else:
        page = int(page_range)
        return (page, page)


def main():
    """Main test execution."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🚀 Testing FIXED Boundary Detection Logic")
    print("=" * 60)
    print(f"📄 PDF: Test_PDF_Set_1.pdf")
    
    # Get basic PDF info
    pdf_doc = fitz.open(test_pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    print(f"📊 Total pages: {total_pages}")
    print(f"🎯 Expected documents: {len(GROUND_TRUTH['documents'])}")
    
    results = {}
    
    # Test 1: Pattern-only detection (baseline)
    print(f"\n🔬 Test 1: Pattern-only detection")
    print("-" * 50)
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=False)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True,
            classify_documents=False  # Skip classification for faster testing
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        results['pattern_only'] = {
            'documents_found': len(documents),
            'processing_time': result.processing_time,
            'boundaries': [doc.page_range for doc in documents]
        }
        
        print(f"✅ Found {len(documents)} documents in {result.processing_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['pattern_only'] = {'documents_found': 0, 'error': str(e)}
    
    # Test 2: Fixed hybrid detection
    print(f"\n🔬 Test 2: FIXED Hybrid detection")
    print("-" * 50)
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=True)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True,
            classify_documents=False  # Skip classification for faster testing
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        results['hybrid_fixed'] = {
            'documents_found': len(documents),
            'processing_time': result.processing_time,
            'boundaries': [doc.page_range for doc in documents],
            'detection_level': getattr(result, 'detection_level', 'unknown')
        }
        
        print(f"✅ Found {len(documents)} documents in {result.processing_time:.1f}s")
        print(f"📈 Detection method: {getattr(result, 'detection_level', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['hybrid_fixed'] = {'documents_found': 0, 'error': str(e)}
    
    # Analysis
    print(f"\n📊 RESULTS COMPARISON")
    print("=" * 60)
    
    expected_docs = len(GROUND_TRUTH["documents"])
    expected_boundaries = [parse_page_range(doc["pages"]) for doc in GROUND_TRUTH["documents"]]
    
    for method_name, method_results in results.items():
        if 'error' in method_results:
            print(f"\n❌ {method_name.upper().replace('_', ' ')}: ERROR")
            print(f"   {method_results['error']}")
            continue
            
        print(f"\n🔍 {method_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        actual_docs = method_results['documents_found']
        actual_boundaries = method_results['boundaries']
        
        # Detection rate
        detection_rate = (actual_docs / expected_docs * 100) if expected_docs > 0 else 0
        print(f"📈 Detection Rate: {detection_rate:.1f}% ({actual_docs}/{expected_docs})")
        
        # Page coverage
        covered_pages = set()
        for start, end in actual_boundaries:
            covered_pages.update(range(start, end + 1))
        coverage = len(covered_pages) / total_pages * 100
        print(f"📄 Page Coverage: {coverage:.1f}% ({len(covered_pages)}/{total_pages})")
        
        # Boundary accuracy
        exact_matches = sum(1 for boundary in actual_boundaries if boundary in expected_boundaries)
        boundary_accuracy = exact_matches / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"🎯 Boundary Accuracy: {boundary_accuracy:.1f}% ({exact_matches}/{len(expected_boundaries)} exact matches)")
        
        # Overall score
        overall_score = (coverage + boundary_accuracy) / 2
        print(f"🏆 Overall Score: {overall_score:.1f}%")
        print(f"⏱️  Processing Time: {method_results['processing_time']:.1f}s")
        
        if 'detection_level' in method_results:
            print(f"🔍 Detection Level: {method_results['detection_level']}")
    
    # Fix assessment
    print(f"\n🔧 FIX ASSESSMENT")
    print("=" * 40)
    
    if 'pattern_only' in results and 'hybrid_fixed' in results:
        pattern_docs = results['pattern_only'].get('documents_found', 0)
        hybrid_docs = results['hybrid_fixed'].get('documents_found', 0)
        
        print(f"Pattern-only detection: {pattern_docs} documents")
        print(f"Fixed hybrid detection: {hybrid_docs} documents")
        
        if hybrid_docs >= pattern_docs * 0.8:  # Allow some variation
            print("✅ SUCCESS: Fixed hybrid preserves high-quality pattern detection!")
            if hybrid_docs == pattern_docs:
                print("🎉 PERFECT: Hybrid exactly matches pattern detection!")
            elif hybrid_docs > pattern_docs * 0.9:
                print("👍 EXCELLENT: Hybrid preserves most pattern boundaries!")
        elif hybrid_docs > pattern_docs * 0.5:
            print("⚠️  PARTIAL: Some improvement but still losing boundaries")
        else:
            print("❌ FAILED: Fix didn't work - still losing most boundaries")
        
        # Check if we're now using pattern detection
        detection_level = results['hybrid_fixed'].get('detection_level', 'unknown')
        if detection_level == 'heuristic':
            print("✅ EXCELLENT: System correctly chose pattern detection over visual!")
        elif detection_level == 'visual':
            print("⚠️  WARNING: Still using visual detection - check merge logic")
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    main()