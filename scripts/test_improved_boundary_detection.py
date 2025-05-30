#!/usr/bin/env python3
"""
Test improved boundary detection with enhanced OCR.

This tests whether the improved OCR handler results in better
boundary detection on Test_PDF_Set_1.pdf.
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
    
    print("🚀 Testing Improved Boundary Detection")
    print("=" * 60)
    print(f"📄 PDF: Test_PDF_Set_1.pdf")
    
    # Get basic PDF info
    pdf_doc = fitz.open(test_pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    print(f"📊 Total pages: {total_pages}")
    print(f"🎯 Expected documents: {len(GROUND_TRUTH['documents'])}")
    
    print(f"\n🔬 Testing Multiple Detection Approaches:")
    print("-" * 50)
    
    results = {}
    
    # Test 1: Pattern-based detection with improved OCR
    print("1️⃣ Pattern-based detection with IMPROVED OCR")
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=False)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        results['pattern_improved'] = {
            'documents_found': len(documents),
            'processing_time': result.processing_time,
            'boundaries': [doc.page_range for doc in documents],
            'types': [doc.type.value for doc in documents]
        }
        
        print(f"   ✅ Found {len(documents)} documents in {result.processing_time:.1f}s")
        for i, doc in enumerate(documents):
            start, end = doc.page_range
            print(f"   📄 Doc {i+1}: Pages {start}-{end} ({doc.type.value})")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['pattern_improved'] = {'documents_found': 0, 'error': str(e)}
    
    # Test 2: Visual detection with improved OCR
    print(f"\n2️⃣ Visual detection with IMPROVED OCR")
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=True)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        results['visual_improved'] = {
            'documents_found': len(documents),
            'processing_time': result.processing_time,
            'boundaries': [doc.page_range for doc in documents],
            'types': [doc.type.value for doc in documents]
        }
        
        print(f"   ✅ Found {len(documents)} documents in {result.processing_time:.1f}s")
        for i, doc in enumerate(documents):
            start, end = doc.page_range
            print(f"   📄 Doc {i+1}: Pages {start}-{end} ({doc.type.value})")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['visual_improved'] = {'documents_found': 0, 'error': str(e)}
    
    # Analysis and comparison
    print(f"\n📊 COMPARISON WITH GROUND TRUTH")
    print("=" * 60)
    
    expected_docs = len(GROUND_TRUTH["documents"])
    expected_boundaries = [parse_page_range(doc["pages"]) for doc in GROUND_TRUTH["documents"]]
    
    for method_name, method_results in results.items():
        if 'error' in method_results:
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
        print(f"🎯 Boundary Accuracy: {boundary_accuracy:.1f}% ({exact_matches}/{len(expected_boundaries)})")
        
        # Overall score
        overall_score = (coverage + boundary_accuracy) / 2
        print(f"🏆 Overall Score: {overall_score:.1f}%")
        print(f"⏱️  Processing Time: {method_results['processing_time']:.1f}s")
    
    # Improvement analysis
    print(f"\n📈 IMPROVEMENT ANALYSIS")
    print("=" * 40)
    
    # Compare with previous results (from final summary)
    previous_pattern = 3  # Previous pattern-based result
    previous_visual = 1   # Previous visual result
    
    if 'pattern_improved' in results and 'documents_found' in results['pattern_improved']:
        current_pattern = results['pattern_improved']['documents_found']
        pattern_improvement = ((current_pattern - previous_pattern) / previous_pattern * 100) if previous_pattern > 0 else 0
        print(f"Pattern Detection: {previous_pattern} → {current_pattern} documents ({pattern_improvement:+.1f}%)")
    
    if 'visual_improved' in results and 'documents_found' in results['visual_improved']:
        current_visual = results['visual_improved']['documents_found']
        visual_improvement = ((current_visual - previous_visual) / previous_visual * 100) if previous_visual > 0 else 0
        print(f"Visual Detection: {previous_visual} → {current_visual} documents ({visual_improvement:+.1f}%)")
    
    # Recommendation
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 30)
    
    best_method = None
    best_score = 0
    
    for method_name, method_results in results.items():
        if 'error' not in method_results and method_results['documents_found'] > 0:
            # Simple scoring: detection rate
            score = method_results['documents_found'] / expected_docs
            if score > best_score:
                best_score = score
                best_method = method_name
    
    if best_method:
        print(f"🏆 Best Method: {best_method.upper().replace('_', ' ')}")
        print(f"📊 Achievement: {best_score:.1%} detection rate")
        
        if best_score >= 0.8:
            print("🎉 Excellent! Ready for production with manual review.")
        elif best_score >= 0.6:
            print("👍 Good improvement! Consider additional enhancements.")
        elif best_score >= 0.4:
            print("⚠️  Moderate improvement. More work needed.")
        else:
            print("🚨 Limited improvement. Consider alternative approaches.")
    else:
        print("❌ No method showed significant improvement.")
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    main()