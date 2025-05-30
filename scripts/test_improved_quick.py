#!/usr/bin/env python3
"""
Quick test of improved boundary detection - just pattern-based to see results faster.
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
    
    print("🚀 Quick Test: Improved Pattern-Based Boundary Detection")
    print("=" * 60)
    print(f"📄 PDF: Test_PDF_Set_1.pdf")
    
    # Get basic PDF info
    pdf_doc = fitz.open(test_pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    print(f"📊 Total pages: {total_pages}")
    print(f"🎯 Expected documents: {len(GROUND_TRUTH['documents'])}")
    
    print(f"\n🔬 Testing Pattern-based Detection with Improved OCR...")
    print("-" * 50)
    
    try:
        # Use pattern-based detection only (faster)
        pdf_splitter = PDFSplitter(use_visual_detection=False)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        print(f"✅ Processing complete in {result.processing_time:.1f}s")
        print(f"📄 Found {len(documents)} documents")
        
        print(f"\n📋 Detected Documents:")
        for i, doc in enumerate(documents):
            start, end = doc.page_range
            doc_type = doc.type.value
            page_count = end - start + 1
            print(f"{i+1:2d}. Pages {start:2d}-{end:2d} ({page_count:2d} pages) | {doc_type}")
        
        # Analysis
        print(f"\n📊 ANALYSIS vs GROUND TRUTH")
        print("-" * 40)
        
        expected_docs = len(GROUND_TRUTH["documents"])
        expected_boundaries = [parse_page_range(doc["pages"]) for doc in GROUND_TRUTH["documents"]]
        actual_boundaries = [doc.page_range for doc in documents]
        
        # Detection rate
        detection_rate = (len(documents) / expected_docs * 100) if expected_docs > 0 else 0
        print(f"📈 Detection Rate: {detection_rate:.1f}% ({len(documents)}/{expected_docs} documents)")
        
        # Page coverage
        covered_pages = set()
        for start, end in actual_boundaries:
            covered_pages.update(range(start, end + 1))
        coverage = len(covered_pages) / total_pages * 100
        print(f"📄 Page Coverage: {coverage:.1f}% ({len(covered_pages)}/{total_pages} pages)")
        
        # Boundary accuracy
        exact_matches = sum(1 for boundary in actual_boundaries if boundary in expected_boundaries)
        boundary_accuracy = exact_matches / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"🎯 Boundary Accuracy: {boundary_accuracy:.1f}% ({exact_matches}/{len(expected_boundaries)} exact matches)")
        
        # Overall score
        overall_score = (coverage + boundary_accuracy) / 2
        print(f"🏆 Overall Score: {overall_score:.1f}%")
        
        # Compare with previous results
        previous_docs = 3  # Previous pattern-based result
        improvement = ((len(documents) - previous_docs) / previous_docs * 100) if previous_docs > 0 else 0
        print(f"📈 Improvement: {previous_docs} → {len(documents)} documents ({improvement:+.1f}%)")
        
        # Success assessment
        if detection_rate >= 80:
            print("🎉 Excellent! Ready for production testing.")
        elif detection_rate >= 60:
            print("👍 Good improvement! Manual review recommended.")
        elif detection_rate >= 40:
            print("⚠️  Moderate improvement. Additional work needed.")
        else:
            print("🚨 Limited improvement from OCR enhancements.")
        
        # Show boundaries that match
        print(f"\n✅ EXACT BOUNDARY MATCHES:")
        matches_found = 0
        for i, expected_doc in enumerate(GROUND_TRUTH["documents"]):
            exp_range = parse_page_range(expected_doc["pages"])
            exp_type = expected_doc["type"]
            
            if exp_range in actual_boundaries:
                matches_found += 1
                print(f"   ✅ Pages {exp_range[0]}-{exp_range[1]}: {exp_type}")
        
        if matches_found == 0:
            print("   ❌ No exact boundary matches found")
        
        print(f"\n✅ Quick test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()