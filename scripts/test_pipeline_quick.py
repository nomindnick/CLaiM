#!/usr/bin/env python3
"""
Quick pipeline evaluation script for Test_PDF_Set_1.pdf
Tests pattern-based boundary detection and classification without heavy visual processing.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

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
        print(f"Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🔍 Quick Pipeline Test - Test_PDF_Set_1.pdf")
    print("=" * 50)
    
    # Get basic PDF info
    pdf_doc = fitz.open(test_pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    print(f"📄 Total pages: {total_pages}")
    
    # Initialize PDFSplitter with pattern-based detection only (faster)
    pdf_splitter = PDFSplitter(use_visual_detection=False)
    
    try:
        print("⚡ Processing with pattern-based detection only...")
        request = PDFProcessingRequest(
            file_path=test_pdf_path,
            user_id="test_user"
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        print(f"✅ Extracted {len(documents)} documents in {result.processing_time:.1f}s")
        
        # Show detected documents
        print("\n📋 Detected Documents:")
        for i, doc in enumerate(documents):
            start_page = doc.page_range[0]
            end_page = doc.page_range[1]
            doc_type = doc.type.value
            print(f"{i+1:2d}. Pages {start_page:2d}-{end_page:2d} | {doc_type:20s}")
        
        # Quick comparison with ground truth
        print(f"\n🎯 Quick Analysis:")
        expected_docs = len(GROUND_TRUTH["documents"])
        actual_docs = len(documents)
        print(f"   Expected documents: {expected_docs}")
        print(f"   Detected documents: {actual_docs}")
        print(f"   Document count match: {'✅' if actual_docs == expected_docs else '❌'}")
        
        # Check page coverage
        covered_pages = set()
        for doc in documents:
            covered_pages.update(range(doc.page_range[0], doc.page_range[1] + 1))
        
        coverage = len(covered_pages) / total_pages * 100
        print(f"   Page coverage: {coverage:.1f}% ({len(covered_pages)}/{total_pages} pages)")
        print(f"   Coverage status: {'✅' if coverage == 100 else '❌'}")
        
        # Quick boundary accuracy check
        expected_boundaries = [parse_page_range(doc["pages"]) for doc in GROUND_TRUTH["documents"]]
        actual_boundaries = [doc.page_range for doc in documents]
        
        exact_matches = sum(1 for boundary in actual_boundaries if boundary in expected_boundaries)
        boundary_accuracy = exact_matches / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"   Boundary accuracy: {boundary_accuracy:.1f}% ({exact_matches}/{len(expected_boundaries)} exact matches)")
        
        # Quick classification check
        # Create mapping of expected page ranges to types
        expected_types = {}
        for doc in GROUND_TRUTH["documents"]:
            expected_types[parse_page_range(doc["pages"])] = doc["type"]
        
        correct_classifications = 0
        for doc in documents:
            expected_type = expected_types.get(doc.page_range, "Unknown")
            # Map document types to expected format
            actual_type = doc.type.value
            
            # Basic type mapping for comparison
            type_mapping = {
                "email": "Email",
                "submittal": "Submittal", 
                "schedule": "Schedule of Values",
                "payment_application": "Application for Payment",
                "invoice": "Invoice",
                "rfi": "Request for Information",
                "drawing": "Plans and Specifications",
                "change_order": "Cost Proposal"
            }
            
            mapped_type = type_mapping.get(actual_type, actual_type.title())
            
            if expected_type in mapped_type or mapped_type in expected_type:
                correct_classifications += 1
        
        classification_accuracy = correct_classifications / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"   Classification accuracy: {classification_accuracy:.1f}% ({correct_classifications}/{len(expected_boundaries)} correct)")
        
        # Overall score
        overall_score = (coverage + boundary_accuracy + classification_accuracy) / 3
        print(f"\n🎯 Overall Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("👍 Good performance!")
        elif overall_score >= 60:
            print("⚠️  Needs improvement")
        else:
            print("🚨 Significant issues detected")
        
        # Show major gaps
        print(f"\n🔧 Priority Improvements Needed:")
        if coverage < 100:
            print(f"   1. Page coverage: Missing {total_pages - len(covered_pages)} pages")
        if boundary_accuracy < 80:
            print(f"   2. Boundary detection: Only {boundary_accuracy:.1f}% accuracy")
        if classification_accuracy < 80:
            print(f"   3. Document classification: Only {classification_accuracy:.1f}% accuracy")
        
        print(f"\n💡 Next steps: Focus on improving the lowest scoring area first.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()