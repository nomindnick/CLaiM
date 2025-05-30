#!/usr/bin/env python3
"""
Test script using visual boundary detection instead of pattern-based detection.
This should be more reliable for poor-quality scanned documents.
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
        print(f"Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🔍 Testing Visual Boundary Detection - Test_PDF_Set_1.pdf")
    print("=" * 60)
    
    # Get basic PDF info
    pdf_doc = fitz.open(test_pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    print(f"📄 Total pages: {total_pages}")
    
    # Initialize PDFSplitter with VISUAL detection enabled
    print("🎯 Using VISUAL boundary detection (more reliable for scanned docs)")
    pdf_splitter = PDFSplitter(use_visual_detection=True)
    
    try:
        print("🔮 Processing with visual boundary detection...")
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
        
        # Compare with ground truth
        print(f"\n🎯 Comparison with Ground Truth:")
        print("-" * 40)
        
        expected_docs = len(GROUND_TRUTH["documents"])
        actual_docs = len(documents)
        print(f"Expected documents: {expected_docs}")
        print(f"Detected documents: {actual_docs}")
        
        # Check page coverage
        covered_pages = set()
        for doc in documents:
            covered_pages.update(range(doc.page_range[0], doc.page_range[1] + 1))
        
        coverage = len(covered_pages) / total_pages * 100
        print(f"Page coverage: {coverage:.1f}% ({len(covered_pages)}/{total_pages} pages)")
        
        # Calculate boundary accuracy
        expected_boundaries = [parse_page_range(doc["pages"]) for doc in GROUND_TRUTH["documents"]]
        actual_boundaries = [doc.page_range for doc in documents]
        
        exact_matches = sum(1 for boundary in actual_boundaries if boundary in expected_boundaries)
        boundary_accuracy = exact_matches / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"Boundary accuracy: {boundary_accuracy:.1f}% ({exact_matches}/{len(expected_boundaries)} exact matches)")
        
        # Overall score
        overall_score = (coverage + boundary_accuracy) / 2
        print(f"\n🎯 Overall Score: {overall_score:.1f}%")
        
        # Show improvements vs pattern-based
        pattern_docs = 3  # Previous result
        improvement = ((actual_docs - pattern_docs) / pattern_docs * 100) if pattern_docs > 0 else 0
        print(f"📈 Improvement over pattern-based: {improvement:+.1f}% more documents detected")
        
        if overall_score >= 80:
            print("🎉 Excellent! Visual detection is working well.")
        elif overall_score >= 60:
            print("👍 Good improvement! Visual detection is more effective.")
        elif actual_docs > pattern_docs:
            print("⚠️  Better than pattern detection, but still needs work.")
        else:
            print("🚨 Visual detection not performing better than patterns.")
        
        # Detailed comparison table
        print(f"\n📊 Expected vs Actual Boundaries:")
        print("-" * 60)
        print(f"{'Expected':<20} {'Actual':<20} {'Match':<10}")
        print("-" * 60)
        
        for i, expected_doc in enumerate(GROUND_TRUTH["documents"]):
            exp_pages = expected_doc["pages"]
            exp_type = expected_doc["type"]
            
            # Find closest actual document
            exp_start, exp_end = parse_page_range(exp_pages)
            best_actual = None
            best_overlap = 0
            
            for doc in documents:
                act_start = doc.page_range[0]
                act_end = doc.page_range[1]
                
                overlap_start = max(exp_start, act_start)
                overlap_end = min(exp_end, act_end)
                overlap = max(0, overlap_end - overlap_start + 1)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_actual = doc
            
            if best_actual:
                actual_pages = f"{best_actual.page_range[0]}-{best_actual.page_range[1]}"
                match_status = "✅ Yes" if exp_pages == actual_pages else "❌ No"
            else:
                actual_pages = "NOT FOUND"
                match_status = "❌ No"
            
            print(f"{exp_pages:<20} {actual_pages:<20} {match_status:<10}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()