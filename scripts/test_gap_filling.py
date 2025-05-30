#!/usr/bin/env python3
"""Test script to verify gap filling works correctly."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest

def main():
    print("=== Testing Gap Filling ===\n")
    
    test_pdf = Path("tests/Test_PDF_Set_2.pdf")
    if not test_pdf.exists():
        print("❌ Test PDF not found")
        return
    
    # Initialize PDF splitter with visual detection
    splitter = PDFSplitter(use_visual_detection=True)
    
    # Create processing request
    request = PDFProcessingRequest(
        file_path=test_pdf,
        split_documents=True,
        classify_documents=False,
        extract_metadata=False,
    )
    
    print(f"Processing PDF with gap filling enabled...")
    result = splitter.process_pdf(request)
    
    print(f"\nResults:")
    print(f"  Total pages in PDF: {result.total_pages}")
    print(f"  Documents extracted: {len(result.documents)}")
    print(f"  Detection level used: {result.detection_level}")
    
    # Analyze page coverage
    all_covered_pages = set()
    total_extracted_pages = 0
    
    print(f"\nDocument breakdown:")
    for i, doc in enumerate(result.documents):
        page_count = doc.page_range[1] - doc.page_range[0] + 1
        total_extracted_pages += page_count
        pages = list(range(doc.page_range[0], doc.page_range[1] + 1))
        all_covered_pages.update(pages)
        
        print(f"  Document {i+1}: pages {doc.page_range[0]}-{doc.page_range[1]} ({page_count} pages)")
        print(f"    Type: {doc.type.value}")
        # Note: title is not available in Document model during processing
    
    # Check coverage
    all_pages = set(range(1, result.total_pages + 1))
    missing_pages = all_pages - all_covered_pages
    
    print(f"\nCoverage analysis:")
    print(f"  Total pages extracted: {total_extracted_pages}/{result.total_pages}")
    print(f"  Coverage percentage: {total_extracted_pages/result.total_pages*100:.1f}%")
    
    if missing_pages:
        print(f"  ❌ Missing pages: {sorted(missing_pages)}")
        print("  Gap filling may not be working correctly")
    else:
        print(f"  ✅ All pages covered!")
        print("  Gap filling is working correctly")
    
    # Check for warnings
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

if __name__ == "__main__":
    main()