#!/usr/bin/env python3
"""
Detailed boundary detection test to understand why boundaries are being missed.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest

# Test files
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF_1 = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"  # Non-searchable
TEST_PDF_2 = PROJECT_ROOT / "tests" / "Test_PDF_Set_2.pdf"  # Searchable

def analyze_boundaries(pdf_path: Path):
    """Analyze boundary detection for a PDF."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {pdf_path.name}")
    print(f"{'='*80}")
    
    # Initialize splitter
    splitter = PDFSplitter()
    
    # Create processing request
    request = PDFProcessingRequest(
        file_path=pdf_path,
        perform_ocr=True,
        privacy_mode="local"
    )
    
    # Process PDF
    result = splitter.process_pdf(request)
    
    if not result.success:
        print(f"ERROR: {result.error}")
        return
    
    print(f"\nTotal pages: {result.total_pages}")
    print(f"Documents found: {result.documents_found}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    # Show detailed boundary information
    print("\nDocument boundaries detected:")
    for i, doc in enumerate(result.documents):
        start_page, end_page = doc.page_range
        page_count = end_page - start_page + 1
        print(f"\nDocument {i+1}:")
        print(f"  - Pages: {start_page}-{end_page} ({page_count} pages)")
        print(f"  - Type: {doc.type.value}")
        print(f"  - Classification confidence: {doc.classification_confidence:.2f}")
        
        # Show first few lines of text
        text_lines = doc.text.strip().split('\n')[:5]
        print(f"  - First few lines:")
        for line in text_lines:
            if line.strip():
                print(f"    {line[:100]}...")
    
    # Show gaps between documents
    print("\nGaps in document detection:")
    last_end = 0
    for i, doc in enumerate(result.documents):
        start_page, end_page = doc.page_range
        if start_page > last_end + 1:
            gap_pages = list(range(last_end + 1, start_page))
            print(f"  - Gap: pages {gap_pages[0]}-{gap_pages[-1]} ({len(gap_pages)} pages)")
        last_end = end_page
    
    # Check if we're missing pages at the end
    if last_end < result.total_pages:
        gap_pages = list(range(last_end + 1, result.total_pages + 1))
        print(f"  - Gap: pages {gap_pages[0]}-{gap_pages[-1]} ({len(gap_pages)} pages)")

def main():
    """Main test function."""
    print("Boundary Detection Analysis")
    
    # Test both PDFs
    for pdf_path in [TEST_PDF_1, TEST_PDF_2]:
        if pdf_path.exists():
            analyze_boundaries(pdf_path)
        else:
            print(f"ERROR: {pdf_path} not found")

if __name__ == "__main__":
    main()