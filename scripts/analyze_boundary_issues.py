#!/usr/bin/env python3
"""Analyze specific boundary detection issues in the test PDF."""

import sys
from pathlib import Path
import fitz

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
from modules.document_processor.construction_patterns import detect_document_type


def main():
    """Analyze specific pages to understand boundary detection issues."""
    pdf_path = Path(__file__).parent.parent / "tests" / "Test_PDF_Set_1.pdf"
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    pdf_doc = fitz.open(str(pdf_path))
    ocr_handler = ImprovedOCRHandler()
    
    # Analyze specific problematic pages
    problem_pages = [
        (4, 5, "Should be boundary between two email chains"),
        (6, 7, "Should be boundary between email and submittal"),
        (12, 13, "Should be boundary between SOV and email"),
        (17, 18, "Should be boundary between payment app and invoice"),
        (19, 20, "Should be boundary between two different invoices"),
        (22, 23, "Should be boundary between invoice and RFI"),
        (25, 26, "Should be boundary between RFI and drawings"),
        (31, 32, "Should be boundary between drawings and cost proposal"),
    ]
    
    print("ANALYZING BOUNDARY DETECTION ISSUES")
    print("=" * 60)
    
    for page1_num, page2_num, description in problem_pages:
        print(f"\nPages {page1_num} and {page2_num}: {description}")
        print("-" * 50)
        
        # Get text from both pages
        page1 = pdf_doc[page1_num - 1]  # 0-indexed
        page2 = pdf_doc[page2_num - 1]
        
        text1 = page1.get_text()
        text2 = page2.get_text()
        
        # Use OCR if needed
        if len(text1.strip()) < 10:
            try:
                text1, _ = ocr_handler.process_page(page1, dpi=200)
            except:
                pass
        
        if len(text2.strip()) < 10:
            try:
                text2, _ = ocr_handler.process_page(page2, dpi=200)
            except:
                pass
        
        # Analyze document types
        types1 = detect_document_type(text1)
        types2 = detect_document_type(text2)
        
        print(f"Page {page1_num}:")
        print(f"  Document types: {types1}")
        print(f"  First 200 chars: {repr(text1[:200])}")
        
        print(f"\nPage {page2_num}:")
        print(f"  Document types: {types2}")
        print(f"  First 200 chars: {repr(text2[:200])}")
        
        # Check for specific patterns
        # Email patterns
        if "From:" in text2[:500] or "Sent:" in text2[:500]:
            print(f"  ✓ Page {page2_num} has email headers")
        
        # Document headers
        headers = ["SUBMITTAL", "PACKING SLIP", "REQUEST FOR INFORMATION", "COST PROPOSAL", "APPLICATION"]
        for header in headers:
            if header in text2[:500].upper():
                print(f"  ✓ Page {page2_num} has {header}")
        
        # Page numbering
        if "Page 1" in text2:
            print(f"  ✓ Page {page2_num} has 'Page 1' indicator")
    
    pdf_doc.close()


if __name__ == "__main__":
    main()