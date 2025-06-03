#!/usr/bin/env python3
"""
Debug boundary detection to see why it's detecting too many boundaries.
"""

import os
import sys
import fitz
from pathlib import Path

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

from modules.document_processor.improved_boundary_detector import ImprovedBoundaryDetector
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler

# Test file
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"

def debug_boundaries():
    """Debug boundary detection."""
    print("Debugging boundary detection")
    print("="*80)
    
    # Open PDF
    pdf_doc = fitz.open(str(TEST_PDF))
    print(f"PDF has {pdf_doc.page_count} pages")
    
    # Initialize OCR handler
    ocr_handler = ImprovedOCRHandler(min_confidence=0.4)
    
    # Initialize improved boundary detector
    detector = ImprovedBoundaryDetector(ocr_handler)
    
    # Test first few pages
    for page_num in range(min(10, pdf_doc.page_count)):
        page = pdf_doc[page_num]
        
        # Get text (using OCR if needed)
        text = page.get_text()
        if len(text.strip()) < 10:
            print(f"\nPage {page_num + 1}: Needs OCR")
            text, confidence = ocr_handler.process_page(page, dpi=200)
            print(f"  OCR confidence: {confidence:.2f}")
        else:
            print(f"\nPage {page_num + 1}: Has text")
        
        # Show first 200 chars
        print(f"  First 200 chars: {text[:200].replace(chr(10), ' ')}")
        
        # Test boundary detection
        if page_num > 0:
            is_boundary = detector._is_document_start(text, page, page_num)
            print(f"  Is document start: {is_boundary}")
            
            # Check specific conditions
            if hasattr(detector, '_page_texts'):
                prev_text = detector._page_texts.get(page_num - 1, "")
                
                # Test individual checks
                is_email_cont = detector._is_email_chain_continuation(text, prev_text)
                is_new_email = detector._is_new_email(text, prev_text)
                is_shipping_cont = detector._is_shipping_doc_continuation(text, prev_text)
                is_new_shipping = detector._is_new_shipping_document(text, prev_text)
                
                print(f"  - Email continuation: {is_email_cont}")
                print(f"  - New email: {is_new_email}")
                print(f"  - Shipping continuation: {is_shipping_cont}")
                print(f"  - New shipping: {is_new_shipping}")
    
    pdf_doc.close()

if __name__ == "__main__":
    debug_boundaries()