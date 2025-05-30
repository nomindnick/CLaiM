#!/usr/bin/env python3
"""
Debug script to see what actual OCR text is extracted from key pages.
This will help understand why boundary detection is failing.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz
from modules.document_processor.ocr_handler import OCRHandler

def main():
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"Error: Test PDF not found at {test_pdf_path}")
        return
    
    # Key pages to analyze based on ground truth boundaries
    key_pages = [1, 5, 7, 9, 13, 14, 18, 20, 23, 26, 32, 34, 35, 36]  # 0-indexed
    expected_docs = [
        "Email Chain", "Email Chain", "Submittal", "Schedule of Values", 
        "Email", "Application for Payment", "Invoice", "Invoice", 
        "Request for Information", "Plans and Specifications", 
        "Cost Proposal", "Cost Proposal", "Cost Proposal", "Email"
    ]
    
    print("🔍 Analyzing OCR text from key boundary pages")
    print("=" * 80)
    
    pdf_doc = fitz.open(test_pdf_path)
    ocr_handler = OCRHandler()
    
    for i, page_num in enumerate(key_pages):
        if page_num >= pdf_doc.page_count:
            continue
            
        page = pdf_doc[page_num]
        expected_doc = expected_docs[i]
        
        print(f"\n📄 PAGE {page_num + 1} - Expected: {expected_doc}")
        print("-" * 50)
        
        # Get direct text first
        direct_text = page.get_text().strip()
        print(f"Direct text length: {len(direct_text)} chars")
        
        if len(direct_text) > 10:
            print(f"Direct text (first 200 chars):")
            print(f"'{direct_text[:200]}...'")
        else:
            print("Direct text: (empty or very short - needs OCR)")
        
        # Get OCR text
        try:
            ocr_text, confidence = ocr_handler.process_page(page, dpi=200)
            print(f"\nOCR confidence: {confidence:.3f}")
            print(f"OCR text length: {len(ocr_text)} chars")
            print(f"OCR text (first 300 chars):")
            print(f"'{ocr_text[:300]}...'")
            
            # Analyze for specific patterns that should trigger boundaries
            print(f"\n🔍 Pattern Analysis:")
            patterns_found = []
            
            # Email patterns
            if any(marker in ocr_text.lower()[:400] for marker in ['from:', 'to:', 'subject:', '@']):
                patterns_found.append("EMAIL patterns")
            
            # Document type patterns
            if any(word in ocr_text.lower() for word in ['submittal', 'transmittal']):
                patterns_found.append("SUBMITTAL patterns")
            
            if any(word in ocr_text.lower() for word in ['schedule', 'values']):
                patterns_found.append("SCHEDULE OF VALUES patterns")
            
            if any(word in ocr_text.lower() for word in ['application', 'payment']):
                patterns_found.append("PAYMENT APPLICATION patterns")
            
            if any(word in ocr_text.lower() for word in ['invoice', 'packing', 'sales order']):
                patterns_found.append("INVOICE/SHIPPING patterns")
            
            if any(word in ocr_text.lower() for word in ['request', 'information', 'rfi']):
                patterns_found.append("RFI patterns")
            
            if any(word in ocr_text.lower() for word in ['cost', 'proposal']):
                patterns_found.append("COST PROPOSAL patterns")
            
            if patterns_found:
                print(f"   Found: {', '.join(patterns_found)}")
            else:
                print("   ❌ No boundary patterns detected")
                
        except Exception as e:
            print(f"❌ OCR failed: {e}")
    
    pdf_doc.close()
    print(f"\n✅ Analysis complete")

if __name__ == "__main__":
    main()