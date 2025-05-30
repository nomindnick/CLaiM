#!/usr/bin/env python3
"""
Test the improved OCR handler against the original.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import fitz
import time
from pathlib import Path

from modules.document_processor.ocr_handler import OCRHandler
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler

def compare_ocr_handlers(pdf_path: str):
    """Compare original vs improved OCR handler."""
    pdf_doc = fitz.open(pdf_path)
    
    print(f"Comparing OCR handlers on: {Path(pdf_path).name}")
    print(f"Total pages: {pdf_doc.page_count}")
    
    # Initialize both handlers
    original_handler = OCRHandler(min_confidence=0.3)
    improved_handler = ImprovedOCRHandler(min_confidence=0.3)
    
    for page_num in range(min(3, pdf_doc.page_count)):  # Test first 3 pages
        page = pdf_doc[page_num]
        
        print(f"\n{'='*60}")
        print(f"PAGE {page_num + 1}")
        print(f"{'='*60}")
        
        # First, check PyMuPDF text extraction
        pymupdf_text = page.get_text()
        print(f"PyMuPDF text: {len(pymupdf_text)} chars")
        print(f"Preview: {repr(pymupdf_text[:100])}")
        
        # Test original handler
        print(f"\n--- Original OCR Handler ---")
        start_time = time.time()
        try:
            orig_text, orig_conf = original_handler.process_page(page)
            orig_time = time.time() - start_time
            print(f"Original: {len(orig_text)} chars, confidence: {orig_conf:.3f}, time: {orig_time:.2f}s")
            print(f"Preview: {repr(orig_text[:100])}")
        except Exception as e:
            print(f"Original handler failed: {e}")
            orig_text, orig_conf, orig_time = "", 0.0, 0.0
        
        # Test improved handler
        print(f"\n--- Improved OCR Handler ---")
        start_time = time.time()
        try:
            imp_text, imp_conf = improved_handler.process_page(page)
            imp_time = time.time() - start_time
            print(f"Improved: {len(imp_text)} chars, confidence: {imp_conf:.3f}, time: {imp_time:.2f}s")
            print(f"Preview: {repr(imp_text[:100])}")
        except Exception as e:
            print(f"Improved handler failed: {e}")
            imp_text, imp_conf, imp_time = "", 0.0, 0.0
        
        # Compare results
        print(f"\n--- Comparison ---")
        if len(pymupdf_text.strip()) > 10:
            print(f"Note: PyMuPDF already extracted {len(pymupdf_text)} chars - OCR may not be needed")
        
        print(f"Confidence improvement: {orig_conf:.3f} -> {imp_conf:.3f} ({imp_conf - orig_conf:+.3f})")
        print(f"Text length: {len(orig_text)} -> {len(imp_text)} ({len(imp_text) - len(orig_text):+d})")
        print(f"Processing time: {orig_time:.2f}s -> {imp_time:.2f}s ({imp_time - orig_time:+.2f}s)")
        
        # Quality assessment
        if imp_conf > orig_conf + 0.1:
            print("✅ Improved handler is significantly better")
        elif imp_conf > orig_conf:
            print("✅ Improved handler is slightly better")
        elif abs(imp_conf - orig_conf) < 0.05:
            print("⚖️  Both handlers perform similarly")
        else:
            print("❌ Original handler performed better")
    
    pdf_doc.close()

def test_edge_cases():
    """Test edge cases and different document types."""
    test_files = [
        "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Daily_Report_20250504.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Invoice_0005.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/RFI_123.pdf",
    ]
    
    for pdf_path in test_files:
        if Path(pdf_path).exists():
            print(f"\n{'='*80}")
            print(f"TESTING: {Path(pdf_path).name}")
            print(f"{'='*80}")
            compare_ocr_handlers(pdf_path)
        else:
            print(f"File not found: {pdf_path}")

if __name__ == "__main__":
    test_edge_cases()