#!/usr/bin/env python3
"""
Test the hybrid text extractor.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import fitz
import time
from pathlib import Path

from modules.document_processor.hybrid_text_extractor import HybridTextExtractor, TextExtractionMethod

def test_hybrid_extractor(pdf_path: str):
    """Test hybrid text extractor on a PDF."""
    pdf_doc = fitz.open(pdf_path)
    
    print(f"Testing hybrid extractor on: {Path(pdf_path).name}")
    print(f"Total pages: {pdf_doc.page_count}")
    
    # Initialize hybrid extractor
    extractor = HybridTextExtractor(min_confidence=0.6)
    
    total_time = 0
    
    for page_num in range(min(3, pdf_doc.page_count)):  # Test first 3 pages
        page = pdf_doc[page_num]
        
        print(f"\n{'='*60}")
        print(f"PAGE {page_num + 1}")
        print(f"{'='*60}")
        
        # Test hybrid extraction
        start_time = time.time()
        text, confidence, method = extractor.extract_text(page)
        extraction_time = time.time() - start_time
        total_time += extraction_time
        
        # Display results
        print(f"Method used: {method.value}")
        print(f"Text length: {len(text)} characters")
        print(f"Confidence: {confidence:.3f}")
        print(f"Processing time: {extraction_time:.2f}s")
        print(f"Text preview: {repr(text[:150])}")
        
        # Quality assessment
        if confidence >= 0.9:
            quality = "Excellent"
        elif confidence >= 0.7:
            quality = "Good" 
        elif confidence >= 0.5:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"Quality assessment: {quality}")
        
        # Show extraction efficiency
        if method == TextExtractionMethod.PYMUPDF:
            print("✅ Used fast PyMuPDF extraction (optimal)")
        elif method == TextExtractionMethod.HYBRID:
            print("🔄 Used hybrid approach (combined sources)")
        else:
            print(f"🔧 Used {method.value} (OCR fallback)")
    
    avg_time = total_time / min(3, pdf_doc.page_count)
    print(f"\nAverage processing time per page: {avg_time:.2f}s")
    
    pdf_doc.close()

def test_all_documents():
    """Test hybrid extractor on all available test documents."""
    test_files = [
        "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Daily_Report_20250504.pdf", 
        "/home/nick/Projects/CLaiM/tests/test_data/Invoice_0005.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/RFI_123.pdf",
    ]
    
    results = []
    
    for pdf_path in test_files:
        if Path(pdf_path).exists():
            print(f"\n{'='*80}")
            print(f"TESTING: {Path(pdf_path).name}")
            print(f"{'='*80}")
            
            try:
                test_hybrid_extractor(pdf_path)
                results.append((Path(pdf_path).name, "Success"))
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append((Path(pdf_path).name, f"Failed: {e}"))
        else:
            print(f"❌ File not found: {pdf_path}")
            results.append((Path(pdf_path).name, "File not found"))
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for filename, status in results:
        if status == "Success":
            print(f"✅ {filename}: {status}")
        else:
            print(f"❌ {filename}: {status}")

def benchmark_extraction_methods():
    """Benchmark different extraction methods."""
    pdf_path = "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Benchmark file not found: {pdf_path}")
        return
    
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]  # Test first page
    
    print(f"\n{'='*60}")
    print("EXTRACTION METHOD BENCHMARK")
    print(f"{'='*60}")
    
    # Test each method
    extractor = HybridTextExtractor()
    
    # Method 1: Pure PyMuPDF
    start_time = time.time()
    pymupdf_text = extractor._extract_with_pymupdf(page)
    pymupdf_time = time.time() - start_time
    
    print(f"PyMuPDF: {len(pymupdf_text)} chars in {pymupdf_time:.3f}s")
    
    # Method 2: Hybrid (intelligent choice)
    start_time = time.time()
    hybrid_text, hybrid_conf, hybrid_method = extractor.extract_text(page)
    hybrid_time = time.time() - start_time
    
    print(f"Hybrid: {len(hybrid_text)} chars in {hybrid_time:.3f}s (used {hybrid_method.value})")
    print(f"Hybrid confidence: {hybrid_conf:.3f}")
    
    # Speed comparison
    if hybrid_time > 0:
        speedup = pymupdf_time / hybrid_time
        print(f"Speed comparison: Hybrid is {speedup:.1f}x vs pure PyMuPDF")
    
    pdf_doc.close()

if __name__ == "__main__":
    print("Testing Hybrid Text Extractor")
    print("=" * 50)
    
    # Run benchmark first
    benchmark_extraction_methods()
    
    # Test all documents
    test_all_documents()