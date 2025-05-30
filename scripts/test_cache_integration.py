#!/usr/bin/env python3
"""
Test OCR cache integration between boundary detection and text extraction.
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.boundary_detector import BoundaryDetector
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
from modules.document_processor.hybrid_text_extractor import HybridTextExtractor
import fitz


def test_cache_integration():
    """Test that boundary detection and text extraction share OCR cache."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🚀 OCR Cache Integration Test")
    print("=" * 40)
    
    # Create shared OCR handler with cache
    ocr_handler = ImprovedOCRHandler(cache_dir=".ocr_cache_integration_test")
    ocr_handler.clear_cache()
    
    # Open PDF
    pdf_doc = fitz.open(test_pdf_path)
    first_page = pdf_doc[0]
    
    print(f"\n📊 Starting with clean cache: {ocr_handler.get_cache_stats().get('cache_size_items', 0)} items")
    
    # Test 1: Boundary detection (should populate cache)
    print(f"\n⚡ Step 1: Boundary detection...")
    boundary_detector = BoundaryDetector(ocr_handler=ocr_handler)
    
    start_time = time.time()
    # Just process first few pages for testing
    boundaries = boundary_detector.detect_boundaries(pdf_doc)
    boundary_time = time.time() - start_time
    
    stats_after_boundary = ocr_handler.get_cache_stats()
    print(f"✅ Boundary detection: {len(boundaries)} boundaries in {boundary_time:.2f}s")
    print(f"📊 Cache after boundary detection: {stats_after_boundary.get('cache_size_items', 0)} items")
    
    # Test 2: Text extraction (should hit cache)
    print(f"\n⚡ Step 2: Text extraction (should hit cache)...")
    text_extractor = HybridTextExtractor(ocr_handler=ocr_handler)
    
    start_time = time.time()
    text, confidence, method = text_extractor.extract_text(first_page)
    extraction_time = time.time() - start_time
    
    stats_after_extraction = ocr_handler.get_cache_stats()
    print(f"✅ Text extraction: {len(text)} chars, {confidence:.3f} confidence in {extraction_time:.3f}s")
    print(f"📊 Cache after text extraction: {stats_after_extraction.get('cache_size_items', 0)} items")
    
    # Test 3: Verify cache effectiveness
    print(f"\n⚡ Step 3: Direct OCR call (should hit cache)...")
    start_time = time.time()
    direct_text, direct_conf = ocr_handler.process_page(first_page)
    direct_time = time.time() - start_time
    
    print(f"✅ Direct OCR: {len(direct_text)} chars, {direct_conf:.3f} confidence in {direct_time:.3f}s")
    
    # Analysis
    print(f"\n📈 CACHE INTEGRATION ANALYSIS")
    print("-" * 40)
    
    cache_items = stats_after_extraction.get('cache_size_items', 0)
    if cache_items > 0:
        print(f"✅ Cache populated: {cache_items} items")
        
        if direct_time < 0.1:  # Very fast, indicating cache hit
            print(f"✅ Cache hit detected: {direct_time:.3f}s (very fast)")
            
            if extraction_time < 0.1:  # Text extractor also hit cache
                print(f"🎉 SUCCESS: Text extractor used cache!")
            else:
                print(f"⚠️  Text extractor may not be using cache optimally")
        else:
            print(f"⚠️  Direct OCR not hitting cache: {direct_time:.3f}s")
    else:
        print(f"❌ No cache items found")
    
    # Verify text consistency
    if direct_text == text:
        print(f"✅ Text consistent between extraction methods")
    else:
        print(f"⚠️  Text differs between extraction methods")
    
    pdf_doc.close()
    
    print(f"\n🎯 INTEGRATION SUMMARY")
    print(f"   Boundary detection time: {boundary_time:.2f}s")
    print(f"   Text extraction time: {extraction_time:.3f}s")
    print(f"   Direct OCR time: {direct_time:.3f}s")
    print(f"   Cache items: {cache_items}")
    print(f"   Cache working: {'✅' if direct_time < 0.1 else '❌'}")
    print(f"   Integration: {'🎉 SUCCESS' if cache_items > 0 and direct_time < 0.1 else '⚠️  NEEDS WORK'}")


if __name__ == "__main__":
    test_cache_integration()