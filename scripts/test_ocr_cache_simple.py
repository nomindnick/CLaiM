#!/usr/bin/env python3
"""
Simple OCR cache test - just verify cache hit/miss behavior.
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
import fitz


def test_simple_caching():
    """Test basic OCR caching functionality."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🚀 Simple OCR Cache Test")
    print("=" * 30)
    
    # Create OCR handler with cache
    ocr_handler = ImprovedOCRHandler(cache_dir=".ocr_cache_simple_test")
    ocr_handler.clear_cache()
    
    # Open PDF and get first page
    pdf_doc = fitz.open(test_pdf_path)
    first_page = pdf_doc[0]
    
    print(f"\n📊 Initial cache stats:")
    stats = ocr_handler.get_cache_stats()
    print(f"   Cache items: {stats.get('cache_size_items', 0)}")
    
    # Test 1: First OCR call (should miss cache)
    print(f"\n⚡ First OCR call (cache miss expected)...")
    start_time = time.time()
    text1, conf1 = ocr_handler.process_page(first_page)
    first_time = time.time() - start_time
    
    print(f"✅ First call: {len(text1)} chars, confidence: {conf1:.3f}, time: {first_time:.2f}s")
    
    # Check cache after first call
    stats = ocr_handler.get_cache_stats()
    print(f"📊 Cache after first call: {stats.get('cache_size_items', 0)} items")
    
    # Test 2: Second OCR call (should hit cache)
    print(f"\n⚡ Second OCR call (cache hit expected)...")
    start_time = time.time()
    text2, conf2 = ocr_handler.process_page(first_page)
    second_time = time.time() - start_time
    
    print(f"✅ Second call: {len(text2)} chars, confidence: {conf2:.3f}, time: {second_time:.2f}s")
    
    # Verify results are identical
    if text1 == text2 and conf1 == conf2:
        print("✅ Results identical - caching working correctly!")
    else:
        print("❌ Results differ - caching issue!")
        return
    
    # Performance analysis
    if second_time < first_time:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"🎉 Performance improvement: {improvement:.1f}% faster")
        
        if improvement > 50:
            print("🎯 EXCELLENT: Major caching benefit!")
        elif improvement > 20:
            print("👍 GOOD: Significant caching benefit")
        else:
            print("⚠️  MINIMAL: Small benefit")
    else:
        print("❌ No performance improvement")
    
    # Test 3: Different page (should miss cache)
    print(f"\n⚡ Different page test (cache miss expected)...")
    second_page = pdf_doc[1]
    start_time = time.time()
    text3, conf3 = ocr_handler.process_page(second_page)
    third_time = time.time() - start_time
    
    print(f"✅ Different page: {len(text3)} chars, confidence: {conf3:.3f}, time: {third_time:.2f}s")
    
    # Final cache stats
    stats = ocr_handler.get_cache_stats()
    print(f"\n📊 Final cache stats:")
    print(f"   Cache items: {stats.get('cache_size_items', 0)}")
    print(f"   Cache size: {stats.get('cache_size_bytes', 0)} bytes")
    
    pdf_doc.close()
    
    print(f"\n🎯 SUMMARY")
    print(f"   Cache working: {'✅' if text1 == text2 else '❌'}")
    print(f"   Performance: {improvement:.1f}% improvement")
    print(f"   Cache items: {stats.get('cache_size_items', 0)}")


if __name__ == "__main__":
    test_simple_caching()