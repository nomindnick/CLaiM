#!/usr/bin/env python3
"""
Comprehensive OCR testing framework.

This script tests all OCR improvements and validates the complete pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
import json
from pathlib import Path
from typing import Dict, List, Any

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest
from modules.document_processor.ocr_handler import OCRHandler
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
from modules.document_processor.hybrid_text_extractor import HybridTextExtractor

def test_complete_pipeline():
    """Test the complete document processing pipeline with improved OCR."""
    test_files = [
        "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Daily_Report_20250504.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Invoice_0005.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/RFI_123.pdf",
    ]
    
    results = []
    
    print(f"{'='*80}")
    print("COMPREHENSIVE OCR PIPELINE TEST")
    print(f"{'='*80}")
    
    # Test both old and new PDF splitters
    splitters = [
        ("Original PDF Splitter", PDFSplitter(use_visual_detection=True, use_hybrid_text_extraction=False)),
        ("Improved PDF Splitter", PDFSplitter(use_visual_detection=True, use_hybrid_text_extraction=True)),
    ]
    
    for pdf_path in test_files:
        if not Path(pdf_path).exists():
            print(f"❌ File not found: {pdf_path}")
            continue
            
        pdf_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"TESTING: {pdf_name}")
        print(f"{'='*60}")
        
        file_results = {"filename": pdf_name, "tests": {}}
        
        for splitter_name, splitter in splitters:
            print(f"\n--- {splitter_name} ---")
            
            # Create processing request
            request = PDFProcessingRequest(
                file_path=Path(pdf_path),
                split_documents=True,
                perform_ocr=True,
                classify_documents=False,
                extract_metadata=False,
                ocr_language="eng",
                min_confidence=0.6
            )
            
            start_time = time.time()
            try:
                result = splitter.process_pdf(request)
                processing_time = time.time() - start_time
                
                # Calculate metrics
                total_chars = sum(len(doc.text) for doc in result.documents)
                total_pages = sum(len(doc.pages) for doc in result.documents)
                avg_confidence = result.average_confidence
                
                print(f"✅ Success: {len(result.documents)} documents, {total_pages} pages")
                print(f"   Total text: {total_chars} characters")
                print(f"   Avg confidence: {avg_confidence:.3f}")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Speed: {total_pages/processing_time:.1f} pages/sec")
                
                # Store results
                file_results["tests"][splitter_name] = {
                    "success": True,
                    "documents": len(result.documents),
                    "pages": total_pages,
                    "characters": total_chars,
                    "confidence": avg_confidence,
                    "time": processing_time,
                    "pages_per_sec": total_pages/processing_time,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ Failed: {e}")
                
                file_results["tests"][splitter_name] = {
                    "success": False,
                    "error": str(e),
                    "time": processing_time
                }
        
        results.append(file_results)
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    for file_result in results:
        print(f"\n{file_result['filename']}:")
        
        for test_name, test_result in file_result["tests"].items():
            if test_result["success"]:
                print(f"  {test_name}:")
                print(f"    ✅ {test_result['documents']} docs, {test_result['pages']} pages")
                print(f"    📊 {test_result['characters']} chars, {test_result['confidence']:.3f} confidence")
                print(f"    ⚡ {test_result['time']:.2f}s ({test_result['pages_per_sec']:.1f} pages/sec)")
            else:
                print(f"  {test_name}: ❌ {test_result['error']}")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    original_times = []
    improved_times = []
    original_confidence = []
    improved_confidence = []
    
    for file_result in results:
        tests = file_result["tests"]
        if "Original PDF Splitter" in tests and tests["Original PDF Splitter"]["success"]:
            original_times.append(tests["Original PDF Splitter"]["time"])
            original_confidence.append(tests["Original PDF Splitter"]["confidence"])
        
        if "Improved PDF Splitter" in tests and tests["Improved PDF Splitter"]["success"]:
            improved_times.append(tests["Improved PDF Splitter"]["time"])
            improved_confidence.append(tests["Improved PDF Splitter"]["confidence"])
    
    if original_times and improved_times:
        avg_original_time = sum(original_times) / len(original_times)
        avg_improved_time = sum(improved_times) / len(improved_times)
        avg_original_conf = sum(original_confidence) / len(original_confidence)
        avg_improved_conf = sum(improved_confidence) / len(improved_confidence)
        
        speedup = avg_original_time / avg_improved_time if avg_improved_time > 0 else 0
        conf_improvement = avg_improved_conf - avg_original_conf
        
        print(f"Average processing time:")
        print(f"  Original: {avg_original_time:.2f}s")
        print(f"  Improved: {avg_improved_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        print(f"\nAverage confidence:")
        print(f"  Original: {avg_original_conf:.3f}")
        print(f"  Improved: {avg_improved_conf:.3f}")
        print(f"  Improvement: {conf_improvement:+.3f}")
        
        if speedup > 1.1:
            print(f"\n🚀 Improved pipeline is {speedup:.1f}x faster!")
        if conf_improvement > 0.1:
            print(f"📈 Improved pipeline has {conf_improvement:.3f} better confidence!")
    
    return results

def test_ocr_accuracy_comparison():
    """Test OCR accuracy comparison between different handlers."""
    pdf_path = "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Test file not found: {pdf_path}")
        return
    
    print(f"\n{'='*60}")
    print("OCR ACCURACY COMPARISON")
    print(f"{'='*60}")
    
    import fitz
    pdf_doc = fitz.open(pdf_path)
    
    # Test different OCR handlers
    handlers = [
        ("Original OCR", OCRHandler(min_confidence=0.3)),
        ("Improved OCR", ImprovedOCRHandler(min_confidence=0.3)),
        ("Hybrid Extractor", HybridTextExtractor(min_confidence=0.3))
    ]
    
    for page_num in range(min(2, pdf_doc.page_count)):
        page = pdf_doc[page_num]
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 40)
        
        # PyMuPDF baseline
        pymupdf_text = page.get_text()
        print(f"PyMuPDF: {len(pymupdf_text)} chars")
        
        for handler_name, handler in handlers:
            try:
                start_time = time.time()
                
                if isinstance(handler, HybridTextExtractor):
                    text, confidence, method = handler.extract_text(page)
                    method_info = f" ({method.value})"
                else:
                    text, confidence = handler.process_page(page)
                    method_info = ""
                
                processing_time = time.time() - start_time
                
                print(f"{handler_name}: {len(text)} chars, {confidence:.3f} conf, {processing_time:.2f}s{method_info}")
                
                # Show text quality
                if len(text) > 50:
                    preview = text[:50].replace('\n', ' ')
                    print(f"  Preview: {repr(preview)}")
                
            except Exception as e:
                print(f"{handler_name}: Failed - {e}")
    
    pdf_doc.close()

def save_test_results(results: List[Dict[str, Any]], output_file: str = "ocr_test_results.json"):
    """Save test results to JSON file."""
    output_path = Path(output_file)
    
    # Add metadata
    test_metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "comprehensive_ocr_test",
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n📄 Test results saved to: {output_path}")

if __name__ == "__main__":
    print("🔬 Starting Comprehensive OCR Testing Framework")
    print("=" * 80)
    
    # Test 1: Complete pipeline comparison
    results = test_complete_pipeline()
    
    # Test 2: OCR accuracy comparison
    test_ocr_accuracy_comparison()
    
    # Save results
    save_test_results(results)
    
    print(f"\n✅ All tests completed!")
    print("Check the results above for performance improvements and accuracy gains.")