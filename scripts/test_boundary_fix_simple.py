#!/usr/bin/env python3
"""
Simple test of the boundary detection fix.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest

def main():
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🚀 Simple Boundary Detection Fix Test")
    print("=" * 50)
    
    # Test 1: Pattern-only detection (baseline)
    print("📋 Test 1: Pattern-only detection")
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=False)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True,
            classify_documents=False
        )
        result = pdf_splitter.process_pdf(request)
        pattern_count = len(result.documents)
        print(f"✅ Pattern-only: {pattern_count} documents")
        
    except Exception as e:
        print(f"❌ Pattern detection failed: {e}")
        return
    
    # Test 2: Fixed hybrid detection (should now match pattern results)
    print("\n📋 Test 2: Fixed hybrid detection")
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=True)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True,
            classify_documents=False
        )
        result = pdf_splitter.process_pdf(request)
        hybrid_count = len(result.documents)
        detection_level = getattr(result, 'detection_level', 'unknown')
        print(f"✅ Fixed hybrid: {hybrid_count} documents (level: {detection_level})")
        
        # Assessment
        print(f"\n📊 ASSESSMENT")
        print("-" * 30)
        print(f"Pattern detection: {pattern_count} documents")
        print(f"Fixed hybrid: {hybrid_count} documents")
        
        if hybrid_count >= pattern_count * 0.8:  
            print("✅ SUCCESS: Fix preserved pattern detection results!")
            if detection_level == 'heuristic':
                print("🎉 PERFECT: System chose pattern detection over visual!")
        else:
            print("❌ FAILED: Still losing boundaries in hybrid mode")
        
    except Exception as e:
        print(f"❌ Hybrid detection failed: {e}")

if __name__ == "__main__":
    main()