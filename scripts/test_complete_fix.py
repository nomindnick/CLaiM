#!/usr/bin/env python3
"""
Test the complete fix: boundary detection + classification.
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
    
    print("🚀 Complete Pipeline Fix Test")
    print("=" * 50)
    print("Testing BOTH boundary detection AND classification fixes...")
    
    # Test with full pipeline including classification
    print("\n📋 Processing with FIXED hybrid detection + classification")
    try:
        pdf_splitter = PDFSplitter(use_visual_detection=True)
        request = PDFProcessingRequest(
            file_path=Path(test_pdf_path),
            user_id="test_user",
            perform_ocr=True,
            split_documents=True,
            classify_documents=True  # Enable classification to test the fix
        )
        result = pdf_splitter.process_pdf(request)
        
        print(f"✅ Processing complete!")
        print(f"📄 Found {len(result.documents)} documents")
        print(f"🔍 Detection level: {getattr(result, 'detection_level', 'unknown')}")
        print(f"⏱️  Processing time: {result.processing_time:.1f}s")
        
        # Show document breakdown
        print(f"\n📋 Document Classification Results:")
        print("-" * 40)
        for i, doc in enumerate(result.documents):
            start, end = doc.page_range
            doc_type = doc.type.value if doc.type else "unknown"
            confidence = getattr(doc, 'classification_confidence', 0.0)
            print(f"{i+1:2d}. Pages {start:2d}-{end:2d} | {doc_type} (conf: {confidence:.2f})")
        
        # Assessment
        print(f"\n📊 ASSESSMENT")
        print("-" * 30)
        
        # Check boundary detection success
        expected_docs = 14  # Roughly expected for this PDF
        if len(result.documents) >= expected_docs * 0.5:  # At least 50% of expected
            print("✅ BOUNDARY DETECTION: Significant improvement!")
            if len(result.documents) >= expected_docs * 0.8:
                print("🎉 EXCELLENT: Found most expected boundaries!")
        else:
            print("❌ BOUNDARY DETECTION: Still needs work")
        
        # Check classification for first few documents (likely emails)
        email_docs = [doc for doc in result.documents[:3] if doc.type.value == 'email']
        if email_docs:
            print("✅ CLASSIFICATION: Successfully classified emails!")
        else:
            print("⚠️  CLASSIFICATION: Check email classification logic")
        
        # Overall assessment
        detection_level = getattr(result, 'detection_level', 'unknown')
        if (len(result.documents) >= 10 and 
            detection_level in ['heuristic', 'visual'] and
            email_docs):
            print("\n🎉 SUCCESS: Both fixes working!")
            print("   - Boundary detection finds multiple documents")
            print("   - Classification correctly identifies emails")
        elif len(result.documents) >= 10:
            print("\n👍 PARTIAL SUCCESS: Boundary detection improved")
            print("   - Check classification logic for emails")
        else:
            print("\n⚠️  NEEDS MORE WORK: Still losing too many boundaries")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()