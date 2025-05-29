#!/usr/bin/env python3
"""
Test script for OCR functionality.

Usage:
    python scripts/test_ocr.py [pdf_file]
    
If no PDF file is provided, it will use the sample PDF in tests/.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.modules.document_processor.pdf_splitter import PDFSplitter
from backend.modules.document_processor.models import PDFProcessingRequest


def test_ocr(pdf_path: Path):
    """Test OCR on a PDF file."""
    print(f"\n🔍 Testing OCR on: {pdf_path.name}")
    print("=" * 60)
    
    # Create processing request with OCR enabled
    request = PDFProcessingRequest(
        file_path=pdf_path,
        split_documents=False,  # Process as single document
        perform_ocr=True,
        ocr_language="eng",
        min_confidence=0.6,
        classify_documents=False,
        extract_metadata=False,
    )
    
    # Process PDF
    splitter = PDFSplitter()
    result = splitter.process_pdf(request)
    
    if result.success:
        print(f"✅ Processing successful!")
        print(f"   Total pages: {result.total_pages}")
        print(f"   OCR pages: {result.ocr_pages}")
        print(f"   Average confidence: {result.average_confidence:.2f}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        # Show OCR results for each page
        if result.documents:
            doc = result.documents[0]
            print(f"\n📄 Document analysis:")
            print(f"   Has OCR content: {doc.has_ocr_content}")
            
            for page in doc.pages:
                if page.is_scanned:
                    print(f"\n   Page {page.page_number} (OCR):")
                    print(f"   - Confidence: {page.confidence:.2f}")
                    print(f"   - Text preview: {page.text[:200]}...")
                else:
                    print(f"\n   Page {page.page_number} (Text):")
                    print(f"   - Native text extracted")
    else:
        print(f"❌ Processing failed!")
        for error in result.errors:
            print(f"   Error: {error}")
    
    return result


def main():
    """Main test function."""
    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        # Use sample PDF
        pdf_path = Path(__file__).parent.parent / "tests" / "Sample PDFs.pdf"
    
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Check if Tesseract is installed
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print(f"✅ Tesseract is installed")
    except Exception as e:
        print(f"❌ Tesseract is not installed!")
        print(f"   Please install Tesseract OCR:")
        print(f"   - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print(f"   - macOS: brew install tesseract")
        print(f"   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    
    # Run test
    test_ocr(pdf_path)


if __name__ == "__main__":
    main()