#!/usr/bin/env python3
"""Test end-to-end classification with PDF processing pipeline."""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest


def test_pdf_classification():
    """Test classification during PDF processing."""
    print("=== Testing End-to-End PDF Classification ===\n")
    
    # Find a test PDF
    test_data_dir = Path(__file__).parent.parent / "tests" / "test_data"
    test_pdfs = list(test_data_dir.glob("*.pdf"))
    
    if not test_pdfs:
        print("No test PDFs found. Skipping end-to-end test.")
        return False
    
    test_pdf = test_pdfs[0]  # Use first available PDF
    print(f"Testing with PDF: {test_pdf.name}")
    
    # Create PDF splitter
    splitter = PDFSplitter()
    
    # Create processing request with classification enabled
    request = PDFProcessingRequest(
        file_path=test_pdf,
        split_documents=True,
        perform_ocr=True,
        extract_metadata=False,  # Skip metadata for this test
        classify_documents=True,  # Enable classification
        ocr_language="eng",
        min_confidence=0.5,
        max_pages=3  # Limit to first 3 documents for quick test
    )
    
    try:
        # Process PDF
        print("Processing PDF with classification enabled...")
        result = splitter.process_pdf(request)
        
        if not result.success:
            print(f"PDF processing failed: {result.errors}")
            return False
        
        print(f"\nProcessing completed in {result.processing_time:.2f}s")
        print(f"Found {result.documents_found} documents")
        
        # Show classification results
        print("\n=== Classification Results ===")
        for i, doc in enumerate(result.documents):
            print(f"\nDocument {i+1} (pages {doc.page_range[0]}-{doc.page_range[1]}):")
            print(f"  Type: {doc.type.value}")
            print(f"  Confidence: {doc.classification_confidence:.3f}")
            print(f"  Text length: {len(doc.text)} characters")
            print(f"  Sample text: {doc.text[:100]}..." if len(doc.text) > 100 else f"  Text: {doc.text}")
        
        # Calculate accuracy if we know expected types
        classified_docs = [doc for doc in result.documents if doc.type.value != "unknown"]
        classification_rate = len(classified_docs) / len(result.documents)
        
        print(f"\n=== Summary ===")
        print(f"Classification rate: {classification_rate:.1%} ({len(classified_docs)}/{len(result.documents)})")
        print(f"Average confidence: {sum(doc.classification_confidence for doc in result.documents) / len(result.documents):.3f}")
        
        # Success if at least half the documents were classified
        success = classification_rate >= 0.5
        print(f"Test result: {'✓ PASS' if success else '✗ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run end-to-end classification test."""
    print("End-to-End Classification Test")
    print("=" * 50)
    
    success = test_pdf_classification()
    
    if success:
        print("\n🎉 End-to-end classification test passed!")
        print("The AI classifier is successfully integrated into the PDF processing pipeline.")
        return 0
    else:
        print("\n❌ End-to-end classification test failed.")
        print("Check the integration between PDF splitter and AI classifier.")
        return 1


if __name__ == "__main__":
    exit(main())