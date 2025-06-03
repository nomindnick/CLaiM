#!/usr/bin/env python3
"""
Test script to evaluate PDF splitting and classification accuracy against ground truth.
Processes both test PDFs and compares results to the expected output.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.ai_classifier.llm_classifier import LLMDocumentClassifier
from modules.ai_classifier.models import ClassificationRequest
from modules.document_processor.models import Document as ProcessedDocument, PDFProcessingRequest

# Test files
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF_1 = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"  # Non-searchable
TEST_PDF_2 = PROJECT_ROOT / "tests" / "Test_PDF_Set_2.pdf"  # Searchable
GROUND_TRUTH = PROJECT_ROOT / "tests" / "Test_PDF_Set_Ground_Truth.json"

def load_ground_truth() -> List[Dict]:
    """Load ground truth data from JSON file."""
    with open(GROUND_TRUTH, 'r') as f:
        data = json.load(f)
    return data['documents']

def parse_page_range(page_range: str) -> Tuple[int, int]:
    """Convert page range string (e.g., '1-4') to start and end integers."""
    if '-' in page_range:
        start, end = page_range.split('-')
        return int(start), int(end)
    else:
        page = int(page_range)
        return page, page

def process_pdf(pdf_path: Path) -> List[ProcessedDocument]:
    """Process a PDF through the full pipeline."""
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*80}")
    
    # Initialize processors
    # Disable visual detection to test improved pattern detector
    splitter = PDFSplitter(use_visual_detection=False)
    classifier = LLMDocumentClassifier()
    
    # Split PDF into documents
    print("\n1. Splitting PDF into documents...")
    start_time = time.time()
    
    try:
        # Create processing request
        request = PDFProcessingRequest(
            file_path=pdf_path,
            perform_ocr=True,  # Enable OCR for non-searchable PDFs
            privacy_mode="local"
        )
        
        # Process PDF
        result = splitter.process_pdf(request)
        
        if not result.success:
            raise Exception(f"PDF processing failed: {result.error}")
            
        documents = result.documents
        split_time = time.time() - start_time
        print(f"   - Split complete in {split_time:.2f}s")
        print(f"   - Found {len(documents)} documents")
        
        # Classify each document
        print("\n2. Classifying documents...")
        for i, doc in enumerate(documents):
            start_page, end_page = doc.page_range
            print(f"   - Classifying document {i+1}/{len(documents)} (pages {start_page}-{end_page})...")
            # Create classification request
            request = ClassificationRequest(
                text=doc.text,
                title=f"Document {i+1} (pages {start_page}-{end_page})"
            )
            classification_result = classifier.classify(request)
            # Update document with classification
            doc.type = classification_result.document_type
            doc.classification = classification_result
            if hasattr(doc, 'classification') and doc.classification:
                print(f"     Type: {doc.classification.document_type.value} (confidence: {doc.classification.confidence:.2f})")
            else:
                print(f"     Type: UNKNOWN")
        
        classification_time = time.time() - start_time - split_time
        print(f"   - Classification complete in {classification_time:.2f}s")
        print(f"   - Total processing time: {time.time() - start_time:.2f}s")
        
        return documents
        
    except Exception as e:
        print(f"ERROR: Failed to process PDF: {e}")
        import traceback
        traceback.print_exc()
        return []

def compare_results(documents: List[ProcessedDocument], ground_truth: List[Dict], pdf_name: str):
    """Compare processing results against ground truth."""
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON: {pdf_name}")
    print(f"{'='*80}")
    
    # Boundary detection accuracy
    print(f"\n1. BOUNDARY DETECTION:")
    print(f"   Expected: {len(ground_truth)} documents")
    print(f"   Found: {len(documents)} documents")
    
    if len(documents) == len(ground_truth):
        print("   ✓ Correct number of documents detected")
    else:
        print(f"   ✗ Incorrect number of documents (off by {abs(len(documents) - len(ground_truth))})")
    
    # Detailed boundary comparison
    print("\n   Boundary Details:")
    for i, gt in enumerate(ground_truth):
        gt_start, gt_end = parse_page_range(gt['pages'])
        print(f"\n   Document {i+1}:")
        print(f"   - Expected: pages {gt_start}-{gt_end} ({gt['type']})")
        
        # Find matching document by page overlap
        matching_doc = None
        best_overlap = 0
        for doc in documents:
            doc_start, doc_end = doc.page_range
            overlap_start = max(doc_start, gt_start)
            overlap_end = min(doc_end, gt_end)
            overlap = max(0, overlap_end - overlap_start + 1)
            if overlap > best_overlap:
                best_overlap = overlap
                matching_doc = doc
        
        if matching_doc and best_overlap > 0:
            doc_start, doc_end = matching_doc.page_range
            print(f"   - Found: pages {doc_start}-{doc_end}")
            if doc_start == gt_start and doc_end == gt_end:
                print("     ✓ Boundaries match exactly")
            else:
                print(f"     ✗ Boundaries don't match (overlap: {best_overlap} pages)")
        else:
            print("   - Found: NO MATCHING DOCUMENT")
            print("     ✗ Document not detected")
    
    # Classification accuracy
    print(f"\n2. CLASSIFICATION ACCURACY:")
    correct_classifications = 0
    total_classifications = 0
    
    for i, gt in enumerate(ground_truth):
        gt_start, gt_end = parse_page_range(gt['pages'])
        gt_type = gt['type']
        
        # Find matching document
        matching_doc = None
        for doc in documents:
            doc_start, doc_end = doc.page_range
            if doc_start == gt_start and doc_end == gt_end:
                matching_doc = doc
                break
        
        if matching_doc and hasattr(matching_doc, 'classification') and matching_doc.classification:
            total_classifications += 1
            predicted_type = matching_doc.classification.document_type.value
            
            # Normalize types for comparison
            gt_type_normalized = gt_type.lower().replace(' ', '_')
            predicted_type_normalized = predicted_type.lower().replace(' ', '_')
            
            if gt_type_normalized == predicted_type_normalized or gt_type in predicted_type or predicted_type in gt_type:
                correct_classifications += 1
                print(f"   ✓ Document {i+1}: Correctly classified as '{gt_type}'")
            else:
                print(f"   ✗ Document {i+1}: Expected '{gt_type}', got '{predicted_type}'")
        else:
            if matching_doc:
                print(f"   ✗ Document {i+1}: Found but not classified (expected '{gt_type}')")
            else:
                print(f"   ✗ Document {i+1}: Not found (expected '{gt_type}')")
    
    if total_classifications > 0:
        accuracy = (correct_classifications / len(ground_truth)) * 100
        print(f"\n   Classification accuracy: {correct_classifications}/{len(ground_truth)} ({accuracy:.1f}%)")
    else:
        print(f"\n   Classification accuracy: 0/{len(ground_truth)} (0.0%)")
    
    # Summary
    print(f"\n3. SUMMARY:")
    boundary_accuracy = min(len(documents), len(ground_truth)) / len(ground_truth) * 100
    print(f"   - Boundary detection: {boundary_accuracy:.1f}% accuracy")
    print(f"   - Classification: {correct_classifications}/{len(ground_truth)} correct")
    
    return {
        'documents_found': len(documents),
        'documents_expected': len(ground_truth),
        'boundary_accuracy': boundary_accuracy,
        'classification_accuracy': accuracy if total_classifications > 0 else 0,
        'correct_classifications': correct_classifications
    }

def main():
    """Main test function."""
    print("PDF Pipeline Accuracy Test")
    print("="*80)
    
    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"Loaded ground truth: {len(ground_truth)} documents expected")
    
    results = {}
    
    # Test PDF Set 1 (non-searchable)
    if TEST_PDF_1.exists():
        print(f"\nTesting {TEST_PDF_1.name} (non-searchable)...")
        documents_1 = process_pdf(TEST_PDF_1)
        results['Set_1'] = compare_results(documents_1, ground_truth, TEST_PDF_1.name)
    else:
        print(f"ERROR: {TEST_PDF_1} not found")
    
    # Test PDF Set 2 (searchable)
    if TEST_PDF_2.exists():
        print(f"\n\nTesting {TEST_PDF_2.name} (searchable)...")
        documents_2 = process_pdf(TEST_PDF_2)
        results['Set_2'] = compare_results(documents_2, ground_truth, TEST_PDF_2.name)
    else:
        print(f"ERROR: {TEST_PDF_2} not found")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for pdf_set, metrics in results.items():
        print(f"\n{pdf_set}:")
        print(f"  - Documents: {metrics['documents_found']}/{metrics['documents_expected']}")
        print(f"  - Boundary accuracy: {metrics['boundary_accuracy']:.1f}%")
        print(f"  - Classification accuracy: {metrics['classification_accuracy']:.1f}%")
        print(f"  - Correct classifications: {metrics['correct_classifications']}/{metrics['documents_expected']}")
    
    # Check success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}")
    
    all_success = True
    for pdf_set, metrics in results.items():
        print(f"\n{pdf_set}:")
        
        # Check if exactly 14 documents
        if metrics['documents_found'] == 14:
            print("  ✓ Split into exactly 14 documents")
        else:
            print(f"  ✗ Should split into 14 documents (found {metrics['documents_found']})")
            all_success = False
        
        # Check boundary accuracy
        if metrics['boundary_accuracy'] == 100:
            print("  ✓ All boundaries correctly detected")
        else:
            print(f"  ✗ Boundary detection needs improvement ({metrics['boundary_accuracy']:.1f}%)")
            all_success = False
        
        # Check classification accuracy
        if metrics['classification_accuracy'] >= 85:
            print(f"  ✓ Classification accuracy meets target ({metrics['classification_accuracy']:.1f}%)")
        else:
            print(f"  ✗ Classification accuracy below target ({metrics['classification_accuracy']:.1f}% < 85%)")
            all_success = False
    
    if all_success:
        print("\n✓ ALL SUCCESS CRITERIA MET!")
    else:
        print("\n✗ Success criteria not met - improvements needed")
    
    return results

if __name__ == "__main__":
    main()