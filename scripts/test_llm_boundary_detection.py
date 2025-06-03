#!/usr/bin/env python3
"""Test LLM-based boundary detection approach."""

import os
import sys
import json
from pathlib import Path
import time

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_path)

import fitz
from modules.document_processor.llm_boundary_detector import LLMBoundaryDetector
from modules.llm_client.ollama_client import OllamaClient

# Test files
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PDF_1 = PROJECT_ROOT / "tests" / "Test_PDF_Set_1.pdf"
GROUND_TRUTH = PROJECT_ROOT / "tests" / "Test_PDF_Set_Ground_Truth.json"

def test_llm_boundary_detection():
    """Test LLM-based boundary detection."""
    print("Testing LLM-Based Boundary Detection")
    print("="*80)
    print("This demonstrates the new architecture that uses LLM for semantic understanding")
    print("rather than pattern matching.\n")
    
    # Load ground truth
    with open(GROUND_TRUTH, 'r') as f:
        ground_truth = json.load(f)['documents']
    
    print(f"Ground truth: {len(ground_truth)} documents expected")
    
    # Open PDF
    pdf_doc = fitz.open(str(TEST_PDF_1))
    print(f"PDF has {pdf_doc.page_count} pages\n")
    
    # Initialize LLM client
    print("Initializing LLM client (Ollama with Llama 3)...")
    try:
        llm_client = OllamaClient(model="llama3:8b-instruct-q4_0")
        
        # Test LLM connection
        test_response = llm_client.complete("Reply with 'OK' if you can read this.")
        if "OK" not in test_response:
            print("Warning: LLM may not be responding correctly")
    except Exception as e:
        print(f"Error: Could not initialize LLM client: {e}")
        print("\nMake sure Ollama is running with: ollama serve")
        print("And that you have the model: ollama pull llama3:8b-instruct-q4_0")
        return
    
    # Initialize LLM boundary detector
    print("Initializing LLM boundary detector...")
    detector = LLMBoundaryDetector(
        llm_client=llm_client,
        window_size=3,  # Analyze 3 pages at a time
        overlap=1,      # 1 page overlap between windows
        confidence_threshold=0.7
    )
    
    # Detect boundaries
    print("\nStarting boundary detection with sliding window analysis...")
    start_time = time.time()
    
    try:
        boundaries = detector.detect_boundaries(pdf_doc)
        
        elapsed_time = time.time() - start_time
        print(f"\nBoundary detection completed in {elapsed_time:.1f} seconds")
        print(f"Found {len(boundaries)} documents")
        
        # Display results
        print("\nDetected boundaries:")
        for i, (start, end) in enumerate(boundaries):
            print(f"Document {i+1}: pages {start+1}-{end+1} ({end-start+1} pages)")
        
        # Compare to ground truth
        print("\nComparison to ground truth:")
        correct = 0
        expected_boundaries = []
        
        for doc in ground_truth:
            pages = doc['pages']
            if '-' in pages:
                start, end = pages.split('-')
                expected_boundaries.append((int(start)-1, int(end)-1))
            else:
                page = int(pages) - 1
                expected_boundaries.append((page, page))
        
        for i, (exp_start, exp_end) in enumerate(expected_boundaries):
            found = False
            for (det_start, det_end) in boundaries:
                if det_start == exp_start and det_end == exp_end:
                    found = True
                    correct += 1
                    break
            status = "✓" if found else "✗"
            doc_type = ground_truth[i].get('type', 'Unknown')
            print(f"{status} Document {i+1} ({doc_type}): expected pages {exp_start+1}-{exp_end+1}")
        
        accuracy = (correct / len(expected_boundaries)) * 100
        print(f"\nAccuracy: {correct}/{len(expected_boundaries)} ({accuracy:.1f}%)")
        
        # Show improvements over pattern-based approach
        print("\n" + "="*80)
        print("Key Advantages of LLM-Based Approach:")
        print("1. Understands document semantics (email threads, multi-page invoices)")
        print("2. Uses sliding windows for context continuity")
        print("3. Provides confidence scores for each boundary")
        print("4. Can explain reasoning for each decision")
        print("5. Handles edge cases through semantic understanding")
        
    except Exception as e:
        print(f"\nError during boundary detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pdf_doc.close()

def demonstrate_architecture():
    """Demonstrate the key concepts of the new architecture."""
    print("\n\n" + "="*80)
    print("NEW ARCHITECTURE DEMONSTRATION")
    print("="*80)
    
    print("\n1. SLIDING WINDOW ANALYSIS:")
    print("   - Analyzes 3 pages at a time with 1 page overlap")
    print("   - Example: Pages [1,2,3], then [3,4,5], then [5,6,7]...")
    print("   - Provides context for multi-page document understanding")
    
    print("\n2. LLM SEMANTIC UNDERSTANDING:")
    print("   - LLM reads actual content, not just pattern matching")
    print("   - Understands 'this is a forwarded email' vs 'new email'")
    print("   - Recognizes 'Page 2 of 3' means continuation")
    
    print("\n3. CONFIDENCE-BASED DECISIONS:")
    print("   - Each boundary has confidence score (0.0-1.0)")
    print("   - Multiple windows vote on same boundary")
    print("   - Only boundaries above threshold are accepted")
    
    print("\n4. DOCUMENT-TYPE AWARENESS:")
    print("   - Different strategies for emails, invoices, drawings")
    print("   - Knows invoices often have multiple pages")
    print("   - Understands email threading patterns")
    
    print("\n5. ITERATIVE REFINEMENT:")
    print("   - Can re-analyze low-confidence boundaries")
    print("   - Validates short documents for potential merging")
    print("   - Uses LLM to decide if fragments should combine")

if __name__ == "__main__":
    # First demonstrate the architecture
    demonstrate_architecture()
    
    # Then run the test
    print("\n\nPress Enter to run the LLM boundary detection test...")
    input()
    
    test_llm_boundary_detection()