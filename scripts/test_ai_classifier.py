#!/usr/bin/env python3
"""Test script for AI classifier integration."""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.ai_classifier.classifier import document_classifier
from modules.ai_classifier.models import ClassificationRequest
from modules.document_processor.models import DocumentType


def test_classifier_basic():
    """Test basic classifier functionality."""
    print("=== Testing AI Classifier Basic Functionality ===\n")
    
    test_cases = [
        {
            "name": "Email Document",
            "text": """
            From: john.smith@contractor.com
            To: mary.jones@schooldistrict.edu
            Subject: Project Update - Phase 1 Complete
            Date: May 30, 2025
            
            Dear Mary,
            
            I wanted to update you on the progress of the elementary school renovation project.
            Phase 1 has been completed on schedule.
            
            Best regards,
            John Smith
            Project Manager
            """,
            "expected": DocumentType.EMAIL
        },
        {
            "name": "RFI Document",
            "text": """
            REQUEST FOR INFORMATION
            RFI #125
            
            Project: Elementary School Renovation
            Date: May 30, 2025
            
            Subject: Clarification needed on electrical specifications
            
            Please provide clarification on the following:
            1. What fire rating is required for electrical panels?
            2. Are GFCI outlets required in all classrooms?
            
            Response required by: June 5, 2025
            
            Submitted by: ABC Electrical Contractors
            """,
            "expected": DocumentType.RFI
        },
        {
            "name": "Invoice Document", 
            "text": """
            INVOICE #INV-2025-001
            
            Bill To: 
            Riverside School District
            123 Education Blvd
            Riverside, CA 92501
            
            Invoice Date: May 30, 2025
            Due Date: June 30, 2025
            
            Description: Concrete work - Foundation Phase
            Quantity: 1
            Amount: $45,000.00
            
            Total Amount Due: $45,000.00
            
            Payment Terms: Net 30 days
            Please remit payment to the address above.
            """,
            "expected": DocumentType.INVOICE
        },
        {
            "name": "Change Order Document",
            "text": """
            CHANGE ORDER #007
            
            Project: Elementary School Renovation  
            Original Contract Amount: $500,000.00
            Previous Change Orders: $25,000.00
            This Change Order: $15,000.00
            New Contract Total: $540,000.00
            
            Description of Change:
            Additional electrical outlets required in library due to 
            technology upgrade requirements not in original scope.
            
            Time Impact: 3 additional days
            
            Contractor Signature: _________________ Date: _______
            Owner Signature: _____________________ Date: _______
            """,
            "expected": DocumentType.CHANGE_ORDER
        },
        {
            "name": "Daily Report Document",
            "text": """
            DAILY CONSTRUCTION REPORT
            Date: May 30, 2025
            Project: Elementary School Renovation
            Weather: Sunny, 75°F
            
            Crew Information:
            - Superintendent: Mike Johnson
            - Crew Size: 12 workers
            - Hours Worked: 8:00 AM - 4:30 PM
            
            Work Performed Today:
            - Continued foundation excavation in Building A
            - Installed rebar for footings in Section 1
            - Delivered concrete materials for tomorrow's pour
            
            Equipment On Site:
            - Excavator (CAT 320)
            - Concrete mixer truck
            - Compaction equipment
            
            Issues/Delays: None
            Safety Incidents: None
            """,
            "expected": DocumentType.DAILY_REPORT
        }
    ]
    
    results = []
    total_time = 0
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        # Create classification request
        request = ClassificationRequest(
            text=test_case["text"],
            title=test_case["name"],
            require_reasoning=True,
            min_confidence=0.3
        )
        
        # Classify
        start_time = time.time()
        result = document_classifier.classify(request)
        classification_time = time.time() - start_time
        total_time += classification_time
        
        # Check result
        is_correct = result.document_type == test_case["expected"]
        
        print(f"  Expected: {test_case['expected'].value}")
        print(f"  Predicted: {result.document_type.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print(f"  Time: {classification_time:.3f}s")
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning}")
        print()
        
        results.append({
            "name": test_case["name"],
            "expected": test_case["expected"],
            "predicted": result.document_type,
            "confidence": result.confidence,
            "correct": is_correct,
            "time": classification_time
        })
    
    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_time = total_time / len(results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    print("=== SUMMARY ===")
    print(f"Total tests: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    
    return results


def test_model_status():
    """Test model status and health."""
    print("\n=== Testing Model Status ===")
    
    try:
        status = document_classifier.get_model_status()
        print(f"Model loaded: {status.distilbert_loaded}")
        print(f"Model path: {status.model_path}")
        print(f"Supported classes: {len(status.supported_classes)}")
        print(f"Classes: {', '.join(status.supported_classes)}")
        
        if status.model_size_mb:
            print(f"Model size: {status.model_size_mb:.1f} MB")
        if status.vocabulary_size:
            print(f"Vocabulary size: {status.vocabulary_size:,}")
        
        return True
    except Exception as e:
        print(f"Error getting model status: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction capabilities."""
    print("\n=== Testing Feature Extraction ===")
    
    sample_text = """
    INVOICE #INV-123
    From: ABC Construction
    Amount Due: $5,000.00
    Date: 05/30/2025
    
    Please remit payment for concrete work completed.
    Signature: ________________
    """
    
    features = document_classifier._extract_features(sample_text, "Test Invoice")
    
    print(f"Text length: {features.text_length}")
    print(f"Word count: {features.word_count}")
    print(f"Has amounts: {features.has_amounts}")
    print(f"Has dates: {features.has_dates}")
    print(f"Has signature area: {features.has_signature_area}")
    print(f"Reference numbers: {features.reference_numbers}")
    print(f"Key phrases found: {len(features.key_phrases)}")
    print(f"Sample phrases: {features.key_phrases[:5]}")
    
    return features


def test_confidence_thresholds():
    """Test different confidence thresholds."""
    print("\n=== Testing Confidence Thresholds ===")
    
    ambiguous_text = "This is some generic text that doesn't clearly match any document type."
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        request = ClassificationRequest(
            text=ambiguous_text,
            min_confidence=threshold
        )
        
        result = document_classifier.classify(request)
        
        print(f"Threshold {threshold}: {result.document_type.value} (conf: {result.confidence:.3f})")
    
    return True


def main():
    """Run all AI classifier tests."""
    print("AI Classifier Integration Test")
    print("=" * 50)
    
    # Test model status first
    if not test_model_status():
        print("Model status test failed. Continuing with other tests...")
    
    # Test feature extraction
    test_feature_extraction()
    
    # Test confidence thresholds
    test_confidence_thresholds()
    
    # Test main classification functionality
    results = test_classifier_basic()
    
    # Overall assessment
    correct_count = sum(1 for r in results if r["correct"])
    total_tests = len(results)
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"AI Classifier Tests: {correct_count}/{total_tests} passed")
    
    if correct_count == total_tests:
        print("🎉 All tests passed! AI classifier is working correctly.")
        return 0
    elif correct_count > total_tests * 0.7:
        print("⚠️  Most tests passed. Minor issues may exist.")
        return 0
    else:
        print("❌ Many tests failed. Check classifier implementation.")
        return 1


if __name__ == "__main__":
    exit(main())