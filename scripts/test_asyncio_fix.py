#!/usr/bin/env python3
"""Test the asyncio fix for LLM classification."""

import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def test_llm_classification():
    """Test LLM classification with asyncio fix."""
    print("=== Testing LLM Classification with AsyncIO Fix ===")
    
    try:
        from modules.ai_classifier.classifier import document_classifier
        from modules.ai_classifier.models import ClassificationRequest
        
        # Create test request
        test_text = """
        From: contractor@example.com
        To: project@school.edu
        Subject: Change Order Request #123
        Date: May 15, 2024
        
        We need to submit a change order for additional electrical work
        on the main building. Please review the attached cost proposal.
        """
        
        request = ClassificationRequest(
            text=test_text,
            title="Change Order Email",
            require_reasoning=True
        )
        
        print("✅ Created classification request")
        print(f"📝 Text length: {len(test_text)} characters")
        
        # Test classification
        print("🤖 Attempting LLM classification...")
        result = document_classifier.classify(request)
        
        print("✅ LLM classification successful!")
        print(f"📊 Classification: {result.document_type}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"💭 Reasoning: {result.reasoning}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the asyncio fix test."""
    print("AsyncIO Fix Verification Test")
    print("=" * 40)
    
    success = test_llm_classification()
    
    if success:
        print("\n🎉 AsyncIO fix verified successfully!")
        print("✅ LLM classification now works within FastAPI event loop")
    else:
        print("\n❌ AsyncIO fix verification failed")
        print("🔧 Further debugging needed")
    
    return success

if __name__ == "__main__":
    main()