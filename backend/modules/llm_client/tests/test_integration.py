"""Integration tests for LLM client infrastructure."""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from modules.llm_client.ollama_client import OllamaClient
from modules.llm_client.router import LLMRouter, PrivacyMode
from modules.llm_client.base_client import LLMRequest, LLMTaskType
from modules.llm_client.prompt_templates import PromptTemplates


def test_ollama_connectivity():
    """Test basic Ollama connectivity."""
    print("=== Testing Ollama Connectivity ===")
    
    client = OllamaClient()
    
    # Check availability
    available = client.is_available()
    print(f"Ollama available: {available}")
    
    if not available:
        print("❌ Ollama service not available")
        return False
    
    # Get model info
    model_info = client.get_model_info()
    print(f"Model info: {model_info}")
    
    return True


def test_basic_generation():
    """Test basic text generation."""
    print("\n=== Testing Basic Generation ===")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama not available, skipping generation test")
        return False
    
    try:
        # Simple test request
        request = LLMRequest(
            prompt="Respond with exactly: 'Hello from Llama 3'",
            task_type=LLMTaskType.GENERAL,
            max_tokens=20,
            temperature=0.1
        )
        
        print("Sending test request...")
        response = client.process_sync(request)
        
        print(f"Response: {response.content}")
        print(f"Model used: {response.model_used}")
        print(f"Processing time: {response.processing_time:.2f}s")
        
        if response.token_usage:
            print(f"Token usage: {response.token_usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False


def test_classification_prompt():
    """Test document classification prompt."""
    print("\n=== Testing Classification Prompt ===")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama not available, skipping classification test")
        return False
    
    try:
        # Test email classification
        email_text = """From: john@contractor.com
To: susan@owner.com
Subject: RE: Project Update

Dear Susan,

Thank you for your email regarding the foundation work. We have completed the excavation and will begin pouring concrete tomorrow morning. Please let me know if you have any questions.

Best regards,
John Smith"""
        
        prompt = PromptTemplates.format_classification_prompt(email_text)
        
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.CLASSIFICATION,
            max_tokens=100,
            temperature=0.1
        )
        
        print("Sending classification request...")
        response = client.process_sync(request)
        
        print(f"Raw response: {response.content}")
        
        # Parse response
        parsed = PromptTemplates.parse_classification_response(response.content)
        print(f"Parsed result: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False


def test_boundary_detection():
    """Test boundary detection prompt."""
    print("\n=== Testing Boundary Detection ===")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama not available, skipping boundary test")
        return False
    
    try:
        # Test clear boundary
        current_segment = """Best regards,
John Smith
Project Manager
ABC Construction

[End of Email]"""
        
        next_segment = """REQUEST FOR INFORMATION
RFI #045
Date: March 15, 2024
Project: Elementary School

To: Design Team
From: General Contractor"""
        
        prompt = PromptTemplates.format_boundary_prompt(current_segment, next_segment)
        
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.BOUNDARY_DETECTION,
            max_tokens=50,
            temperature=0.1
        )
        
        print("Sending boundary detection request...")
        response = client.process_sync(request)
        
        print(f"Raw response: {response.content}")
        
        # Parse response
        parsed = PromptTemplates.parse_boundary_response(response.content)
        print(f"Parsed result: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Boundary detection test failed: {e}")
        return False


def test_router():
    """Test LLM router functionality."""
    print("\n=== Testing LLM Router ===")
    
    try:
        router = LLMRouter()
        
        # Get status
        status = router.get_status()
        print(f"Router status: {status}")
        
        # Test classification through router
        email_text = """INVOICE #INV-2024-0892
ABC Construction Company
123 Main Street

Bill To: School District
Date: March 15, 2024
Due Date: April 15, 2024

Description: Concrete work - Foundation
Quantity: 250 cubic yards
Rate: $150.00
Amount: $37,500.00

Total Amount Due: $37,500.00"""
        
        if status["clients"]["ollama"]:
            print("Testing router classification...")
            response = router.classify_document(
                text=email_text,
                privacy_mode=PrivacyMode.FULL_LOCAL
            )
            
            print(f"Router response: {response.content}")
            
            # Parse response
            parsed = PromptTemplates.parse_classification_response(response.content)
            print(f"Parsed result: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmark."""
    print("\n=== Performance Benchmark ===")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama not available, skipping benchmark")
        return
    
    try:
        # Test different request sizes
        test_cases = [
            ("Short", "Classify this: Email from contractor"),
            ("Medium", "Classify this document: " + "A" * 500),
            ("Long", "Classify this document: " + "B" * 2000)
        ]
        
        for name, text in test_cases:
            request = LLMRequest(
                prompt=f"Respond with: 'Processed {name} request'",
                task_type=LLMTaskType.GENERAL,
                max_tokens=20,
                temperature=0.1
            )
            
            response = client.process_sync(request)
            print(f"{name} request: {response.processing_time:.2f}s")
            
            if response.token_usage:
                tokens = response.token_usage
                print(f"  Tokens: {tokens['prompt_tokens']} in, {tokens['completion_tokens']} out")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def main():
    """Run all tests."""
    print("CLaiM LLM Client Integration Tests")
    print("=" * 40)
    
    success_count = 0
    total_tests = 5
    
    # Run tests
    if test_ollama_connectivity():
        success_count += 1
    
    if test_basic_generation():
        success_count += 1
    
    if test_classification_prompt():
        success_count += 1
    
    if test_boundary_detection():
        success_count += 1
    
    if test_router():
        success_count += 1
    
    # Run benchmark
    run_performance_benchmark()
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ All tests passed! LLM infrastructure is ready.")
    else:
        print(f"❌ {total_tests - success_count} tests failed.")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()