"""Base LLM client abstraction for CLaiM document processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class LLMTaskType(Enum):
    """Types of LLM tasks."""
    CLASSIFICATION = "classification"
    BOUNDARY_DETECTION = "boundary_detection"
    GENERAL = "general"


@dataclass
class LLMRequest:
    """Request for LLM processing."""
    prompt: str
    task_type: LLMTaskType
    model_params: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    timeout: int = 300  # 5 minutes


@dataclass
class LLMResponse:
    """Response from LLM processing."""
    content: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.default_timeout = kwargs.get('timeout', 60)  # Increased from 30s to 60s
        self.default_temperature = kwargs.get('temperature', 0.1)
        self.default_max_tokens = kwargs.get('max_tokens', 4000)
    
    @abstractmethod
    async def process(self, request: LLMRequest) -> LLMResponse:
        """Process an LLM request.
        
        Args:
            request: LLM request with prompt and parameters
            
        Returns:
            LLM response with generated content
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available.
        
        Returns:
            True if service is available
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def process_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for process method."""
        import asyncio
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we need to use a different approach
            import concurrent.futures
            import threading
            
            # Create a new event loop in a separate thread
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.process(request))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=request.timeout)
                
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.process(request))
    
    def classify_document(self, 
                         text: str, 
                         categories: List[str],
                         context: Optional[str] = None) -> LLMResponse:
        """Convenience method for document classification.
        
        Args:
            text: Document text to classify
            categories: List of possible categories
            context: Optional context (title, etc.)
            
        Returns:
            Classification response
        """
        from .prompt_templates import PromptTemplates
        
        prompt = PromptTemplates.format_classification_prompt(
            text, categories, context
        )
        
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.CLASSIFICATION,
            temperature=0.1,
            max_tokens=200
        )
        
        return self.process_sync(request)
    
    def detect_boundary(self,
                       current_segment: str,
                       next_segment: str) -> LLMResponse:
        """Convenience method for boundary detection.
        
        Args:
            current_segment: End of current document segment
            next_segment: Start of next segment
            
        Returns:
            Boundary detection response
        """
        from .prompt_templates import PromptTemplates
        
        prompt = PromptTemplates.format_boundary_prompt(
            current_segment, next_segment
        )
        
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.BOUNDARY_DETECTION,
            temperature=0.1,
            max_tokens=100
        )
        
        return self.process_sync(request)


class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMServiceUnavailable(LLMError):
    """Exception raised when LLM service is unavailable."""
    pass


class LLMTimeout(LLMError):
    """Exception raised when LLM request times out."""
    pass


class LLMRateLimited(LLMError):
    """Exception raised when LLM requests are rate limited."""
    pass