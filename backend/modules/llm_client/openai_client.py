"""OpenAI client for cloud LLM processing."""

import time
from typing import Dict, Any, Optional
from loguru import logger

from .base_client import LLMClient, LLMRequest, LLMResponse, LLMError, LLMServiceUnavailable, LLMTimeout, LLMRateLimited


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",  # Updated to current model
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self._client = None
        
        # OpenAI-specific settings
        self.max_retries = kwargs.get('max_retries', 3)
        self.request_timeout = kwargs.get('request_timeout', 30)
        
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=self.request_timeout,
                    max_retries=self.max_retries
                )
            except ImportError:
                raise LLMError("OpenAI library not installed. Install with: pip install openai")
            except Exception as e:
                raise LLMError(f"Failed to initialize OpenAI client: {e}")
        
        return self._client
    
    async def process(self, request: LLMRequest) -> LLMResponse:
        """Process request using OpenAI API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
            
            # Build request parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens or self.default_max_tokens,
                "timeout": request.timeout or self.default_timeout
            }
            
            logger.debug(f"Sending OpenAI request to {self.model_name}")
            
            # Make API call
            response = client.chat.completions.create(**params)
            
            # Extract response content
            if not response.choices:
                raise LLMError("No choices in OpenAI response")
            
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty content in OpenAI response")
            
            processing_time = time.time() - start_time
            
            # Extract token usage
            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            logger.info(f"OpenAI response completed in {processing_time:.2f}s")
            
            return LLMResponse(
                content=content.strip(),
                model_used=response.model,
                processing_time=processing_time,
                token_usage=token_usage,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle specific OpenAI errors
            if "rate limit" in error_msg or "quota" in error_msg:
                logger.error(f"OpenAI rate limit exceeded: {e}")
                raise LLMRateLimited(f"OpenAI rate limit: {e}")
            elif "timeout" in error_msg:
                logger.error(f"OpenAI request timed out: {e}")
                raise LLMTimeout(f"OpenAI timeout: {e}")
            elif "api key" in error_msg or "unauthorized" in error_msg:
                logger.error(f"OpenAI authentication error: {e}")
                raise LLMServiceUnavailable(f"OpenAI auth error: {e}")
            elif "service unavailable" in error_msg or "connection" in error_msg:
                logger.error(f"OpenAI service unavailable: {e}")
                raise LLMServiceUnavailable(f"OpenAI unavailable: {e}")
            else:
                logger.error(f"OpenAI client error: {e}")
                raise LLMError(f"OpenAI error: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available.
        
        Returns:
            True if service is available
        """
        try:
            if not self.api_key:
                return False
                
            client = self._get_client()
            
            # Make a minimal test request
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            
            return True
            
        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            "name": self.model_name,
            "provider": "openai",
            "available": self.is_available()
        }
        
        # Add known model specifications
        model_specs = {
            "gpt-4o": {
                "context_window": 128000,
                "max_output": 16384,
                "pricing_per_1m": {"input": 2.50, "output": 10.00}
            },
            "gpt-4o-mini": {
                "context_window": 128000,
                "max_output": 16384,
                "pricing_per_1m": {"input": 0.15, "output": 0.60}
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "max_output": 4096,
                "pricing_per_1m": {"input": 10.00, "output": 30.00}
            }
        }
        
        if self.model_name in model_specs:
            model_info.update(model_specs[self.model_name])
        
        try:
            # Try to get real-time model info if available
            client = self._get_client()
            models = client.models.list()
            
            for model in models.data:
                if model.id == self.model_name:
                    model_info.update({
                        "id": model.id,
                        "created": model.created,
                        "owned_by": model.owned_by
                    })
                    break
                    
        except Exception as e:
            logger.debug(f"Could not fetch OpenAI model details: {e}")
            model_info["error"] = str(e)
        
        return model_info
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Estimate cost for token usage.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD, or None if pricing unknown
        """
        model_info = self.get_model_info()
        pricing = model_info.get("pricing_per_1m")
        
        if not pricing:
            return None
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost