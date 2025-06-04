"""Ollama client for local LLM processing."""

import asyncio
import json
import time
from typing import Dict, Any, Optional
import httpx
from loguru import logger

from .base_client import LLMClient, LLMRequest, LLMResponse, LLMError, LLMServiceUnavailable, LLMTimeout, LLMTaskType


class OllamaClient(LLMClient):
    """Client for Ollama local LLM service."""
    
    def __init__(self, 
                 model_name: str = "llama3:8b-instruct-q5_K_M",
                 host: str = "http://localhost:11434",
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.host = host.rstrip('/')
        self.api_base = f"{self.host}/api"
        
        # Ollama-specific settings
        self.keep_alive = kwargs.get('keep_alive', '5m')  # Keep model loaded for 5 minutes
        self.num_ctx = kwargs.get('num_ctx', 8192)  # Context window
        
    async def process(self, request: LLMRequest) -> LLMResponse:
        """Process request using Ollama API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": request.max_tokens or self.default_max_tokens,
                }
            }
            
            # Add keep_alive to prevent model unloading
            if self.keep_alive:
                payload["keep_alive"] = self.keep_alive
            
            logger.debug(f"Sending Ollama request to {self.model_name}")
            
            # Make API call
            async with httpx.AsyncClient(timeout=request.timeout or self.default_timeout) as client:
                response = await client.post(
                    f"{self.api_base}/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise LLMError(error_msg)
                
                result = response.json()
                
                # Extract response content
                content = result.get("response", "").strip()
                
                if not content:
                    raise LLMError("Empty response from Ollama")
                
                processing_time = time.time() - start_time
                
                # Parse token usage if available
                token_usage = None
                if "eval_count" in result:
                    token_usage = {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    }
                
                logger.info(f"Ollama response completed in {processing_time:.2f}s")
                
                return LLMResponse(
                    content=content,
                    model_used=self.model_name,
                    processing_time=processing_time,
                    token_usage=token_usage,
                    raw_response=result
                )
                
        except httpx.TimeoutException:
            error_msg = f"Ollama request timed out after {request.timeout or self.default_timeout}s"
            logger.error(error_msg)
            raise LLMTimeout(error_msg)
            
        except httpx.ConnectError:
            error_msg = "Cannot connect to Ollama service"
            logger.error(error_msg)
            raise LLMServiceUnavailable(error_msg)
            
        except Exception as e:
            error_msg = f"Ollama client error: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def is_available(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if service is available
        """
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model.
        
        Returns:
            Dictionary with model information
        """
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                # Get model list
                response = client.get(f"{self.host}/api/tags")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    
                    # Find our model
                    for model in models:
                        if model.get("name", "").startswith(self.model_name.split(":")[0]):
                            return {
                                "name": model.get("name", self.model_name),
                                "size": model.get("size", 0),
                                "digest": model.get("digest", "unknown"),
                                "modified_at": model.get("modified_at", "unknown"),
                                "host": self.host,
                                "context_window": self.num_ctx,
                                "available": True
                            }
                
                # If model not found in list, return basic info
                return {
                    "name": self.model_name,
                    "host": self.host,
                    "context_window": self.num_ctx,
                    "available": self.is_available()
                }
                
        except Exception as e:
            logger.warning(f"Could not get Ollama model info: {e}")
            return {
                "name": self.model_name,
                "host": self.host,
                "error": str(e),
                "available": False
            }
    
    def pull_model(self) -> bool:
        """Pull/download the model if not available.
        
        Returns:
            True if model is available after pull
        """
        try:
            import httpx
            logger.info(f"Pulling Ollama model: {self.model_name}")
            
            with httpx.Client(timeout=300.0) as client:  # 5 minute timeout for model download
                response = client.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model_name}
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model: {self.model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload the model from memory.
        
        Returns:
            True if model was unloaded
        """
        try:
            import httpx
            logger.info(f"Unloading Ollama model: {self.model_name}")
            
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model_name,
                        "keep_alive": 0  # Unload immediately
                    }
                )
                
                # Note: This endpoint may return an error but still unload the model
                logger.info(f"Model unload requested for: {self.model_name}")
                return True
                
        except Exception as e:
            logger.warning(f"Error unloading model: {e}")
            return False
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Simple synchronous completion method for compatibility.
        
        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.GENERAL,
            temperature=kwargs.get('temperature', self.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.default_max_tokens),
            timeout=kwargs.get('timeout', self.default_timeout)
        )
        
        response = self.process_sync(request)
        return response.content