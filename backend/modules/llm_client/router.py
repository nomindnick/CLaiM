"""Privacy-aware LLM routing for CLaiM document processing."""

import os
from typing import Optional, Dict, Any
from enum import Enum
from loguru import logger

from .base_client import LLMClient, LLMRequest, LLMResponse, LLMError, LLMServiceUnavailable, LLMTaskType
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
# Privacy mode enum (we'll define it here for now)
class PrivacyMode:
    FULL_LOCAL = "full_local"
    HYBRID_SAFE = "hybrid_safe"
    FULL_FEATURED = "full_featured"


class LLMRouter:
    """Routes LLM requests based on privacy settings and availability."""
    
    def __init__(self,
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "llama3:8b-instruct-q5_K_M",
                 openai_model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None):
        """Initialize LLM router.
        
        Args:
            ollama_host: Ollama service host
            ollama_model: Ollama model name
            openai_model: OpenAI model name
            openai_api_key: OpenAI API key
        """
        self.ollama_client = None
        self.openai_client = None
        
        # Initialize Ollama client
        try:
            self.ollama_client = OllamaClient(
                model_name=ollama_model,
                host=ollama_host
            )
            logger.info(f"Initialized Ollama client: {ollama_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAIClient(
                    model_name=openai_model,
                    api_key=api_key
                )
                logger.info(f"Initialized OpenAI client: {openai_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.info("No OpenAI API key provided, OpenAI client disabled")
        
        # Track client availability
        self._client_status = {}
        self._update_client_status()
    
    def _update_client_status(self):
        """Update client availability status."""
        self._client_status = {
            "ollama": self.ollama_client.is_available() if self.ollama_client else False,
            "openai": self.openai_client.is_available() if self.openai_client else False
        }
        
        logger.debug(f"Client status: {self._client_status}")
    
    def route_request(self, 
                     request: LLMRequest,
                     privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE,
                     force_local: bool = False,
                     has_sensitive_content: bool = False) -> LLMResponse:
        """Route LLM request based on privacy settings.
        
        Args:
            request: LLM request to process
            privacy_mode: Privacy mode setting
            force_local: Force use of local model
            has_sensitive_content: Whether content contains sensitive data
            
        Returns:
            LLM response
            
        Raises:
            LLMServiceUnavailable: If no suitable client is available
        """
        # Determine which client to use
        client = self._select_client(
            privacy_mode=privacy_mode,
            force_local=force_local,
            has_sensitive_content=has_sensitive_content
        )
        
        if not client:
            raise LLMServiceUnavailable("No suitable LLM client available")
        
        # Process request
        try:
            response = client.process_sync(request)
            logger.info(f"Request processed by {client.__class__.__name__}")
            return response
            
        except Exception as e:
            logger.error(f"Request failed with {client.__class__.__name__}: {e}")
            
            # Try fallback if primary client fails
            fallback_client = self._get_fallback_client(client)
            if fallback_client:
                logger.info(f"Attempting fallback to {fallback_client.__class__.__name__}")
                try:
                    response = fallback_client.process_sync(request)
                    logger.info(f"Fallback successful with {fallback_client.__class__.__name__}")
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Re-raise original error if no fallback worked
            raise e
    
    def _select_client(self,
                      privacy_mode: PrivacyMode,
                      force_local: bool,
                      has_sensitive_content: bool) -> Optional[LLMClient]:
        """Select appropriate client based on privacy settings.
        
        Args:
            privacy_mode: Privacy mode setting
            force_local: Force local processing
            has_sensitive_content: Whether content is sensitive
            
        Returns:
            Selected LLM client or None
        """
        # Update client status
        self._update_client_status()
        
        # Force local if requested
        if force_local:
            if self._client_status.get("ollama", False):
                return self.ollama_client
            else:
                logger.warning("Local processing forced but Ollama unavailable")
                return None
        
        # Handle privacy modes
        if privacy_mode == PrivacyMode.FULL_LOCAL:
            # Always use local
            if self._client_status.get("ollama", False):
                return self.ollama_client
            else:
                logger.error("Full local mode requested but Ollama unavailable")
                return None
        
        elif privacy_mode == PrivacyMode.HYBRID_SAFE:
            # Use cloud for non-sensitive content, local for sensitive
            if has_sensitive_content:
                if self._client_status.get("ollama", False):
                    return self.ollama_client
                else:
                    logger.warning("Sensitive content detected but Ollama unavailable")
                    return None
            else:
                # Prefer OpenAI for speed, fallback to Ollama
                if self._client_status.get("openai", False):
                    return self.openai_client
                elif self._client_status.get("ollama", False):
                    return self.ollama_client
                else:
                    return None
        
        elif privacy_mode == PrivacyMode.FULL_FEATURED:
            # Prefer cloud, fallback to local
            if self._client_status.get("openai", False):
                return self.openai_client
            elif self._client_status.get("ollama", False):
                return self.ollama_client
            else:
                return None
        
        else:
            logger.error(f"Unknown privacy mode: {privacy_mode}")
            return None
    
    def _get_fallback_client(self, failed_client: LLMClient) -> Optional[LLMClient]:
        """Get fallback client when primary fails.
        
        Args:
            failed_client: The client that failed
            
        Returns:
            Fallback client or None
        """
        if isinstance(failed_client, OpenAIClient):
            # OpenAI failed, try Ollama
            if self._client_status.get("ollama", False):
                return self.ollama_client
        elif isinstance(failed_client, OllamaClient):
            # Ollama failed, try OpenAI (if privacy allows)
            if self._client_status.get("openai", False):
                return self.openai_client
        
        return None
    
    def has_sensitive_content(self, text: str) -> bool:
        """Check if text contains sensitive content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if sensitive content detected
        """
        # Simple heuristic-based detection
        sensitive_patterns = [
            # Personal information
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email (if flagged as sensitive)
            
            # Financial information
            r'\$[\d,]+\.?\d*',  # Dollar amounts (if large)
            r'\b(?:salary|wage|payment|cost)\b.*\$[\d,]+',
            
            # Legal sensitive terms
            r'\b(?:confidential|proprietary|trade secret|attorney-client)\b',
            r'\b(?:settlement|litigation|lawsuit|damages)\b',
            
            # Contract sensitive terms
            r'\b(?:penalty|liquidated damages|termination)\b'
        ]
        
        import re
        text_lower = text.lower()
        
        for pattern in sensitive_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get router and client status.
        
        Returns:
            Status information dictionary
        """
        self._update_client_status()
        
        status = {
            "clients": self._client_status,
            "ollama_info": self.ollama_client.get_model_info() if self.ollama_client else None,
            "openai_info": self.openai_client.get_model_info() if self.openai_client else None
        }
        
        return status
    
    def classify_document(self,
                         text: str,
                         categories: Optional[list] = None,
                         context: Optional[str] = None,
                         privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE) -> LLMResponse:
        """Convenience method for document classification.
        
        Args:
            text: Document text to classify
            categories: List of possible categories
            context: Optional context (title, etc.)
            privacy_mode: Privacy mode for processing
            
        Returns:
            Classification response
        """
        from .prompt_templates import PromptTemplates
        
        # Check for sensitive content
        has_sensitive = self.has_sensitive_content(text)
        
        # Create classification prompt
        prompt = PromptTemplates.format_classification_prompt(text, categories, context)
        
        # Create request with longer timeout for large documents
        timeout = 90 if len(text) > 5000 else 60  # 90s for large docs, 60s for normal
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.CLASSIFICATION,
            temperature=0.1,
            max_tokens=200,
            timeout=timeout
        )
        
        # Route and process
        return self.route_request(
            request=request,
            privacy_mode=privacy_mode,
            has_sensitive_content=has_sensitive
        )
    
    def detect_boundary(self,
                       current_segment: str,
                       next_segment: str,
                       privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE) -> LLMResponse:
        """Convenience method for boundary detection.
        
        Args:
            current_segment: End of current document segment
            next_segment: Start of next segment
            privacy_mode: Privacy mode for processing
            
        Returns:
            Boundary detection response
        """
        from .prompt_templates import PromptTemplates
        
        # Check for sensitive content in both segments
        combined_text = current_segment + " " + next_segment
        has_sensitive = self.has_sensitive_content(combined_text)
        
        # Create boundary detection prompt
        prompt = PromptTemplates.format_boundary_prompt(current_segment, next_segment)
        
        # Create request with timeout based on segment size
        timeout = 75 if len(combined_text) > 3000 else 60
        request = LLMRequest(
            prompt=prompt,
            task_type=LLMTaskType.BOUNDARY_DETECTION,
            temperature=0.1,
            max_tokens=100,
            timeout=timeout
        )
        
        # Route and process
        return self.route_request(
            request=request,
            privacy_mode=privacy_mode,
            has_sensitive_content=has_sensitive
        )