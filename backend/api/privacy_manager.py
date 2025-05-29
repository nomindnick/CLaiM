"""Privacy mode management for AI operations."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .config import PrivacyMode, settings


class PrivacyContext(BaseModel):
    """Context for privacy-aware operations."""
    
    mode: PrivacyMode
    has_sensitive_content: bool = False
    user_consent: bool = True
    operation_type: str = "general"


class PrivacyManager:
    """Manages privacy modes and routes AI operations accordingly."""
    
    def __init__(self):
        self.current_mode = settings.privacy_mode
        self._sensitive_patterns = [
            "ssn", "social security",
            "tax id", "ein",
            "bank account", "routing number",
            "medical", "health",
            "salary", "compensation",
        ]
    
    def set_mode(self, mode: PrivacyMode) -> None:
        """Update the current privacy mode."""
        self.current_mode = mode
    
    def get_mode(self) -> PrivacyMode:
        """Get the current privacy mode."""
        return self.current_mode
    
    def check_sensitive_content(self, content: str) -> bool:
        """Check if content contains potentially sensitive information."""
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in self._sensitive_patterns)
    
    def should_use_local(self, context: PrivacyContext) -> bool:
        """Determine if local processing should be used."""
        if context.mode == PrivacyMode.FULL_LOCAL:
            return True
        
        if context.has_sensitive_content:
            return True
        
        if not context.user_consent:
            return True
        
        # For hybrid mode, use local for certain operations
        if context.mode == PrivacyMode.HYBRID_SAFE:
            local_operations = ["classification", "extraction", "embedding"]
            if context.operation_type in local_operations:
                return True
        
        return False
    
    def get_privacy_info(self) -> Dict[str, Any]:
        """Get current privacy status information."""
        return {
            "mode": self.current_mode.value,
            "description": self._get_mode_description(),
            "local_only": self.current_mode == PrivacyMode.FULL_LOCAL,
            "api_enabled": self.current_mode in [PrivacyMode.HYBRID_SAFE, PrivacyMode.FULL_FEATURED],
        }
    
    def _get_mode_description(self) -> str:
        """Get human-readable description of current mode."""
        descriptions = {
            PrivacyMode.FULL_LOCAL: "All processing happens on your computer. No data leaves your system.",
            PrivacyMode.HYBRID_SAFE: "Core operations are local. Only non-sensitive analysis uses cloud APIs.",
            PrivacyMode.FULL_FEATURED: "Cloud APIs are used for enhanced capabilities when beneficial.",
        }
        return descriptions.get(self.current_mode, "Unknown mode")


# Global privacy manager instance
privacy_manager = PrivacyManager()