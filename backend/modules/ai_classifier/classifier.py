"""Main document classifier module - now using LLM-based classification.

This module serves as the interface for document classification, now powered by
Large Language Models for dramatically improved accuracy on real-world documents.
"""

from .llm_classifier import llm_document_classifier

# Export the LLM classifier as the main interface
document_classifier = llm_document_classifier

# For backward compatibility, export the legacy classifier as well
try:
    from .legacy_classifier import document_classifier as legacy_document_classifier
except ImportError:
    # If legacy classifier import fails, create a dummy for safety
    legacy_document_classifier = None

# Export main classes and functions for backward compatibility
from .models import (
    ClassificationResult,
    ClassificationRequest, 
    ClassificationFeatures
)

__all__ = [
    'document_classifier',
    'legacy_document_classifier',
    'ClassificationResult',
    'ClassificationRequest',
    'ClassificationFeatures'
]