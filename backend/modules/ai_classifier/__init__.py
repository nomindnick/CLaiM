"""AI-powered document classification module for construction litigation documents."""

from .classifier import document_classifier, legacy_document_classifier
from .models import ClassificationResult, ClassificationRequest
from .llm_classifier import LLMDocumentClassifier

__all__ = ["document_classifier", "legacy_document_classifier", "LLMDocumentClassifier", "ClassificationResult", "ClassificationRequest"]