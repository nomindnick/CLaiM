"""Custom exceptions for CLaiM application."""

from typing import Any, Dict, Optional


class CLaiMException(Exception):
    """Base exception for all CLaiM errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentProcessingError(CLaiMException):
    """Raised when document processing fails."""
    pass


class PDFExtractionError(DocumentProcessingError):
    """Raised when PDF extraction fails."""
    pass


class ClassificationError(CLaiMException):
    """Raised when document classification fails."""
    pass


class StorageError(CLaiMException):
    """Raised when storage operations fail."""
    pass


class SearchError(CLaiMException):
    """Raised when search operations fail."""
    pass


class ModelLoadError(CLaiMException):
    """Raised when AI model loading fails."""
    pass


class PrivacyError(CLaiMException):
    """Raised when privacy constraints are violated."""
    pass


class ValidationError(CLaiMException):
    """Raised when input validation fails."""
    pass


class OCRError(DocumentProcessingError):
    """Raised when OCR processing fails."""
    pass