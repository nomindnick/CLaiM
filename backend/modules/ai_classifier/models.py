"""Pydantic models for document classification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from ..document_processor.models import DocumentType


class ClassificationFeatures(BaseModel):
    """Features extracted from a document for classification."""
    
    text_length: int = Field(description="Number of characters in document")
    word_count: int = Field(description="Number of words in document")
    has_numbers: bool = Field(description="Contains numeric references")
    has_amounts: bool = Field(description="Contains dollar amounts")
    has_dates: bool = Field(description="Contains date references")
    has_signature_area: bool = Field(description="Has signature or approval areas")
    has_forms: bool = Field(description="Contains form-like structures")
    has_tables: bool = Field(description="Contains tables")
    
    # Content patterns
    subject_line: Optional[str] = None
    reference_numbers: List[str] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """Result of document classification."""
    
    document_type: DocumentType = Field(description="Predicted document type")
    confidence: float = Field(description="Classification confidence (0-1)", ge=0.0, le=1.0)
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative classifications with confidence scores"
    )
    features: ClassificationFeatures = Field(description="Features used for classification")
    reasoning: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of classification decision"
    )
    processing_time: float = Field(default=0.0, description="Time taken for classification")


class ClassificationRequest(BaseModel):
    """Request for document classification."""
    
    text: str = Field(description="Document text to classify")
    title: Optional[str] = None
    metadata: Optional[Dict] = None
    require_reasoning: bool = Field(default=False, description="Include explanation in result")
    min_confidence: float = Field(default=0.3, description="Minimum confidence for classification")


class ModelStatus(BaseModel):
    """Status of classification models."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    distilbert_loaded: bool = Field(description="DistilBERT model is loaded")
    model_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    vocabulary_size: Optional[int] = None
    supported_classes: List[str] = Field(default_factory=list)
    last_loaded: Optional[str] = None