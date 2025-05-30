"""Pydantic models for document processing."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class DocumentType(str, Enum):
    """Types of construction documents."""
    
    EMAIL = "email"
    RFI = "rfi"  # Request for Information
    RFP = "rfp"  # Request for Proposal
    CHANGE_ORDER = "change_order"
    SUBMITTAL = "submittal"
    INVOICE = "invoice"
    CONTRACT = "contract"
    DRAWING = "drawing"
    SPECIFICATION = "specification"
    MEETING_MINUTES = "meeting_minutes"
    DAILY_REPORT = "daily_report"
    LETTER = "letter"
    MEMORANDUM = "memorandum"
    PAYMENT_APPLICATION = "payment_application"
    SCHEDULE = "schedule"
    UNKNOWN = "unknown"


class DocumentPage(BaseModel):
    """Represents a single page from a document."""
    
    page_number: int = Field(description="Page number in the source PDF")
    text: str = Field(default="", description="Extracted text from the page")
    is_scanned: bool = Field(default=False, description="Whether the page was scanned (OCR needed)")
    confidence: float = Field(default=1.0, description="OCR confidence score (0-1)")
    has_tables: bool = Field(default=False, description="Whether the page contains tables")
    has_images: bool = Field(default=False, description="Whether the page contains images")
    
    model_config = ConfigDict(from_attributes=True)


class Party(BaseModel):
    """Represents a party (person or organization) in a document."""
    
    name: str
    role: Optional[str] = None  # e.g., "Contractor", "Owner", "Architect"
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    
    title: Optional[str] = None
    date: Optional[datetime] = None
    parties: List[Party] = Field(default_factory=list)
    reference_numbers: List[str] = Field(default_factory=list)  # RFI#, CO#, etc.
    amounts: List[float] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Document relationships
    responds_to: Optional[str] = None  # Document ID this responds to
    references: List[str] = Field(default_factory=list)  # Other referenced documents
    
    # Additional metadata
    project_number: Optional[str] = None
    project_name: Optional[str] = None
    subject: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class Document(BaseModel):
    """Represents a logical document extracted from a PDF."""
    
    id: str = Field(description="Unique document identifier")
    source_pdf_id: str = Field(description="ID of the source PDF")
    source_pdf_path: Path = Field(description="Path to the source PDF")
    
    # Document content
    type: DocumentType = Field(default=DocumentType.UNKNOWN)
    pages: List[DocumentPage] = Field(default_factory=list)
    page_range: tuple[int, int] = Field(description="Start and end page numbers in source PDF")
    
    # Extracted data
    text: str = Field(default="", description="Full text of the document")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # Processing information
    classification_confidence: float = Field(default=0.0, description="Confidence in document type classification")
    processing_time: float = Field(default=0.0, description="Time taken to process (seconds)")
    processing_errors: List[str] = Field(default_factory=list)
    
    # Computed properties
    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)
    
    @property
    def has_ocr_content(self) -> bool:
        """Whether any pages required OCR."""
        return any(page.is_scanned for page in self.pages)
    
    @property
    def average_ocr_confidence(self) -> float:
        """Average OCR confidence across scanned pages."""
        scanned_pages = [p for p in self.pages if p.is_scanned]
        if not scanned_pages:
            return 1.0
        return sum(p.confidence for p in scanned_pages) / len(scanned_pages)
    
    model_config = ConfigDict(from_attributes=True)


class ProcessingResult(BaseModel):
    """Result of processing a PDF file."""
    
    success: bool
    source_pdf_path: Path
    source_pdf_id: str
    total_pages: int
    documents_found: int
    documents: List[Document] = Field(default_factory=list)
    processing_time: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Statistics
    page_classification: Dict[DocumentType, int] = Field(default_factory=dict)
    ocr_pages: int = 0
    average_confidence: float = 1.0
    
    # Boundary detection metadata
    detection_level: Optional[str] = None
    boundary_confidence: Optional[Dict[int, float]] = None
    
    model_config = ConfigDict(from_attributes=True)


class PDFProcessingRequest(BaseModel):
    """Request to process a PDF file."""
    
    file_path: Path
    split_documents: bool = True
    perform_ocr: bool = True
    extract_metadata: bool = True
    classify_documents: bool = True
    
    # Processing options
    ocr_language: str = "eng"
    min_confidence: float = 0.5
    max_pages: Optional[int] = None  # Limit processing for testing
    force_visual_detection: bool = False  # Force visual boundary detection
    
    model_config = ConfigDict(from_attributes=True)