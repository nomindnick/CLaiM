"""Data models for the storage module."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from modules.document_processor.models import DocumentType


class StorageStatus(str, Enum):
    """Status of storage operations."""
    PENDING = "pending"
    STORED = "stored"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentMetadata(BaseModel):
    """Metadata extracted from documents for storage and search."""
    model_config = ConfigDict(extra="allow")
    
    # Dates found in document
    dates: List[datetime] = Field(default_factory=list)
    
    # Parties mentioned (normalized)
    parties: List[str] = Field(default_factory=list)
    
    # Reference numbers (RFI#, CO#, Invoice#, etc.)
    reference_numbers: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Financial amounts
    amounts: List[float] = Field(default_factory=list)
    
    # Custom metadata fields
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class StoredPage(BaseModel):
    """Individual page stored in the database."""
    id: str
    document_id: str
    page_number: int
    text: str
    is_scanned: bool = False
    has_tables: bool = False
    has_images: bool = False
    ocr_confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StoredDocument(BaseModel):
    """Document stored in the database with all metadata."""
    # Identifiers
    id: str
    source_pdf_id: str
    source_pdf_path: Path
    
    # Document properties
    type: DocumentType = DocumentType.UNKNOWN
    page_range: tuple[int, int]  # (start, end) pages in source PDF
    page_count: int
    
    # Content
    title: Optional[str] = None
    text: str
    summary: Optional[str] = None
    
    # Metadata
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # AI-generated
    embedding: Optional[List[float]] = None  # Vector embeddings
    key_facts: List[str] = Field(default_factory=list)
    classification_confidence: float = 0.0
    
    # Storage info
    status: StorageStatus = StorageStatus.PENDING
    storage_path: Optional[Path] = None  # Path to extracted document PDF
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    
    # Relationships (document IDs)
    responds_to: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    related_documents: List[str] = Field(default_factory=list)


class SearchFilter(BaseModel):
    """Filters for document search."""
    # Text search
    query: Optional[str] = None
    
    # Type filter
    document_types: Optional[List[DocumentType]] = None
    
    # Date range
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    # Parties
    parties: Optional[List[str]] = None
    
    # Reference numbers
    reference_numbers: Optional[List[str]] = None
    
    # Amount range
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    
    # Metadata filters
    has_tables: Optional[bool] = None
    has_images: Optional[bool] = None
    is_ocr: Optional[bool] = None
    
    # Pagination
    offset: int = 0
    limit: int = 50
    
    # Sort options
    sort_by: str = "created_at"
    sort_descending: bool = True


class SearchResult(BaseModel):
    """Result from document search."""
    documents: List[StoredDocument]
    total_count: int
    offset: int
    limit: int
    search_time_ms: float
    
    # Search metadata
    query_used: Optional[str] = None
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    
    # Facets for filtering
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class StorageStats(BaseModel):
    """Statistics about stored documents."""
    total_documents: int = 0
    total_pages: int = 0
    total_pdfs: int = 0
    
    # Document type breakdown
    documents_by_type: Dict[DocumentType, int] = Field(default_factory=dict)
    
    # Storage sizes
    total_size_mb: float = 0.0
    database_size_mb: float = 0.0
    file_storage_size_mb: float = 0.0
    
    # Processing stats
    documents_with_ocr: int = 0
    documents_with_embeddings: int = 0
    average_pages_per_document: float = 0.0
    
    # Dates
    earliest_document: Optional[datetime] = None
    latest_document: Optional[datetime] = None
    last_indexed: Optional[datetime] = None


class BulkImportRequest(BaseModel):
    """Request for bulk document import."""
    source_pdf_paths: List[Path]
    process_options: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 10
    continue_on_error: bool = True


class BulkImportResult(BaseModel):
    """Result of bulk import operation."""
    total_files: int
    successful_imports: int
    failed_imports: int
    errors: List[Dict[str, str]] = Field(default_factory=list)
    import_time_seconds: float
    documents_created: int


class BulkDeleteRequest(BaseModel):
    """Request for bulk document deletion."""
    document_ids: List[str]
    delete_files: bool = True


class BulkDeleteResult(BaseModel):
    """Result of bulk delete operation."""
    total_documents: int
    successful_deletions: int
    failed_deletions: int
    errors: List[Dict[str, str]] = Field(default_factory=list)
    delete_time_seconds: float