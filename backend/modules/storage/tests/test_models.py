"""Tests for storage models."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.modules.document_processor.models import DocumentType
from backend.modules.storage.models import (
    StorageStatus,
    DocumentMetadata,
    StoredPage,
    StoredDocument,
    SearchFilter,
    SearchResult,
    StorageStats,
)


class TestDocumentMetadata:
    """Test DocumentMetadata model."""
    
    def test_empty_metadata(self):
        """Test creating empty metadata."""
        metadata = DocumentMetadata()
        assert metadata.dates == []
        assert metadata.parties == []
        assert metadata.reference_numbers == {}
        assert metadata.amounts == []
        assert metadata.custom_fields == {}
    
    def test_metadata_with_data(self):
        """Test metadata with various fields."""
        metadata = DocumentMetadata(
            dates=[datetime(2024, 1, 15), datetime(2024, 2, 1)],
            parties=["ABC Construction", "XYZ School District"],
            reference_numbers={"RFI": ["123", "124"], "CO": ["45"]},
            amounts=[10000.0, 25000.50],
            custom_fields={"project_number": "P-2024-001"}
        )
        
        assert len(metadata.dates) == 2
        assert "ABC Construction" in metadata.parties
        assert metadata.reference_numbers["RFI"] == ["123", "124"]
        assert 25000.50 in metadata.amounts
        assert metadata.custom_fields["project_number"] == "P-2024-001"


class TestStoredPage:
    """Test StoredPage model."""
    
    def test_create_page(self):
        """Test creating a stored page."""
        page = StoredPage(
            id="page-123",
            document_id="doc-456",
            page_number=1,
            text="This is page content",
        )
        
        assert page.id == "page-123"
        assert page.document_id == "doc-456"
        assert page.page_number == 1
        assert page.text == "This is page content"
        assert page.is_scanned is False
        assert page.ocr_confidence == 1.0
    
    def test_scanned_page(self):
        """Test creating a scanned page."""
        page = StoredPage(
            id="page-789",
            document_id="doc-456",
            page_number=2,
            text="OCR extracted text",
            is_scanned=True,
            ocr_confidence=0.85,
        )
        
        assert page.is_scanned is True
        assert page.ocr_confidence == 0.85


class TestStoredDocument:
    """Test StoredDocument model."""
    
    def test_create_document(self):
        """Test creating a stored document."""
        doc = StoredDocument(
            id="doc-123",
            source_pdf_id="pdf-456",
            source_pdf_path=Path("/storage/pdfs/test.pdf"),
            type=DocumentType.RFI,
            page_range=(1, 3),
            page_count=3,
            text="Document content here",
        )
        
        assert doc.id == "doc-123"
        assert doc.source_pdf_id == "pdf-456"
        assert doc.type == DocumentType.RFI
        assert doc.page_range == (1, 3)
        assert doc.status == StorageStatus.PENDING
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.embedding is None
    
    def test_document_with_metadata(self):
        """Test document with full metadata."""
        metadata = DocumentMetadata(
            dates=[datetime(2024, 1, 15)],
            parties=["ABC Construction"],
            reference_numbers={"RFI": ["123"]},
            amounts=[10000.0],
        )
        
        doc = StoredDocument(
            id="doc-789",
            source_pdf_id="pdf-789",
            source_pdf_path=Path("/storage/pdfs/test.pdf"),
            type=DocumentType.INVOICE,
            page_range=(5, 6),
            page_count=2,
            title="Invoice #123",
            text="Invoice content",
            metadata=metadata,
            classification_confidence=0.95,
            status=StorageStatus.INDEXED,
        )
        
        assert doc.title == "Invoice #123"
        assert doc.metadata.reference_numbers["RFI"] == ["123"]
        assert doc.classification_confidence == 0.95
        assert doc.status == StorageStatus.INDEXED


class TestSearchFilter:
    """Test SearchFilter model."""
    
    def test_empty_filter(self):
        """Test empty search filter."""
        filter = SearchFilter()
        assert filter.query is None
        assert filter.document_types is None
        assert filter.limit == 50
        assert filter.offset == 0
        assert filter.sort_by == "created_at"
        assert filter.sort_descending is True
    
    def test_filter_with_criteria(self):
        """Test search filter with various criteria."""
        filter = SearchFilter(
            query="construction delay",
            document_types=[DocumentType.RFI, DocumentType.EMAIL],
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31),
            parties=["ABC Construction"],
            amount_min=1000.0,
            amount_max=50000.0,
            limit=20,
            offset=40,
        )
        
        assert filter.query == "construction delay"
        assert DocumentType.RFI in filter.document_types
        assert filter.date_from.year == 2024
        assert "ABC Construction" in filter.parties
        assert filter.amount_min == 1000.0
        assert filter.limit == 20
        assert filter.offset == 40


class TestSearchResult:
    """Test SearchResult model."""
    
    def test_empty_result(self):
        """Test empty search result."""
        result = SearchResult(
            documents=[],
            total_count=0,
            offset=0,
            limit=50,
            search_time_ms=12.5,
        )
        
        assert len(result.documents) == 0
        assert result.total_count == 0
        assert result.search_time_ms == 12.5
    
    def test_result_with_documents(self):
        """Test search result with documents."""
        doc = StoredDocument(
            id="doc-123",
            source_pdf_id="pdf-456",
            source_pdf_path=Path("/storage/pdfs/test.pdf"),
            type=DocumentType.RFI,
            page_range=(1, 3),
            page_count=3,
            text="Document content",
        )
        
        result = SearchResult(
            documents=[doc],
            total_count=15,
            offset=10,
            limit=50,
            search_time_ms=25.3,
            query_used="test query",
            filters_applied={"document_types": ["RFI"]},
        )
        
        assert len(result.documents) == 1
        assert result.total_count == 15
        assert result.query_used == "test query"
        assert "document_types" in result.filters_applied


class TestStorageStats:
    """Test StorageStats model."""
    
    def test_empty_stats(self):
        """Test empty storage stats."""
        stats = StorageStats()
        assert stats.total_documents == 0
        assert stats.total_pages == 0
        assert stats.total_pdfs == 0
        assert stats.documents_by_type == {}
        assert stats.total_size_mb == 0.0
    
    def test_stats_with_data(self):
        """Test storage stats with data."""
        stats = StorageStats(
            total_documents=150,
            total_pages=2500,
            total_pdfs=10,
            documents_by_type={
                DocumentType.RFI: 50,
                DocumentType.EMAIL: 75,
                DocumentType.INVOICE: 25,
            },
            total_size_mb=1250.5,
            database_size_mb=50.5,
            file_storage_size_mb=1200.0,
            documents_with_ocr=30,
            documents_with_embeddings=140,
            average_pages_per_document=16.67,
            earliest_document=datetime(2023, 1, 1),
            latest_document=datetime(2024, 12, 15),
        )
        
        assert stats.total_documents == 150
        assert stats.documents_by_type[DocumentType.RFI] == 50
        assert stats.total_size_mb == 1250.5
        assert stats.documents_with_ocr == 30
        assert stats.average_pages_per_document == 16.67
        assert stats.earliest_document.year == 2023