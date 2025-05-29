"""Tests for document processor models."""

from datetime import datetime
from pathlib import Path

import pytest

from ..models import (
    Document,
    DocumentPage,
    DocumentMetadata,
    DocumentType,
    Party,
    ProcessingResult,
    PDFProcessingRequest,
)


class TestDocumentModels:
    """Test document-related models."""
    
    def test_document_page_creation(self):
        """Test creating a DocumentPage."""
        page = DocumentPage(
            page_number=1,
            text="Sample text",
            is_scanned=False,
            confidence=0.95,
            has_tables=True,
            has_images=False,
        )
        
        assert page.page_number == 1
        assert page.text == "Sample text"
        assert not page.is_scanned
        assert page.confidence == 0.95
        assert page.has_tables
        assert not page.has_images
    
    def test_party_creation(self):
        """Test creating a Party."""
        party = Party(
            name="ABC Construction Inc.",
            role="Contractor",
            email="info@abc.com",
            phone="555-1234",
            company="ABC Construction Inc.",
        )
        
        assert party.name == "ABC Construction Inc."
        assert party.role == "Contractor"
        assert party.email == "info@abc.com"
    
    def test_document_metadata_creation(self):
        """Test creating DocumentMetadata."""
        metadata = DocumentMetadata(
            title="RFI #123",
            date=datetime(2024, 1, 15),
            parties=[
                Party(name="John Doe", role="Architect"),
                Party(name="Jane Smith", role="Contractor"),
            ],
            reference_numbers=["RFI-123", "CO-456"],
            amounts=[10000.50, 25000.00],
            project_number="P-2024-001",
            project_name="School Renovation",
        )
        
        assert metadata.title == "RFI #123"
        assert metadata.date.year == 2024
        assert len(metadata.parties) == 2
        assert metadata.parties[0].name == "John Doe"
        assert len(metadata.reference_numbers) == 2
        assert sum(metadata.amounts) == 35000.50
    
    def test_document_creation(self):
        """Test creating a Document."""
        pages = [
            DocumentPage(page_number=1, text="Page 1"),
            DocumentPage(page_number=2, text="Page 2", is_scanned=True, confidence=0.8),
        ]
        
        doc = Document(
            id="doc-123",
            source_pdf_id="pdf-456",
            source_pdf_path=Path("/tmp/test.pdf"),
            type=DocumentType.RFI,
            pages=pages,
            page_range=(1, 2),
            text="Page 1\n\nPage 2",
            classification_confidence=0.95,
        )
        
        assert doc.id == "doc-123"
        assert doc.type == DocumentType.RFI
        assert doc.page_count == 2
        assert doc.has_ocr_content
        assert doc.average_ocr_confidence == 0.8
    
    def test_document_type_enum(self):
        """Test DocumentType enum."""
        assert DocumentType.EMAIL.value == "email"
        assert DocumentType.RFI.value == "rfi"
        assert DocumentType.CHANGE_ORDER.value == "change_order"
        
        # Test that all types are unique
        all_types = [dt.value for dt in DocumentType]
        assert len(all_types) == len(set(all_types))
    
    def test_processing_result(self):
        """Test ProcessingResult model."""
        result = ProcessingResult(
            success=True,
            source_pdf_path=Path("/tmp/test.pdf"),
            source_pdf_id="pdf-123",
            total_pages=10,
            documents_found=3,
            processing_time=5.23,
            page_classification={
                DocumentType.EMAIL: 2,
                DocumentType.RFI: 1,
            },
            ocr_pages=2,
            average_confidence=0.92,
        )
        
        assert result.success
        assert result.documents_found == 3
        assert result.processing_time == 5.23
        assert result.page_classification[DocumentType.EMAIL] == 2
    
    def test_pdf_processing_request(self):
        """Test PDFProcessingRequest model."""
        request = PDFProcessingRequest(
            file_path=Path("/tmp/test.pdf"),
            split_documents=True,
            perform_ocr=True,
            extract_metadata=True,
            classify_documents=True,
            ocr_language="eng",
            min_confidence=0.7,
            max_pages=100,
        )
        
        assert request.file_path == Path("/tmp/test.pdf")
        assert request.split_documents
        assert request.ocr_language == "eng"
        assert request.min_confidence == 0.7