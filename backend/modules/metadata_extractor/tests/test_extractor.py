"""Tests for the main metadata extractor."""

import pytest
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from modules.document_processor.models import Document, DocumentPage, DocumentType, DocumentMetadata
from ..extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test metadata extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_rfi_document(self):
        """Create a sample RFI document."""
        text = """
        REQUEST FOR INFORMATION
        
        Date: May 15, 2024
        RFI #: 123
        Project: Lincoln High School Gymnasium Renovation
        Project No: 2024-LHS-001
        
        To: ABC Construction Company
        Attention: John Smith, Project Manager
        Email: jsmith@abcconstruction.com
        
        From: XYZ School District
        Facilities Department
        Contact: Jane Doe
        Email: jdoe@xyzschools.edu
        
        Subject: Concrete Specifications for Foundation
        
        We are requesting clarification on the concrete specifications for the gymnasium
        foundation. The plans show 3000 PSI concrete, but the specifications indicate
        4000 PSI. Please confirm which is correct.
        
        This RFI requires a response by May 22, 2024 to avoid delays.
        
        Potential cost impact: $15,000.00
        """
        
        return Document(
            id=str(uuid4()),
            source_pdf_id="test-pdf-001",
            source_pdf_path=Path("/test/rfi.pdf"),
            type=DocumentType.RFI,
            pages=[DocumentPage(page_number=1, text=text)],
            page_range=(1, 1),
            text=text,
            metadata=DocumentMetadata()
        )
    
    @pytest.fixture
    def sample_change_order_document(self):
        """Create a sample change order document."""
        text = """
        CHANGE ORDER
        
        Change Order No: 007
        Date: June 1, 2024
        Contract #: C-2024-100
        
        Owner: Orange County School District
        Contractor: Metro Builders Inc.
        Architect: Design Associates LLC
        
        Project: Riverside Elementary School Addition
        
        Description of Change:
        Additional structural reinforcement required due to soil conditions.
        
        Original Contract Amount: $1,234,567.00
        Previous Change Orders: $50,000.00
        This Change Order: $75,890.50
        New Contract Total: $1,360,457.50
        
        Time Extension: 15 calendar days
        
        Approved by:
        Mike Johnson, Project Manager
        mjohnson@metrobuilders.com
        (714) 555-1234
        """
        
        return Document(
            id=str(uuid4()),
            source_pdf_id="test-pdf-002",
            source_pdf_path=Path("/test/co.pdf"),
            type=DocumentType.CHANGE_ORDER,
            pages=[DocumentPage(page_number=1, text=text)],
            page_range=(1, 1),
            text=text,
            metadata=DocumentMetadata()
        )
    
    def test_extract_metadata_from_rfi(self, extractor, sample_rfi_document):
        """Test metadata extraction from RFI document."""
        metadata = extractor.extract_metadata(sample_rfi_document)
        
        # Check title
        assert metadata.title == "RFI #123"
        assert metadata.subject == "RFI #123"
        
        # Check dates
        assert metadata.date == datetime(2024, 5, 15)
        
        # Check reference numbers
        assert "RFI 123" in metadata.reference_numbers
        assert "Project #2024-LHS-001" in metadata.reference_numbers
        
        # Check amounts
        assert 15000.0 in metadata.amounts
        
        # Check parties
        party_names = [p.name for p in metadata.parties]
        assert any("ABC Construction" in name for name in party_names)
        assert any("XYZ School District" in name for name in party_names)
        
        # Check emails
        emails = [p.email for p in metadata.parties if p.email]
        assert "jsmith@abcconstruction.com" in emails
        assert "jdoe@xyzschools.edu" in emails
        
        # Check project info
        assert metadata.project_number == "2024-LHS-001"
        assert "Lincoln High School" in metadata.project_name
        
        # Check keywords
        assert any(kw in ["delay", "impact", "specification"] for kw in metadata.keywords)
    
    def test_extract_metadata_from_change_order(self, extractor, sample_change_order_document):
        """Test metadata extraction from change order document."""
        metadata = extractor.extract_metadata(sample_change_order_document)
        
        # Check title
        assert metadata.title == "Change Order #007"
        
        # Check dates
        assert metadata.date == datetime(2024, 6, 1)
        
        # Check reference numbers
        assert "Change Order #007" in metadata.reference_numbers
        assert "Contract #C-2024-100" in metadata.reference_numbers
        
        # Check amounts
        amounts = sorted(metadata.amounts, reverse=True)
        assert 1360457.50 in amounts
        assert 1234567.00 in amounts
        assert 75890.50 in amounts
        assert 50000.00 in amounts
        
        # Check parties
        party_names = [p.name for p in metadata.parties]
        assert any("Orange County School District" in name for name in party_names)
        assert any("Metro Builders" in name for name in party_names)
        assert any("Design Associates" in name for name in party_names)
        
        # Check phone
        phones = [p.phone for p in metadata.parties if p.phone]
        assert any("714" in phone for phone in phones)
    
    def test_extract_keywords(self, extractor, sample_rfi_document):
        """Test keyword extraction."""
        metadata = extractor.extract_metadata(sample_rfi_document)
        
        # Modify document to include construction keywords
        sample_rfi_document.text += "\nThis may cause a delay in the schedule due to design error."
        metadata = extractor.extract_metadata(sample_rfi_document)
        
        assert "delay" in metadata.keywords
        assert "schedule" in metadata.keywords
        assert "design error" in metadata.keywords
    
    def test_party_normalization(self, extractor):
        """Test that party names are normalized."""
        text = """
        From: ABC Construction Inc.
        Also known as: ABC Const. Inc
        Parent company: ABC Construction Incorporated
        """
        
        doc = Document(
            id=str(uuid4()),
            source_pdf_id="test",
            source_pdf_path=Path("/test/test.pdf"),
            type=DocumentType.LETTER,
            pages=[DocumentPage(page_number=1, text=text)],
            page_range=(1, 1),
            text=text
        )
        
        metadata = extractor.extract_metadata(doc)
        
        # Should normalize to single party
        abc_parties = [p for p in metadata.parties if "ABC" in p.name]
        assert len(abc_parties) == 1
        assert "ABC Construction" in abc_parties[0].name
    
    def test_determine_document_date(self, extractor):
        """Test document date determination."""
        text = """
        Various dates mentioned:
        Contract signed: January 1, 2024
        Meeting held: February 15, 2024
        
        Date: March 10, 2024
        
        Future deadline: December 31, 2024
        """
        
        doc = Document(
            id=str(uuid4()),
            source_pdf_id="test",
            source_pdf_path=Path("/test/test.pdf"),
            type=DocumentType.LETTER,
            pages=[DocumentPage(page_number=1, text=text)],
            page_range=(1, 1),
            text=text
        )
        
        metadata = extractor.extract_metadata(doc)
        
        # Should pick the date with "Date:" indicator
        assert metadata.date == datetime(2024, 3, 10)
    
    def test_empty_document(self, extractor):
        """Test extraction from document with minimal content."""
        doc = Document(
            id=str(uuid4()),
            source_pdf_id="test",
            source_pdf_path=Path("/test/empty.pdf"),
            type=DocumentType.UNKNOWN,
            pages=[DocumentPage(page_number=1, text="")],
            page_range=(1, 1),
            text=""
        )
        
        metadata = extractor.extract_metadata(doc)
        
        assert metadata.title == "Unknown"
        assert metadata.date is None
        assert len(metadata.parties) == 0
        assert len(metadata.amounts) == 0
        assert len(metadata.reference_numbers) == 0