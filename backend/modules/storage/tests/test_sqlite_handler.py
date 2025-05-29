"""Tests for SQLite handler."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from backend.modules.document_processor.models import DocumentType
from backend.modules.storage.models import (
    StorageStatus,
    DocumentMetadata,
    StoredDocument,
    StoredPage,
    SearchFilter,
)
from backend.modules.storage.sqlite_handler import SQLiteHandler
from backend.shared.exceptions import StorageError


class TestSQLiteHandler:
    """Test suite for SQLite handler."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            handler = SQLiteHandler(db_path)
            yield handler
    
    def test_init_database(self, temp_db):
        """Test database initialization."""
        # Check that database file exists
        assert temp_db.db_path.exists()
        
        # Check tables exist
        with temp_db._get_connection() as conn:
            tables = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """).fetchall()
            
            table_names = [row[0] for row in tables]
            assert "documents" in table_names
            assert "pages" in table_names
            assert "document_relationships" in table_names
            assert "documents_fts" in table_names
    
    def test_save_document(self, temp_db):
        """Test saving a document."""
        # Create test document
        doc = StoredDocument(
            id="test-doc-123",
            source_pdf_id="pdf-456",
            source_pdf_path=Path("/test/path.pdf"),
            type=DocumentType.RFI,
            page_range=(1, 3),
            page_count=3,
            title="Test RFI Document",
            text="This is test content for RFI #123",
            metadata=DocumentMetadata(
                parties=["ABC Construction", "XYZ District"],
                reference_numbers={"RFI": ["123"]},
                amounts=[10000.0],
            ),
            classification_confidence=0.95,
            status=StorageStatus.STORED,
        )
        
        # Save document
        doc_id = temp_db.save_document(doc)
        assert doc_id == "test-doc-123"
        
        # Verify saved
        saved_doc = temp_db.get_document(doc_id)
        assert saved_doc is not None
        assert saved_doc.title == "Test RFI Document"
        assert saved_doc.type == DocumentType.RFI
        assert saved_doc.classification_confidence == 0.95
        assert "ABC Construction" in saved_doc.metadata.parties
    
    def test_save_document_with_pages(self, temp_db):
        """Test saving document with pages."""
        # Create document
        doc = StoredDocument(
            id="doc-with-pages",
            source_pdf_id="pdf-789",
            source_pdf_path=Path("/test/pages.pdf"),
            type=DocumentType.CONTRACT,
            page_range=(1, 2),
            page_count=2,
            text="Contract content",
        )
        
        # Create pages
        pages = [
            StoredPage(
                id="page-1",
                document_id="doc-with-pages",
                page_number=1,
                text="Page 1 content",
                has_tables=True,
            ),
            StoredPage(
                id="page-2",
                document_id="doc-with-pages",
                page_number=2,
                text="Page 2 OCR content",
                is_scanned=True,
                ocr_confidence=0.85,
            ),
        ]
        
        # Save document with pages
        doc_id = temp_db.save_document(doc, pages)
        
        # Verify pages saved
        with temp_db._get_connection() as conn:
            saved_pages = conn.execute("""
                SELECT * FROM pages WHERE document_id = ?
                ORDER BY page_number
            """, (doc_id,)).fetchall()
            
            assert len(saved_pages) == 2
            assert saved_pages[0]["text"] == "Page 1 content"
            assert saved_pages[1]["is_scanned"] == 1
            assert saved_pages[1]["ocr_confidence"] == 0.85
    
    def test_save_document_with_relationships(self, temp_db):
        """Test saving document with relationships."""
        # Save first document
        doc1 = StoredDocument(
            id="doc-1",
            source_pdf_id="pdf-1",
            source_pdf_path=Path("/test/doc1.pdf"),
            type=DocumentType.EMAIL,
            page_range=(1, 1),
            page_count=1,
            text="Email content",
        )
        temp_db.save_document(doc1)
        
        # Save second document that references first
        doc2 = StoredDocument(
            id="doc-2",
            source_pdf_id="pdf-2",
            source_pdf_path=Path("/test/doc2.pdf"),
            type=DocumentType.RFI,
            page_range=(1, 2),
            page_count=2,
            text="RFI content",
            responds_to=["doc-1"],
            references=["doc-1"],
        )
        temp_db.save_document(doc2)
        
        # Verify relationships
        saved_doc = temp_db.get_document("doc-2")
        assert "doc-1" in saved_doc.responds_to
        assert "doc-1" in saved_doc.references
    
    def test_search_documents_by_text(self, temp_db):
        """Test full-text search."""
        # Save test documents
        docs = [
            StoredDocument(
                id=f"doc-{i}",
                source_pdf_id=f"pdf-{i}",
                source_pdf_path=Path(f"/test/doc{i}.pdf"),
                type=DocumentType.RFI,
                page_range=(1, 1),
                page_count=1,
                title=f"Document {i}",
                text=text,
            )
            for i, text in enumerate([
                "Request for Information about concrete pour schedule",
                "Change order for additional steel reinforcement",
                "Meeting minutes discussing concrete issues",
            ])
        ]
        
        for doc in docs:
            temp_db.save_document(doc)
        
        # Search for "concrete"
        filter = SearchFilter(query="concrete")
        results = temp_db.search_documents(filter)
        
        assert results.total_count == 2
        assert len(results.documents) == 2
        # Both documents mentioning "concrete" should be found
        found_texts = [doc.text for doc in results.documents]
        assert any("concrete pour" in text for text in found_texts)
        assert any("concrete issues" in text for text in found_texts)
    
    def test_search_documents_by_type(self, temp_db):
        """Test searching by document type."""
        # Save documents of different types
        doc_types = [
            (DocumentType.RFI, "RFI content"),
            (DocumentType.EMAIL, "Email content"),
            (DocumentType.INVOICE, "Invoice content"),
            (DocumentType.RFI, "Another RFI"),
        ]
        
        for i, (doc_type, text) in enumerate(doc_types):
            doc = StoredDocument(
                id=f"type-doc-{i}",
                source_pdf_id=f"pdf-{i}",
                source_pdf_path=Path(f"/test/doc{i}.pdf"),
                type=doc_type,
                page_range=(1, 1),
                page_count=1,
                text=text,
            )
            temp_db.save_document(doc)
        
        # Search for RFIs only
        filter = SearchFilter(document_types=[DocumentType.RFI])
        results = temp_db.search_documents(filter)
        
        assert results.total_count == 2
        assert all(doc.type == DocumentType.RFI for doc in results.documents)
    
    def test_search_with_pagination(self, temp_db):
        """Test search pagination."""
        # Save 10 documents
        for i in range(10):
            doc = StoredDocument(
                id=f"page-doc-{i}",
                source_pdf_id=f"pdf-{i}",
                source_pdf_path=Path(f"/test/doc{i}.pdf"),
                type=DocumentType.EMAIL,
                page_range=(1, 1),
                page_count=1,
                text=f"Email number {i}",
            )
            temp_db.save_document(doc)
        
        # Get first page
        filter = SearchFilter(limit=5, offset=0)
        results = temp_db.search_documents(filter)
        
        assert results.total_count == 10
        assert len(results.documents) == 5
        assert results.offset == 0
        assert results.limit == 5
        
        # Get second page
        filter = SearchFilter(limit=5, offset=5)
        results = temp_db.search_documents(filter)
        
        assert len(results.documents) == 5
        assert results.offset == 5
    
    def test_get_document_not_found(self, temp_db):
        """Test getting non-existent document."""
        doc = temp_db.get_document("non-existent-id")
        assert doc is None
    
    def test_delete_document(self, temp_db):
        """Test deleting a document."""
        # Save document
        doc = StoredDocument(
            id="doc-to-delete",
            source_pdf_id="pdf-del",
            source_pdf_path=Path("/test/delete.pdf"),
            type=DocumentType.EMAIL,
            page_range=(1, 1),
            page_count=1,
            text="Delete me",
        )
        temp_db.save_document(doc)
        
        # Verify it exists
        assert temp_db.get_document("doc-to-delete") is not None
        
        # Delete it
        assert temp_db.delete_document("doc-to-delete") is True
        
        # Verify it's gone
        assert temp_db.get_document("doc-to-delete") is None
        
        # Try deleting again
        assert temp_db.delete_document("doc-to-delete") is False
    
    def test_update_document_status(self, temp_db):
        """Test updating document status."""
        # Save document
        doc = StoredDocument(
            id="status-doc",
            source_pdf_id="pdf-status",
            source_pdf_path=Path("/test/status.pdf"),
            type=DocumentType.EMAIL,
            page_range=(1, 1),
            page_count=1,
            text="Status test",
            status=StorageStatus.PENDING,
        )
        temp_db.save_document(doc)
        
        # Update status
        assert temp_db.update_document_status("status-doc", StorageStatus.INDEXED) is True
        
        # Verify updated
        updated_doc = temp_db.get_document("status-doc")
        assert updated_doc.status == StorageStatus.INDEXED
    
    def test_get_stats(self, temp_db):
        """Test getting storage statistics."""
        # Save some documents
        doc_types = [
            DocumentType.RFI,
            DocumentType.RFI,
            DocumentType.EMAIL,
            DocumentType.INVOICE,
        ]
        
        for i, doc_type in enumerate(doc_types):
            doc = StoredDocument(
                id=f"stats-doc-{i}",
                source_pdf_id=f"pdf-{i//2}",  # Two docs per PDF
                source_pdf_path=Path(f"/test/doc{i}.pdf"),
                type=doc_type,
                page_range=(1, 2),
                page_count=2,
                text=f"Document {i} content",
            )
            temp_db.save_document(doc)
        
        # Get stats
        stats = temp_db.get_stats()
        
        assert stats.total_documents == 4
        assert stats.total_pages == 8  # 4 docs × 2 pages
        assert stats.total_pdfs == 2  # Two unique source PDFs
        assert stats.documents_by_type[DocumentType.RFI] == 2
        assert stats.documents_by_type[DocumentType.EMAIL] == 1
        assert stats.documents_by_type[DocumentType.INVOICE] == 1
        assert stats.average_pages_per_document == 2.0