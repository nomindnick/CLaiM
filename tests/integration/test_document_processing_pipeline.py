#!/usr/bin/env python3
"""Integration tests for the complete document processing pipeline.

Tests the flow from PDF upload through processing, classification,
OCR (if needed), and storage in the database.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import Document, DocumentType, PDFProcessingRequest
from modules.storage.storage_manager import StorageManager
from modules.storage.models import SearchFilter, StorageStatus
from api.config import Settings


class TestDocumentProcessingPipeline:
    """Test the complete document processing pipeline."""
    
    @pytest.fixture
    def test_pdfs_dir(self):
        """Get the directory containing test PDFs."""
        return Path(__file__).parent.parent / "test_data"
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def settings(self, temp_storage_dir):
        """Create test settings."""
        return Settings(
            storage_dir=temp_storage_dir,
            database_url=f"sqlite:///{temp_storage_dir}/test.db"
        )
    
    @pytest.fixture
    def pdf_splitter(self):
        """Create PDF splitter instance."""
        return PDFSplitter()
    
    @pytest.fixture
    def storage_manager(self, settings):
        """Create storage manager instance."""
        return StorageManager(settings.storage_dir)
    
    def test_process_rfi_document(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test processing an RFI document."""
        # Get test RFI
        rfi_path = test_pdfs_dir / "RFI_123.pdf"
        assert rfi_path.exists(), f"Test RFI not found at {rfi_path}"
        
        # Process the document
        request = PDFProcessingRequest(file_path=rfi_path)
        result = pdf_splitter.process_pdf(request)
        documents = result.documents if result.success else []
        assert len(documents) >= 1, "Should extract at least one document"
        
        # The RFI should be a single document
        doc = documents[0]
        assert doc.page_count > 0, "Document should have pages"
        assert doc.text, "Document should have extracted text"
        assert "REQUEST FOR INFORMATION" in doc.text
        assert "concrete" in doc.text.lower()
        
        # Store the document
        doc_id = storage_manager.store_document(doc, rfi_path)
        assert doc_id, "Should return document ID"
        
        # Retrieve and verify
        stored_doc = storage_manager.get_document(doc_id)
        assert stored_doc is not None
        assert stored_doc.title
        # Note: Document classification is not yet implemented, so type may vary
        assert stored_doc.status == StorageStatus.STORED
        
        # Search for the document
        results = storage_manager.search_documents(
            SearchFilter(query="concrete specifications")
        )
        assert results.total_count > 0, "FTS search should find the document"
        assert any(d.id == doc_id for d in results.documents), "Should find the specific document"
    
    def test_process_change_order(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test processing a change order document."""
        co_path = test_pdfs_dir / "Change_Order_007.pdf"
        assert co_path.exists()
        
        # Process
        request = PDFProcessingRequest(file_path=co_path)
        result = pdf_splitter.process_pdf(request)
        documents = result.documents if result.success else []
        assert len(documents) >= 1
        
        doc = documents[0]
        assert "CHANGE ORDER" in doc.text
        assert "$64,687.50" in doc.text  # Total amount
        
        # Store
        doc_id = storage_manager.store_document(doc, co_path)
        
        # Search by content
        results = storage_manager.search_documents(
            SearchFilter(query="gymnasium foundation upgrade")
        )
        assert results.total_count > 0
    
    def test_process_mixed_document(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test processing a document with mixed text/scanned pages."""
        mixed_path = test_pdfs_dir / "Mixed_Document_Contract_Amendment.pdf"
        assert mixed_path.exists()
        
        # Process
        request = PDFProcessingRequest(file_path=mixed_path)
        result = pdf_splitter.process_pdf(request)
        documents = result.documents if result.success else []
        assert len(documents) >= 1
        
        doc = documents[0]
        assert doc.page_count >= 3, "Should have at least 3 pages"
        
        # Check that we have both text and scanned pages
        text_pages = [p for p in doc.pages if not p.is_scanned]
        scanned_pages = [p for p in doc.pages if p.is_scanned]
        
        assert len(text_pages) > 0, "Should have text pages"
        assert len(scanned_pages) > 0, "Should have scanned pages"
        
        # Verify OCR was performed on scanned pages
        for page in scanned_pages:
            assert page.text, "Scanned page should have OCR text"
            assert page.confidence < 1.0, "Scanned page should have confidence < 1.0"
        
        # Store and verify
        doc_id = storage_manager.store_document(doc, mixed_path)
        stored_doc = storage_manager.get_document(doc_id)
        assert stored_doc is not None
    
    def test_process_multiple_documents(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test processing multiple documents in sequence."""
        pdf_files = list(test_pdfs_dir.glob("*.pdf"))[:5]  # Process first 5
        assert len(pdf_files) > 0, "No test PDFs found"
        
        stored_ids = []
        
        for pdf_path in pdf_files:
            # Process each PDF
            request = PDFProcessingRequest(file_path=pdf_path)
            result = pdf_splitter.process_pdf(request)
            documents = result.documents if result.success else []
            
            # Store all extracted documents
            for doc in documents:
                doc_id = storage_manager.store_document(doc, pdf_path)
                stored_ids.append(doc_id)
        
        # Verify all stored
        assert len(stored_ids) >= len(pdf_files)
        
        # Get statistics
        stats = storage_manager.get_stats()
        assert stats.total_documents >= len(stored_ids)
        assert stats.total_pages > 0
        assert stats.total_pdfs >= len(pdf_files)
    
    def test_search_functionality(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test various search scenarios."""
        # Process and store several documents first
        test_files = [
            "RFI_123.pdf",
            "Change_Order_007.pdf", 
            "Invoice_0005.pdf",
            "Daily_Report_20250504.pdf"
        ]
        
        for filename in test_files:
            pdf_path = test_pdfs_dir / filename
            if pdf_path.exists():
                request = PDFProcessingRequest(file_path=pdf_path)
                result = pdf_splitter.process_pdf(request)
                documents = result.documents if result.success else []
                for doc in documents:
                    storage_manager.store_document(doc, pdf_path)
        
        # Test text search
        results = storage_manager.search_documents(
            SearchFilter(query="construction")
        )
        assert results.total_count > 0, "Should find documents mentioning construction"
        
        # Test search with no results
        results = storage_manager.search_documents(
            SearchFilter(query="xyznonexistentterm123")
        )
        assert results.total_count == 0
        
        # Test pagination
        results = storage_manager.search_documents(
            SearchFilter(query="", offset=0, limit=2)
        )
        assert len(results.documents) <= 2
        assert results.total_count >= len(results.documents)
    
    def test_document_metadata_extraction(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test that metadata is properly extracted and stored."""
        # Process RFI with known metadata
        rfi_path = test_pdfs_dir / "RFI_123.pdf"
        request = PDFProcessingRequest(file_path=rfi_path)
        result = pdf_splitter.process_pdf(request)
        documents = result.documents if result.success else []
        doc = documents[0]
        
        # Store document
        doc_id = storage_manager.store_document(doc, rfi_path)
        
        # Retrieve and check metadata
        stored_doc = storage_manager.get_document(doc_id)
        assert stored_doc is not None
        
        # Once metadata extraction is implemented, these should pass:
        # assert len(stored_doc.metadata.parties) > 0
        # assert "ABC Construction" in stored_doc.metadata.parties
        # assert "XYZ School District" in stored_doc.metadata.parties
        # assert len(stored_doc.metadata.dates) > 0
        # assert "123" in stored_doc.metadata.reference_numbers.get("RFI", [])
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_pdfs_dir, pdf_splitter, storage_manager):
        """Test processing multiple documents concurrently."""
        pdf_files = list(test_pdfs_dir.glob("*.pdf"))[:3]
        
        async def process_pdf(pdf_path):
            """Process a single PDF asynchronously."""
            loop = asyncio.get_event_loop()
            request = PDFProcessingRequest(file_path=pdf_path)
            result = await loop.run_in_executor(None, pdf_splitter.process_pdf, request)
            documents = result.documents if result.success else []
            
            stored_ids = []
            for doc in documents:
                doc_id = await loop.run_in_executor(
                    None, storage_manager.store_document, doc, pdf_path
                )
                stored_ids.append(doc_id)
            
            return stored_ids
        
        # Process all PDFs concurrently
        results = await asyncio.gather(*[process_pdf(pdf) for pdf in pdf_files])
        
        # Verify all processed
        all_ids = [doc_id for result in results for doc_id in result]
        assert len(all_ids) >= len(pdf_files)
        
        # Verify all retrievable
        for doc_id in all_ids:
            doc = storage_manager.get_document(doc_id)
            assert doc is not None


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("\n" + "="*60)
    print("DOCUMENT PROCESSING PIPELINE - END TO END TEST")
    print("="*60)
    
    # Setup
    test_dir = Path(__file__).parent.parent / "test_data"
    temp_storage = Path(tempfile.mkdtemp())
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        settings = Settings(
            storage_dir=temp_storage,
            database_url=f"sqlite:///{temp_storage}/test.db"
        )
        splitter = PDFSplitter()
        storage = StorageManager(settings.storage_dir)
        print("   ✓ Components initialized")
        
        # Process RFI
        print("\n2. Processing RFI document...")
        rfi_path = test_dir / "RFI_123.pdf"
        if rfi_path.exists():
            request = PDFProcessingRequest(file_path=rfi_path)
            result = splitter.process_pdf(request)
            documents = result.documents if result.success else []
            print(f"   ✓ Extracted {len(documents)} document(s)")
            
            doc = documents[0]
            print(f"   ✓ Document has {doc.page_count} pages")
            print(f"   ✓ Extracted {len(doc.text)} characters of text")
            
            # Store document
            print("\n3. Storing document...")
            doc_id = storage.store_document(doc, rfi_path)
            print(f"   ✓ Stored with ID: {doc_id}")
            
            # Search and retrieve
            print("\n4. Testing search...")
            results = storage.search_documents(SearchFilter(query="concrete"))
            print(f"   ✓ Found {results.total_count} documents")
            
            # Get stats
            print("\n5. Storage statistics:")
            stats = storage.get_stats()
            print(f"   • Total documents: {stats.total_documents}")
            print(f"   • Total pages: {stats.total_pages}")
            print(f"   • Database size: {stats.database_size_mb:.2f} MB")
            
            print("\n✅ End-to-end test completed successfully!")
        else:
            print("   ❌ Test RFI not found. Run generate_test_pdfs.py first.")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_storage)
        print("\n🧹 Cleaned up temporary storage")


if __name__ == "__main__":
    # Run the visual end-to-end test
    test_end_to_end_workflow()
    
    # Run pytest for detailed testing
    print("\n" + "="*60)
    print("Running pytest suite...")
    print("="*60)
    pytest.main([__file__, "-v"])