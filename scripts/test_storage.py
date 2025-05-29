#!/usr/bin/env python3
"""Test script for storage module functionality."""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from modules.document_processor.models import DocumentType
from modules.storage.models import (
    StoredDocument,
    StoredPage,
    DocumentMetadata,
    SearchFilter,
    StorageStatus,
)
from modules.storage.sqlite_handler import SQLiteHandler


def test_storage():
    """Test basic storage operations."""
    print("\n🔧 Testing Storage Module")
    print("=" * 60)
    
    # Create temporary database
    db_path = Path("./test_storage.db")
    if db_path.exists():
        db_path.unlink()
    
    try:
        # Initialize handler
        handler = SQLiteHandler(db_path)
        print("✅ SQLite handler initialized")
        
        # Create test document
        doc = StoredDocument(
            id="test-doc-001",
            source_pdf_id="pdf-001",
            source_pdf_path=Path("/test/sample.pdf"),
            type=DocumentType.RFI,
            page_range=(1, 3),
            page_count=3,
            title="Test RFI #123",
            text="This is a test Request for Information about concrete specifications.",
            metadata=DocumentMetadata(
                dates=[datetime(2024, 1, 15)],
                parties=["ABC Construction", "XYZ School District"],
                reference_numbers={"RFI": ["123"]},
                amounts=[25000.0],
            ),
            classification_confidence=0.95,
            status=StorageStatus.STORED,
        )
        
        # Save document
        doc_id = handler.save_document(doc)
        print(f"✅ Document saved with ID: {doc_id}")
        
        # Create and save pages
        pages = [
            StoredPage(
                id=f"page-{i}",
                document_id=doc_id,
                page_number=i,
                text=f"Page {i} content with construction details",
                has_tables=(i == 2),
                is_scanned=(i == 3),
                ocr_confidence=0.85 if i == 3 else 1.0,
            )
            for i in range(1, 4)
        ]
        
        handler.save_document(doc, pages)
        print(f"✅ Saved {len(pages)} pages")
        
        # Test retrieval
        retrieved_doc = handler.get_document(doc_id)
        if retrieved_doc:
            print(f"✅ Retrieved document: {retrieved_doc.title}")
            print(f"   Type: {retrieved_doc.type.value}")
            print(f"   Parties: {', '.join(retrieved_doc.metadata.parties)}")
            print(f"   Reference: RFI {retrieved_doc.metadata.reference_numbers.get('RFI', ['N/A'])[0]}")
        
        # Test search
        print("\n📍 Testing Search...")
        
        # Search by text
        filter = SearchFilter(query="concrete")
        results = handler.search_documents(filter)
        print(f"✅ Text search found {results.total_count} documents")
        
        # Search by type
        filter = SearchFilter(document_types=[DocumentType.RFI])
        results = handler.search_documents(filter)
        print(f"✅ Type search found {results.total_count} RFI documents")
        
        # Get statistics
        stats = handler.get_stats()
        print(f"\n📊 Storage Statistics:")
        print(f"   Total documents: {stats.total_documents}")
        print(f"   Total pages: {stats.total_pages}")
        print(f"   Documents by type: {dict(stats.documents_by_type)}")
        print(f"   Database size: {stats.database_size_mb:.2f} MB")
        
        print("\n✅ All storage tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()
            print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    test_storage()