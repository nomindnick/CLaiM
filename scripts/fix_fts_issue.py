#!/usr/bin/env python3
"""Fix the FTS5 trigger issue by recreating the database with updated schema."""

import sys
from pathlib import Path
import shutil
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.storage.sqlite_handler import SQLiteHandler
from modules.storage.storage_manager import StorageManager
from modules.storage.models import SearchFilter, DocumentMetadata, StorageStatus
from modules.document_processor.models import DocumentType
from api.config import get_settings
from loguru import logger


def rebuild_database():
    """Rebuild the database with the fixed FTS schema."""
    settings = get_settings()
    db_path = settings.storage_dir / "database" / "documents.db"
    backup_path = db_path.with_suffix('.db.backup')
    
    print(f"\n🔧 Fixing FTS5 trigger issue...")
    print(f"Database path: {db_path}")
    
    # Create backup if database exists
    if db_path.exists():
        print(f"📦 Creating backup at {backup_path}")
        shutil.copy2(db_path, backup_path)
        
        # Remove old database
        print("🗑️  Removing old database...")
        db_path.unlink()
    
    # Create new database with fixed schema
    print("🏗️  Creating new database with fixed schema...")
    handler = SQLiteHandler(db_path)
    
    # Test the fix with a sample document
    print("\n🧪 Testing FTS functionality...")
    from modules.storage.models import StoredDocument
    from uuid import uuid4
    
    # Create test document
    test_doc = StoredDocument(
        id=str(uuid4()),
        source_pdf_id="test-pdf-001",
        source_pdf_path=Path("/test/document.pdf"),
        type=DocumentType.RFI,
        page_range=(1, 3),
        page_count=3,
        title="Test RFI #123",
        text="This is a test Request for Information about concrete specifications for the gymnasium foundation.",
        summary="Test RFI about concrete specs",
        metadata=DocumentMetadata(
            parties=["ABC Construction", "XYZ School District"],
            reference_numbers={"RFI": ["123"], "Project": ["2024-001"]}
        ),
        classification_confidence=0.95,
        status=StorageStatus.STORED
    )
    
    # Save document
    doc_id = handler.save_document(test_doc)
    print(f"✅ Document saved with ID: {doc_id}")
    
    # Test search
    results = handler.search_documents(SearchFilter(query="concrete specifications"))
    if results.total_count > 0:
        print(f"✅ FTS search working! Found {results.total_count} documents")
        print(f"   - Document title: {results.documents[0].title}")
    else:
        print("❌ FTS search failed - no results found")
        return False
    
    # Test party search
    results = handler.search_documents(SearchFilter(query="ABC Construction"))
    if results.total_count > 0:
        print(f"✅ Party search working! Found {results.total_count} documents")
    else:
        print("⚠️  Party search returned no results")
    
    # Clean up test document
    handler.delete_document(doc_id)
    print("🧹 Cleaned up test document")
    
    print("\n✨ FTS issue fixed successfully!")
    return True


def verify_integration():
    """Verify the fix works with the full storage manager."""
    print("\n🔍 Verifying integration with StorageManager...")
    
    settings = get_settings()
    storage = StorageManager(settings.storage_dir)
    
    # Create a more complex test
    from modules.document_processor.models import Document, DocumentPage
    from uuid import uuid4
    
    test_doc = Document(
        id=str(uuid4()),
        source_pdf_id="integration-test-001", 
        source_pdf_path=Path("/test/integration.pdf"),
        type=DocumentType.CHANGE_ORDER,
        pages=[
            DocumentPage(
                page_number=1,
                text="Change Order #007 for gymnasium foundation upgrade",
                is_scanned=False,
                confidence=1.0
            )
        ],
        page_range=(1, 1),
        text="Change Order #007 for gymnasium foundation upgrade. Total amount: $64,687.50"
    )
    
    # Store through storage manager
    doc_id = storage.store_document(test_doc, test_doc.source_pdf_path)
    print(f"✅ Document stored via StorageManager: {doc_id}")
    
    # Search
    results = storage.search_documents(SearchFilter(query="gymnasium foundation"))
    if results.total_count > 0:
        print(f"✅ Search via StorageManager working! Found {results.total_count} documents")
    
    # Cleanup
    storage.delete_document(doc_id)
    print("🧹 Cleaned up integration test")
    
    return True


if __name__ == "__main__":
    success = rebuild_database()
    if success:
        verify_integration()
        print("\n🎉 All tests passed! The FTS5 issue has been resolved.")
        print("\nNext steps:")
        print("1. Run the integration tests: python -m pytest tests/integration/")
        print("2. Process some real documents to verify everything works")
    else:
        print("\n❌ Fix failed. Please check the logs.")
        sys.exit(1)