#!/usr/bin/env python3
"""Integration test script showing document processor + storage working together."""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.models import (
    Document,
    DocumentPage,
    DocumentType,
    PDFProcessingRequest,
)
from modules.storage.models import StoredDocument, DocumentMetadata
from modules.storage.sqlite_handler import SQLiteHandler


def test_integration():
    """Test document processor to storage integration."""
    print("\n🔗 Testing Document Processor + Storage Integration")
    print("=" * 60)
    
    # 1. Simulate document processor output
    print("\n1️⃣ Creating processed document...")
    doc = Document(
        id="doc-integration-001",
        source_pdf_id="pdf-integration-001", 
        source_pdf_path=Path("/test/construction_project.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text="REQUEST FOR INFORMATION\nRFI #123\nDate: January 15, 2024\n\nTo: ABC Construction\nFrom: XYZ School District\n\nRe: Concrete specifications for gymnasium foundation",
                is_scanned=False,
                has_tables=False,
                has_images=False,
                confidence=1.0,
            ),
            DocumentPage(
                page_number=2,
                text="The specified concrete strength of 4000 PSI appears insufficient for the gymnasium foundation given the soil conditions...",
                is_scanned=False,
                has_tables=True,
                has_images=False,
                confidence=1.0,
            ),
        ],
        page_range=(1, 2),
        text="REQUEST FOR INFORMATION\nRFI #123\n...",  # Combined text
        type=DocumentType.RFI,
        title="RFI #123 - Concrete Specifications",
        classification_confidence=0.95,
    )
    print(f"✅ Created document: {doc.title}")
    
    # 2. Convert to storage format
    print("\n2️⃣ Converting to storage format...")
    metadata = DocumentMetadata(
        dates=[datetime(2024, 1, 15)],
        parties=["ABC Construction", "XYZ School District"],
        reference_numbers={"RFI": ["123"]},
        amounts=[],  # No amounts in this RFI
    )
    
    stored_doc = StoredDocument(
        id=doc.id,
        source_pdf_id=doc.source_pdf_id,
        source_pdf_path=doc.source_pdf_path,
        type=doc.type,
        page_range=doc.page_range,
        page_count=len(doc.pages),
        title=doc.title,
        text=doc.text,
        metadata=metadata,
        classification_confidence=doc.classification_confidence,
    )
    print("✅ Converted to storage format")
    
    # 3. Store in database
    print("\n3️⃣ Storing in database...")
    db_path = Path("./test_integration.db")
    if db_path.exists():
        db_path.unlink()
    
    try:
        db = SQLiteHandler(db_path)
        
        # Convert pages
        stored_pages = []
        for page in doc.pages:
            from modules.storage.models import StoredPage
            stored_page = StoredPage(
                id=f"page-{doc.id}-{page.page_number}",
                document_id=doc.id,
                page_number=page.page_number,
                text=page.text,
                is_scanned=page.is_scanned,
                has_tables=page.has_tables,
                has_images=page.has_images,
                ocr_confidence=page.confidence,
            )
            stored_pages.append(stored_page)
        
        # Save to database
        saved_id = db.save_document(stored_doc, stored_pages)
        print(f"✅ Saved to database with ID: {saved_id}")
        
        # 4. Test retrieval and search
        print("\n4️⃣ Testing retrieval and search...")
        
        # Get by ID
        retrieved = db.get_document(saved_id)
        if retrieved:
            print(f"✅ Retrieved document: {retrieved.title}")
            print(f"   Parties: {', '.join(retrieved.metadata.parties)}")
            print(f"   Reference: RFI #{retrieved.metadata.reference_numbers['RFI'][0]}")
        
        # Search by text
        from modules.storage.models import SearchFilter
        filter = SearchFilter(query="concrete gymnasium")
        results = db.search_documents(filter)
        print(f"\n✅ Search for 'concrete gymnasium' found {results.total_count} documents")
        
        # Search by type
        filter = SearchFilter(document_types=[DocumentType.RFI])
        results = db.search_documents(filter)
        print(f"✅ Found {results.total_count} RFI documents")
        
        # 5. Demonstrate full pipeline
        print("\n5️⃣ Full Pipeline Summary:")
        print("   📄 PDF → Document Processor → Structured Document")
        print("   💾 Structured Document → Storage Module → SQLite Database")
        print("   🔍 Database → Search → Results")
        
        print("\n✅ Integration test successful!")
        
    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()
            print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    test_integration()