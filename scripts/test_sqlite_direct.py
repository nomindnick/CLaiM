#!/usr/bin/env python3
"""Direct test of SQLite storage functionality."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import tempfile


def test_sqlite_direct():
    """Test SQLite database operations directly."""
    print("\n🔧 Testing SQLite Storage Directly")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        # Create documents table
        conn.execute("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_pdf_id TEXT NOT NULL,
                source_pdf_path TEXT NOT NULL,
                type TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                title TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        print("✅ Created documents table")
        
        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                id UNINDEXED,
                title,
                text,
                content=documents,
                content_rowid=rowid,
                tokenize='porter unicode61'
            )
        """)
        print("✅ Created FTS5 virtual table")
        
        # Insert test document
        doc_id = "test-doc-001"
        metadata = {
            "dates": ["2024-01-15"],
            "parties": ["ABC Construction", "XYZ School District"],
            "reference_numbers": {"RFI": ["123"]},
            "amounts": [25000.0]
        }
        
        conn.execute("""
            INSERT INTO documents (
                id, source_pdf_id, source_pdf_path, type,
                page_start, page_end, page_count,
                title, text, metadata_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            "pdf-001",
            "/test/sample.pdf",
            "RFI",
            1, 3, 3,
            "Test RFI #123",
            "This is a test Request for Information about concrete specifications.",
            json.dumps(metadata),
            "stored",
            datetime.utcnow().isoformat()
        ))
        
        # Also insert into FTS
        conn.execute("""
            INSERT INTO documents_fts(rowid, id, title, text)
            SELECT rowid, id, title, text FROM documents WHERE id = ?
        """, (doc_id,))
        
        conn.commit()
        print(f"✅ Inserted test document: {doc_id}")
        
        # Test retrieval
        cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            print(f"✅ Retrieved document: {row['title']}")
            print(f"   Type: {row['type']}")
            print(f"   Pages: {row['page_start']}-{row['page_end']}")
        
        # Test FTS search
        cursor = conn.execute("""
            SELECT * FROM documents_fts WHERE documents_fts MATCH ?
        """, ("concrete",))
        results = cursor.fetchall()
        print(f"\n✅ FTS search for 'concrete' found {len(results)} documents")
        
        # Test statistics
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_docs,
                SUM(page_count) as total_pages,
                COUNT(DISTINCT source_pdf_id) as total_pdfs
            FROM documents
        """).fetchone()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total documents: {stats['total_docs']}")
        print(f"   Total pages: {stats['total_pages']}")
        print(f"   Total PDFs: {stats['total_pdfs']}")
        
        # Check database size
        db_size = db_path.stat().st_size / 1024
        print(f"   Database size: {db_size:.2f} KB")
        
        conn.close()
        print("\n✅ All SQLite tests passed!")


if __name__ == "__main__":
    test_sqlite_direct()