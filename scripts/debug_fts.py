#!/usr/bin/env python3
"""Debug FTS trigger issue."""

import sqlite3
import json
from pathlib import Path
import tempfile

# Create test database
with tempfile.TemporaryDirectory() as temp_dir:
    db_path = Path(temp_dir) / "test.db"
    conn = sqlite3.connect(str(db_path))
    
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
            summary TEXT,
            metadata_json TEXT NOT NULL,
            embedding_json TEXT,
            key_facts_json TEXT,
            classification_confidence REAL DEFAULT 0.0,
            status TEXT NOT NULL,
            storage_path TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            indexed_at TIMESTAMP
        )
    """)
    
    # Create FTS5 virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            id UNINDEXED,
            title,
            text,
            parties,
            reference_numbers,
            content=documents,
            content_rowid=rowid,
            tokenize='porter unicode61'
        )
    """)
    
    # Test insert without trigger first
    metadata = {
        "dates": ["2024-01-15"],
        "parties": ["ABC Construction", "XYZ School District"],
        "reference_numbers": {"RFI": ["123"]},
        "amounts": []
    }
    
    conn.execute("""
        INSERT INTO documents (
            id, source_pdf_id, source_pdf_path, type,
            page_start, page_end, page_count,
            title, text, metadata_json, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "test-001",
        "pdf-001",
        "/test.pdf",
        "rfi",
        1, 1, 1,
        "Test RFI",
        "Test content",
        json.dumps(metadata),
        "stored",
        "2024-01-01T00:00:00",
        "2024-01-01T00:00:00"
    ))
    
    # Now test FTS insert
    try:
        conn.execute("""
            INSERT INTO documents_fts(rowid, id, title, text, parties, reference_numbers)
            SELECT rowid, id, title, text,
                   json_extract(metadata_json, '$.parties'),
                   json_extract(metadata_json, '$.reference_numbers')
            FROM documents WHERE id = 'test-001'
        """)
        print("✅ Manual FTS insert successful")
        
        # Test search
        cursor = conn.execute("""
            SELECT * FROM documents_fts WHERE documents_fts MATCH 'construction'
        """)
        results = cursor.fetchall()
        print(f"✅ FTS search found {len(results)} results")
        
    except Exception as e:
        print(f"❌ FTS insert failed: {e}")
        
    # Now test with trigger
    print("\nCreating trigger...")
    try:
        conn.execute("""
            CREATE TRIGGER documents_fts_insert
            AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, id, title, text, parties, reference_numbers)
                VALUES (
                    new.rowid,
                    new.id,
                    new.title,
                    new.text,
                    json_extract(new.metadata_json, '$.parties'),
                    json_extract(new.metadata_json, '$.reference_numbers')
                );
            END
        """)
        print("✅ Trigger created successfully")
        
        # Test insert with trigger
        conn.execute("""
            INSERT INTO documents (
                id, source_pdf_id, source_pdf_path, type,
                page_start, page_end, page_count,
                title, text, metadata_json, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "test-002",
            "pdf-002",
            "/test2.pdf",
            "change_order",
            1, 2, 2,
            "Test CO",
            "Change order content",
            json.dumps(metadata),
            "stored",
            "2024-01-02T00:00:00",
            "2024-01-02T00:00:00"
        ))
        print("✅ Insert with trigger successful")
        
    except Exception as e:
        print(f"❌ Trigger operation failed: {e}")
        print("\nTrying alternative trigger syntax...")
        
        # Drop the bad trigger
        conn.execute("DROP TRIGGER IF EXISTS documents_fts_insert")
        
        # Try simpler trigger
        conn.execute("""
            CREATE TRIGGER documents_fts_insert
            AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(id, title, text)
                VALUES (new.id, new.title, new.text);
            END
        """)
        print("✅ Simplified trigger created")
        
    conn.close()
    print("\n✅ All tests completed")