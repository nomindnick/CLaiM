#!/usr/bin/env python3
"""Debug FTS issue in detail."""

import sqlite3
from pathlib import Path

db_path = Path("storage/database/documents.db")

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    
    # List all triggers
    print("Current triggers in database:")
    print("="*60)
    cursor = conn.execute("""
        SELECT name, sql FROM sqlite_master 
        WHERE type = 'trigger' AND name LIKE '%fts%'
    """)
    for row in cursor:
        print(f"Trigger: {row[0]}")
        print(f"SQL: {row[1]}")
        print("-"*60)
    
    # Check FTS table structure
    print("\nFTS table structure:")
    print("="*60)
    cursor = conn.execute("PRAGMA table_info(documents_fts)")
    for row in cursor:
        print(row)
    
    conn.close()
else:
    print(f"Database not found at {db_path}")