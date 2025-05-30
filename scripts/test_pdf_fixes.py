#!/usr/bin/env python3
"""Test script to verify PDF splitting and viewing fixes."""

import os
import sys
import sqlite3
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz  # PyMuPDF

def main():
    print("=== Testing PDF Splitting and Viewing Fixes ===\n")
    
    # 1. Check database entries vs actual PDFs
    print("1. Checking database entries vs actual extracted PDFs:")
    db_path = Path("storage/database/documents.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, page_count, page_start, page_end, storage_path 
        FROM documents 
        ORDER BY created_at DESC LIMIT 10
    """)
    
    total_db_pages = 0
    total_actual_pages = 0
    
    for row in cursor.fetchall():
        doc_id, title, db_page_count, page_start, page_end, storage_path = row
        
        # Calculate expected page count from range
        expected_page_count = page_end - page_start + 1
        
        # Check extracted PDF
        extracted_path = Path("storage/extracted") / f"{doc_id}.pdf"
        
        if extracted_path.exists():
            pdf = fitz.open(str(extracted_path))
            actual_page_count = pdf.page_count
            pdf.close()
            
            total_db_pages += db_page_count
            total_actual_pages += actual_page_count
            
            status = "✅" if db_page_count == actual_page_count == expected_page_count else "❌"
            print(f"{status} {title[:50]}...")
            print(f"   DB pages: {db_page_count}, Actual PDF: {actual_page_count}, Expected: {expected_page_count}")
            print(f"   Page range: {page_start}-{page_end}")
            print(f"   Storage path: {storage_path}")
            print(f"   Extracted exists: {extracted_path.exists()}")
            
            if db_page_count != actual_page_count:
                print(f"   ⚠️  Mismatch: DB shows {db_page_count} but PDF has {actual_page_count}")
        else:
            print(f"❌ {title[:50]}... - Extracted PDF not found")
        print()
    
    conn.close()
    
    print(f"Summary: DB total pages: {total_db_pages}, Actual PDF pages: {total_actual_pages}")
    
    # 2. Test original PDF page count
    print("\n2. Checking original test PDF:")
    test_pdf = Path("tests/Test_PDF_Set_2.pdf")
    if test_pdf.exists():
        pdf = fitz.open(str(test_pdf))
        original_pages = pdf.page_count
        pdf.close()
        print(f"Original PDF pages: {original_pages}")
        print(f"Total extracted pages: {total_actual_pages}")
        print(f"Coverage: {total_actual_pages}/{original_pages} = {total_actual_pages/original_pages*100:.1f}%")
        
        if total_actual_pages < original_pages:
            missing_pages = original_pages - total_actual_pages
            print(f"⚠️  {missing_pages} pages were not included in any extracted document")
    else:
        print("❌ Test PDF not found")
    
    # 3. Test path resolution for viewing endpoint
    print("\n3. Testing PDF viewing path resolution:")
    extracted_dir = Path("storage/extracted")
    
    for pdf_file in extracted_dir.glob("*.pdf"):
        doc_id = pdf_file.stem
        print(f"Document {doc_id}:")
        print(f"  File exists: {pdf_file.exists()}")
        print(f"  Absolute path: {pdf_file.resolve()}")
        print(f"  Size: {pdf_file.stat().st_size} bytes")

if __name__ == "__main__":
    main()