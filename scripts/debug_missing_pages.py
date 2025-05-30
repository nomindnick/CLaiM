#!/usr/bin/env python3
"""Debug script to find missing pages in PDF splitting."""

import os
import sys
import sqlite3
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz  # PyMuPDF

def main():
    print("=== Debugging Missing Pages ===\n")
    
    # 1. Check what pages are covered by documents
    print("1. Analyzing page coverage from database:")
    db_path = Path("storage/database/documents.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, page_start, page_end, source_pdf_id
        FROM documents 
        ORDER BY page_start ASC
    """)
    
    all_covered_pages = set()
    documents_info = []
    
    for row in cursor.fetchall():
        doc_id, title, page_start, page_end, source_pdf_id = row
        page_range = list(range(page_start, page_end + 1))
        all_covered_pages.update(page_range)
        documents_info.append({
            'id': doc_id,
            'title': title[:50] + "..." if len(title) > 50 else title,
            'pages': page_range,
            'source_pdf_id': source_pdf_id
        })
        print(f"Doc: {title[:50]}...")
        print(f"     Pages {page_start}-{page_end} ({len(page_range)} pages)")
    
    conn.close()
    
    # 2. Find missing pages
    print(f"\n2. Page coverage analysis:")
    print(f"   Pages covered: {sorted(all_covered_pages)}")
    
    # Check original PDF
    test_pdf = Path("tests/Test_PDF_Set_2.pdf")
    if test_pdf.exists():
        pdf = fitz.open(str(test_pdf))
        total_pages = pdf.page_count
        pdf.close()
        
        all_pages = set(range(1, total_pages + 1))
        missing_pages = all_pages - all_covered_pages
        
        print(f"   Total pages in PDF: {total_pages}")
        print(f"   Pages covered: {len(all_covered_pages)}")
        print(f"   Missing pages: {sorted(missing_pages)} ({len(missing_pages)} pages)")
        
        # Find gaps between documents
        print(f"\n3. Gaps between documents:")
        sorted_docs = sorted(documents_info, key=lambda x: x['pages'][0])
        
        for i in range(len(sorted_docs)):
            current_doc = sorted_docs[i]
            current_end = max(current_doc['pages'])
            
            if i < len(sorted_docs) - 1:
                next_doc = sorted_docs[i + 1]
                next_start = min(next_doc['pages'])
                
                if next_start > current_end + 1:
                    gap_pages = list(range(current_end + 1, next_start))
                    print(f"   Gap after '{current_doc['title']}' (page {current_end})")
                    print(f"   Missing pages: {gap_pages}")
            else:
                # Check if there are pages after the last document
                if current_end < total_pages:
                    gap_pages = list(range(current_end + 1, total_pages + 1))
                    print(f"   Gap after last document '{current_doc['title']}' (page {current_end})")
                    print(f"   Missing pages: {gap_pages}")
        
        # Check if there are pages before the first document
        if sorted_docs:
            first_start = min(sorted_docs[0]['pages'])
            if first_start > 1:
                gap_pages = list(range(1, first_start))
                print(f"   Gap before first document (starts at page {first_start})")
                print(f"   Missing pages: {gap_pages}")
    
    else:
        print("❌ Test PDF not found")
    
    # 4. Check if this is a boundary detection issue
    print(f"\n4. Checking boundary detection logic...")
    print("   This might be caused by:")
    print("   - Pages that don't match any document type patterns")
    print("   - Boundary detection algorithm missing transitions")
    print("   - Pages being filtered out during processing")
    print("   - OCR issues preventing proper text extraction")

if __name__ == "__main__":
    main()