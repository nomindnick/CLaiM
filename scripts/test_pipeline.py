#!/usr/bin/env python3
"""Test the complete document processing pipeline."""

import time
import json
import requests
from pathlib import Path


def test_pipeline():
    """Test upload -> process -> store -> retrieve pipeline."""
    # API base URL
    base_url = "http://localhost:8000/api/v1"
    
    # Find a test PDF
    test_pdfs = list(Path("tests/test_data").glob("*.pdf"))
    if not test_pdfs:
        print("❌ No test PDFs found")
        return
    
    test_pdf = test_pdfs[0]
    print(f"📄 Using test PDF: {test_pdf}")
    
    # Step 1: Upload document
    print("\n1️⃣ Uploading document...")
    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/documents/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        upload_data = response.json()
        print(f"✅ Upload successful!")
        print(f"   Document ID: {upload_data.get('document_id')}")
        print(f"   Job ID: {upload_data.get('job_id')}")
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return
    
    # Step 2: Wait for processing
    print("\n2️⃣ Waiting for background processing...")
    print("   (This includes OCR, metadata extraction, and storage)")
    time.sleep(5)  # Give it time to process
    
    # Step 3: List documents
    print("\n3️⃣ Listing stored documents...")
    try:
        response = requests.get(f"{base_url}/documents/list")
        
        if response.status_code != 200:
            print(f"❌ List failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        list_data = response.json()
        print(f"✅ Found {list_data['total']} documents")
        
        if list_data['documents']:
            for doc in list_data['documents'][:3]:  # Show first 3
                print(f"\n   Document: {doc['id']}")
                print(f"   - Title: {doc['title']}")
                print(f"   - Type: {doc['type']}")
                print(f"   - Pages: {doc['page_count']}")
                if doc.get('metadata'):
                    meta = doc['metadata']
                    if meta.get('parties'):
                        print(f"   - Parties: {', '.join(meta['parties'][:3])}")
                    if meta.get('dates'):
                        print(f"   - Dates: {', '.join(meta['dates'][:3])}")
                    if meta.get('amounts'):
                        print(f"   - Amounts: ${', '.join(str(a) for a in meta['amounts'][:3])}")
        
    except Exception as e:
        print(f"❌ List error: {e}")
        return
    
    # Step 4: Get specific document
    if list_data['documents']:
        first_doc = list_data['documents'][0]
        doc_id = first_doc['id']
        
        print(f"\n4️⃣ Retrieving document {doc_id}...")
        try:
            response = requests.get(f"{base_url}/documents/{doc_id}")
            
            if response.status_code != 200:
                print(f"❌ Get failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return
            
            doc_data = response.json()
            print(f"✅ Retrieved document!")
            print(f"   Storage path: {doc_data.get('storage_path')}")
            
            # Pretty print metadata
            if doc_data.get('metadata'):
                print("\n   Extracted Metadata:")
                print(json.dumps(doc_data['metadata'], indent=4))
            
        except Exception as e:
            print(f"❌ Get error: {e}")
    
    # Step 5: Search documents
    print("\n5️⃣ Testing search functionality...")
    try:
        # Search for RFI documents
        response = requests.post(
            f"{base_url}/storage/search",
            json={"text_query": "RFI", "limit": 5}
        )
        
        if response.status_code == 200:
            search_data = response.json()
            print(f"✅ Search found {search_data['total_results']} results")
            for result in search_data['documents'][:2]:
                print(f"   - {result['title']} (Score: {result.get('score', 'N/A')})")
        else:
            print(f"⚠️  Search returned {response.status_code}")
    
    except Exception as e:
        print(f"⚠️  Search error (non-critical): {e}")
    
    print("\n✅ Pipeline test complete!")


if __name__ == "__main__":
    print("🧪 Testing Document Processing Pipeline")
    print("=" * 50)
    print("Make sure the backend is running: ./scripts/start_backend.sh")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ Backend is not responding. Please start it first.")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Please start it first.")
        print("   Run: ./scripts/start_backend.sh")
        exit(1)
    
    test_pipeline()