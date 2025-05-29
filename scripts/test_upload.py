#!/usr/bin/env python3
"""Test document upload endpoint."""

import sys
import requests
from pathlib import Path

def test_upload():
    """Test the document upload endpoint."""
    # API endpoint
    url = "http://localhost:8000/api/v1/documents/upload"
    
    # Find a test PDF
    test_pdfs = list(Path("tests/test_data").glob("*.pdf"))
    if not test_pdfs:
        print("❌ No test PDFs found in tests/test_data/")
        print("Creating a simple test PDF...")
        
        # Create a simple test PDF
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            test_pdf_path = Path("tests/test_data/test_upload.pdf")
            test_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            c = canvas.Canvas(str(test_pdf_path), pagesize=letter)
            c.drawString(100, 750, "Test PDF for Upload")
            c.drawString(100, 730, "This is a test document.")
            c.save()
            
            print(f"✅ Created test PDF: {test_pdf_path}")
            test_pdfs = [test_pdf_path]
        except ImportError:
            print("❌ reportlab not installed. Please install it or provide a test PDF.")
            return
    
    test_pdf = test_pdfs[0]
    print(f"📄 Using test PDF: {test_pdf}")
    
    # Test upload
    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            response = requests.post(url, files=files)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"   Document ID: {data.get('document_id')}")
            print(f"   Job ID: {data.get('job_id')}")
            print(f"   Status: {data.get('status')}")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start with: cd backend && uvicorn api.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Document Upload Endpoint")
    print("=" * 40)
    test_upload()