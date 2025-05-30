#!/usr/bin/env python3
"""
Simple OCR engine test - just test first few pages with different engines.
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz
import numpy as np
from PIL import Image

# Import our existing OCR handler
from modules.document_processor.ocr_handler import OCRHandler
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler


def test_tesseract_engines(pdf_path: str):
    """Test our two Tesseract-based engines."""
    print("🔍 Testing Tesseract-based OCR engines")
    print("=" * 60)
    
    pdf_doc = fitz.open(pdf_path)
    
    # Test pages where we expect different document types
    test_pages = [0, 4, 8, 12]  # Pages 1, 5, 9, 13 (0-indexed)
    expected_types = ["Email Chain", "Email Chain", "Schedule of Values", "Email"]
    
    original_handler = OCRHandler()
    improved_handler = ImprovedOCRHandler()
    
    for i, page_num in enumerate(test_pages):
        if page_num >= pdf_doc.page_count:
            continue
            
        page = pdf_doc[page_num]
        expected_type = expected_types[i]
        
        print(f"\n📄 PAGE {page_num + 1} - Expected: {expected_type}")
        print("-" * 50)
        
        # Test original OCR handler
        print("🔧 Original OCR Handler:")
        try:
            start_time = time.time()
            text1, conf1 = original_handler.process_page(page, dpi=300)
            time1 = time.time() - start_time
            print(f"   ✅ Confidence: {conf1:.3f}, Length: {len(text1)} chars, Time: {time1:.2f}s")
            print(f"   📝 First 100 chars: '{text1[:100]}...'")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test improved OCR handler
        print("🔧 Improved OCR Handler:")
        try:
            start_time = time.time()
            text2, conf2 = improved_handler.process_page(page, dpi=300)
            time2 = time.time() - start_time
            print(f"   ✅ Confidence: {conf2:.3f}, Length: {len(text2)} chars, Time: {time2:.2f}s")
            print(f"   📝 First 100 chars: '{text2[:100]}...'")
            
            # Compare improvement
            if conf2 > conf1:
                improvement = ((conf2 - conf1) / conf1 * 100) if conf1 > 0 else float('inf')
                print(f"   📈 Improvement: +{improvement:.1f}% confidence")
            else:
                decline = ((conf1 - conf2) / conf1 * 100) if conf1 > 0 else 0
                print(f"   📉 Decline: -{decline:.1f}% confidence")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    pdf_doc.close()


def test_easyocr_if_available(pdf_path: str):
    """Test EasyOCR if available."""
    try:
        import easyocr
        
        print(f"\n🔧 Testing EasyOCR")
        print("-" * 30)
        
        # Initialize EasyOCR
        print("   Loading EasyOCR model...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("   ✅ EasyOCR loaded")
        
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]  # Test just the first page
        
        # Convert page to image
        mat = fitz.Matrix(300/72.0, 300/72.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Run EasyOCR
        print("   Running EasyOCR on page 1...")
        start_time = time.time()
        results = reader.readtext(img_array, detail=1)
        processing_time = time.time() - start_time
        
        # Extract text and confidence
        texts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            if text.strip():
                texts.append(text.strip())
                confidences.append(conf)
        
        full_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        print(f"   ✅ Confidence: {avg_confidence:.3f}, Length: {len(full_text)} chars, Time: {processing_time:.2f}s")
        print(f"   📝 First 100 chars: '{full_text[:100]}...'")
        
        pdf_doc.close()
        
    except ImportError:
        print("⚠️  EasyOCR not available")
    except Exception as e:
        print(f"❌ EasyOCR failed: {e}")


def analyze_ocr_quality(text: str, confidence: float) -> dict:
    """Analyze OCR quality indicators."""
    if not text:
        return {"readable": False, "has_words": False, "has_patterns": False, "score": 0.0}
    
    # Basic readability checks
    word_count = len(text.split())
    char_count = len(text)
    
    # Check for readable patterns
    import re
    has_email_patterns = bool(re.search(r'(from|to|subject):', text, re.IGNORECASE))
    has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
    has_amounts = bool(re.search(r'\$[\d,]+', text))
    has_words = word_count > 5
    
    # Calculate quality score
    quality_score = 0.0
    if confidence > 0.7:
        quality_score += 0.4
    elif confidence > 0.5:
        quality_score += 0.2
    
    if has_words:
        quality_score += 0.2
    if has_email_patterns or has_dates or has_amounts:
        quality_score += 0.2
    if word_count > 20:
        quality_score += 0.2
    
    return {
        "readable": confidence > 0.5 and has_words,
        "has_words": has_words,
        "has_patterns": has_email_patterns or has_dates or has_amounts,
        "word_count": word_count,
        "confidence": confidence,
        "quality_score": quality_score
    }


def main():
    """Main test execution."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    print(f"🚀 Simple OCR Engine Test")
    print(f"📄 PDF: {test_pdf_path}")
    
    # Test Tesseract engines
    test_tesseract_engines(test_pdf_path)
    
    # Test EasyOCR if available
    test_easyocr_if_available(test_pdf_path)
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    main()