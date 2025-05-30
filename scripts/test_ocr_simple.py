#!/usr/bin/env python3
"""
Simple OCR test to isolate the problem.

This script will test OCR with minimal preprocessing to identify where the issue occurs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import fitz
import pytesseract
from PIL import Image
import io

def test_simple_ocr(pdf_path: str):
    """Test OCR with minimal preprocessing."""
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]  # Test first page only
    
    print(f"Testing simple OCR on: {pdf_path}")
    
    # 1. Extract text with PyMuPDF
    pymupdf_text = page.get_text()
    print(f"\nPyMuPDF text ({len(pymupdf_text)} chars):")
    print(repr(pymupdf_text[:200]))
    
    # 2. Convert to image with minimal processing
    mat = fitz.Matrix(300/72.0, 300/72.0)  # 300 DPI
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    print(f"\nImage info: {img.mode}, {img.size}")
    
    # 3. Test OCR directly on RGB image
    try:
        ocr_text_rgb = pytesseract.image_to_string(img, lang='eng')
        print(f"\nDirect OCR on RGB image ({len(ocr_text_rgb)} chars):")
        print(repr(ocr_text_rgb[:200]))
    except Exception as e:
        print(f"Direct RGB OCR failed: {e}")
    
    # 4. Convert to grayscale and test
    gray_img = img.convert('L')
    try:
        ocr_text_gray = pytesseract.image_to_string(gray_img, lang='eng')
        print(f"\nOCR on grayscale image ({len(ocr_text_gray)} chars):")
        print(repr(ocr_text_gray[:200]))
    except Exception as e:
        print(f"Grayscale OCR failed: {e}")
    
    # 5. Test with different Tesseract configs
    configs = [
        '--oem 3 --psm 3',  # Default
        '--oem 3 --psm 1',  # Automatic page segmentation with OSD
        '--oem 3 --psm 6',  # Uniform block of text
        '--oem 3 --psm 4',  # Single column text
    ]
    
    for i, config in enumerate(configs):
        try:
            ocr_text_config = pytesseract.image_to_string(gray_img, lang='eng', config=config)
            print(f"\nConfig {i+1} ({config}) - {len(ocr_text_config)} chars:")
            print(repr(ocr_text_config[:200]))
        except Exception as e:
            print(f"Config {i+1} failed: {e}")
    
    # 6. Get OCR data with confidence
    try:
        ocr_data = pytesseract.image_to_data(gray_img, lang='eng', output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"\nConfidence analysis:")
            print(f"Average confidence: {avg_conf:.2f}")
            print(f"Confidence range: {min(confidences)} - {max(confidences)}")
        else:
            print("\nNo confident OCR results found")
    except Exception as e:
        print(f"OCR data extraction failed: {e}")
    
    pdf_doc.close()

def test_tesseract_installation():
    """Test if Tesseract is properly installed."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        
        langs = pytesseract.get_languages()
        print(f"Available languages: {langs}")
        
        # Test with a simple image
        from PIL import Image, ImageDraw, ImageFont
        test_img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(test_img)
        draw.text((10, 10), "Hello World", fill='black')
        
        test_text = pytesseract.image_to_string(test_img)
        print(f"Simple test OCR result: {repr(test_text.strip())}")
        
    except Exception as e:
        print(f"Tesseract test failed: {e}")

if __name__ == "__main__":
    print("=== Testing Tesseract Installation ===")
    test_tesseract_installation()
    
    print("\n" + "="*60)
    print("=== Testing OCR on PDF ===")
    
    pdf_path = "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf"
    if os.path.exists(pdf_path):
        test_simple_ocr(pdf_path)
    else:
        print(f"PDF not found: {pdf_path}")