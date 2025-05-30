#!/usr/bin/env python3
"""
Debug OCR process by saving intermediate images and analyzing quality.

This script will help identify issues with image preprocessing and OCR accuracy
by saving images at various stages of the processing pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import fitz
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from modules.document_processor.ocr_handler import OCRHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ocr_process(pdf_path: str, output_dir: str = "ocr_debug_output"):
    """
    Debug the OCR process by saving images at each preprocessing step.
    
    Args:
        pdf_path: Path to PDF file to analyze
        output_dir: Directory to save debug images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize OCR handler
    ocr_handler = OCRHandler(min_confidence=0.3)
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem
    
    print(f"Analyzing PDF: {pdf_path}")
    print(f"Total pages: {pdf_doc.page_count}")
    print(f"Debug images will be saved to: {output_path}")
    
    for page_num in range(min(5, pdf_doc.page_count)):  # Test first 5 pages
        page = pdf_doc[page_num]
        print(f"\n--- Processing Page {page_num + 1} ---")
        
        # 1. Test PyMuPDF text extraction first
        page_text = page.get_text()
        print(f"PyMuPDF extracted text length: {len(page_text)} characters")
        print(f"PyMuPDF text preview: {repr(page_text[:100])}")
        
        # Try different get_text() options
        text_dict = page.get_text("dict")  # Structured text with positions
        text_blocks = page.get_text("blocks")  # Text blocks
        print(f"PyMuPDF dict extraction: {len(str(text_dict))} characters")
        print(f"PyMuPDF blocks extraction: {len(str(text_blocks))} blocks")
        
        # Check if page has embedded fonts or is image-based
        page_dict = page.get_text("dict")
        has_fonts = any("font" in block.get("spans", [{}])[0] for block in page_dict.get("blocks", []) if "spans" in block)
        print(f"Page has embedded fonts: {has_fonts}")
        
        # 2. Convert page to image using different DPI settings
        for dpi in [150, 300, 600]:
            try:
                # Standard conversion
                mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save original image
                original_path = output_path / f"{pdf_name}_page{page_num+1}_{dpi}dpi_1_original.png"
                img.save(original_path)
                print(f"Saved original image ({dpi} DPI): {img.size} - {original_path}")
                
                # Convert to grayscale
                if img.mode != 'L':
                    gray_img = img.convert('L')
                else:
                    gray_img = img
                
                gray_path = output_path / f"{pdf_name}_page{page_num+1}_{dpi}dpi_2_grayscale.png"
                gray_img.save(gray_path)
                
                # Apply OCR preprocessing
                processed_img = ocr_handler._preprocess_image(gray_img)
                processed_path = output_path / f"{pdf_name}_page{page_num+1}_{dpi}dpi_3_processed.png"
                processed_img.save(processed_path)
                print(f"Saved processed image: {processed_path}")
                
                # Perform OCR
                try:
                    ocr_text, confidence = ocr_handler._perform_ocr(processed_img)
                    print(f"OCR at {dpi} DPI - Confidence: {confidence:.3f}")
                    print(f"OCR text length: {len(ocr_text)} characters")
                    print(f"OCR text preview: {repr(ocr_text[:100])}")
                    
                    # Save OCR results
                    results_path = output_path / f"{pdf_name}_page{page_num+1}_{dpi}dpi_4_ocr_results.txt"
                    with open(results_path, 'w', encoding='utf-8') as f:
                        f.write(f"Confidence: {confidence:.3f}\n")
                        f.write(f"Text:\n{ocr_text}")
                    
                except Exception as e:
                    print(f"OCR failed at {dpi} DPI: {e}")
                
            except Exception as e:
                print(f"Image conversion failed at {dpi} DPI: {e}")
    
    pdf_doc.close()

def test_page_text_extraction_methods(pdf_path: str):
    """Test different PyMuPDF text extraction methods."""
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]  # Test first page
    
    print(f"\n=== Testing PyMuPDF text extraction methods ===")
    
    # Method 1: Basic text extraction
    text1 = page.get_text()
    print(f"Method 1 - get_text(): {len(text1)} chars")
    
    # Method 2: Text with layout preservation
    text2 = page.get_text("text")
    print(f"Method 2 - get_text('text'): {len(text2)} chars")
    
    # Method 3: Text blocks
    blocks = page.get_text("blocks")
    text3 = " ".join([block[4] for block in blocks if len(block) > 4])
    print(f"Method 3 - get_text('blocks'): {len(text3)} chars")
    
    # Method 4: Dictionary format (structured)
    text_dict = page.get_text("dict")
    text4 = ""
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text4 += span.get("text", "")
    print(f"Method 4 - get_text('dict'): {len(text4)} chars")
    
    # Method 5: HTML format
    try:
        html = page.get_text("html")
        print(f"Method 5 - get_text('html'): {len(html)} chars")
    except:
        print("Method 5 - get_text('html'): Failed")
    
    # Method 6: XHTML format
    try:
        xhtml = page.get_text("xhtml")
        print(f"Method 6 - get_text('xhtml'): {len(xhtml)} chars")
    except:
        print("Method 6 - get_text('xhtml'): Failed")
    
    # Show first 200 characters of each method
    print(f"\nText previews:")
    print(f"Method 1: {repr(text1[:200])}")
    print(f"Method 2: {repr(text2[:200])}")
    print(f"Method 3: {repr(text3[:200])}")
    print(f"Method 4: {repr(text4[:200])}")
    
    pdf_doc.close()

def test_pymupdf_parameters(pdf_path: str, output_dir: str = "ocr_debug_output"):
    """Test different PyMuPDF rendering parameters."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]  # Test first page
    pdf_name = Path(pdf_path).stem
    
    print(f"\n=== Testing PyMuPDF rendering parameters ===")
    
    # Test different matrix parameters
    test_configs = [
        {"dpi": 300, "alpha": False, "colorspace": None},
        {"dpi": 300, "alpha": True, "colorspace": None},
        {"dpi": 300, "alpha": False, "colorspace": fitz.csGRAY},
        {"dpi": 600, "alpha": False, "colorspace": None},
        {"dpi": 150, "alpha": False, "colorspace": None},
    ]
    
    for i, config in enumerate(test_configs):
        try:
            mat = fitz.Matrix(config["dpi"]/72.0, config["dpi"]/72.0)
            pix = page.get_pixmap(
                matrix=mat, 
                alpha=config["alpha"],
                colorspace=config["colorspace"]
            )
            
            if config["colorspace"] == fitz.csGRAY:
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            img_path = output_path / f"{pdf_name}_config{i+1}_{config['dpi']}dpi.png"
            img.save(img_path)
            print(f"Config {i+1}: DPI={config['dpi']}, Alpha={config['alpha']}, Size={img.size} -> {img_path}")
            
        except Exception as e:
            print(f"Config {i+1} failed: {e}")
    
    pdf_doc.close()

def main():
    """Main function to run OCR debugging."""
    # Test with available PDFs
    test_pdfs = [
        "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Daily_Report_20250504.pdf",
        "/home/nick/Projects/CLaiM/tests/test_data/Invoice_0005.pdf",
    ]
    
    for pdf_path in test_pdfs:
        if Path(pdf_path).exists():
            print(f"\n{'='*60}")
            print(f"DEBUGGING: {Path(pdf_path).name}")
            print(f"{'='*60}")
            
            # Test different text extraction methods
            test_page_text_extraction_methods(pdf_path)
            
            # Test PyMuPDF parameters
            test_pymupdf_parameters(pdf_path)
            
            # Debug full OCR process
            debug_ocr_process(pdf_path)
            
            break  # Just test first available PDF for now
        else:
            print(f"PDF not found: {pdf_path}")

if __name__ == "__main__":
    main()