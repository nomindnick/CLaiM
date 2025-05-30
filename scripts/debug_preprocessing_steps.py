#!/usr/bin/env python3
"""
Debug each preprocessing step to identify which one breaks OCR.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from pathlib import Path

def test_preprocessing_steps(pdf_path: str, output_dir: str = "preprocessing_debug"):
    """Test each preprocessing step individually."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]  # Test first page
    pdf_name = Path(pdf_path).stem
    
    print(f"Testing preprocessing steps on: {pdf_path}")
    
    # Convert to base image
    mat = fitz.Matrix(300/72.0, 300/72.0)
    pix = page.get_pixmap(matrix=mat)
    original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Test 0: Original image
    original_path = output_path / f"{pdf_name}_step0_original.png"
    original_img.save(original_path)
    
    ocr_text = pytesseract.image_to_string(original_img, lang='eng')
    conf_data = pytesseract.image_to_data(original_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 0 - Original RGB: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 1: Convert to grayscale
    if original_img.mode != 'L':
        gray_img = original_img.convert('L')
    else:
        gray_img = original_img
    
    step1_path = output_path / f"{pdf_name}_step1_grayscale.png"
    gray_img.save(step1_path)
    
    ocr_text = pytesseract.image_to_string(gray_img, lang='eng')
    conf_data = pytesseract.image_to_data(gray_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 1 - Grayscale: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Convert to OpenCV format for advanced preprocessing
    cv_image = np.array(gray_img)
    
    # Step 2: Denoising
    cv_denoised = cv2.fastNlMeansDenoising(cv_image, h=40)
    denoised_img = Image.fromarray(cv_denoised)
    
    step2_path = output_path / f"{pdf_name}_step2_denoised.png"
    denoised_img.save(step2_path)
    
    ocr_text = pytesseract.image_to_string(denoised_img, lang='eng')
    conf_data = pytesseract.image_to_data(denoised_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 2 - Denoised: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 3: Deskewing (simplified)
    def simple_deskew(image):
        coords = np.column_stack(np.where(image > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image
    
    cv_deskewed = simple_deskew(cv_denoised.copy())
    deskewed_img = Image.fromarray(cv_deskewed)
    
    step3_path = output_path / f"{pdf_name}_step3_deskewed.png"
    deskewed_img.save(step3_path)
    
    ocr_text = pytesseract.image_to_string(deskewed_img, lang='eng')
    conf_data = pytesseract.image_to_data(deskewed_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 3 - Deskewed: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cv_clahe = clahe.apply(cv_deskewed)
    clahe_img = Image.fromarray(cv_clahe)
    
    step4_path = output_path / f"{pdf_name}_step4_clahe.png"
    clahe_img.save(step4_path)
    
    ocr_text = pytesseract.image_to_string(clahe_img, lang='eng')
    conf_data = pytesseract.image_to_data(clahe_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 4 - CLAHE: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 5: Binarization (OTSU)
    _, otsu = cv2.threshold(cv_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_img = Image.fromarray(otsu)
    
    step5a_path = output_path / f"{pdf_name}_step5a_otsu.png"
    otsu_img.save(step5a_path)
    
    ocr_text = pytesseract.image_to_string(otsu_img, lang='eng')
    conf_data = pytesseract.image_to_data(otsu_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 5a - OTSU: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 5b: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(cv_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    adaptive_img = Image.fromarray(adaptive)
    
    step5b_path = output_path / f"{pdf_name}_step5b_adaptive.png"
    adaptive_img.save(step5b_path)
    
    ocr_text = pytesseract.image_to_string(adaptive_img, lang='eng')
    conf_data = pytesseract.image_to_data(adaptive_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 5b - Adaptive: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 6: Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph_img = Image.fromarray(morph)
    
    step6_path = output_path / f"{pdf_name}_step6_morphology.png"
    morph_img.save(step6_path)
    
    ocr_text = pytesseract.image_to_string(morph_img, lang='eng')
    conf_data = pytesseract.image_to_data(morph_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 6 - Morphology: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 7: Line removal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(morph, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv_line_removed = morph.copy()
    for c in cnts:
        cv2.drawContours(cv_line_removed, [c], -1, (255,255,255), 2)
    
    line_removed_img = Image.fromarray(cv_line_removed)
    
    step7_path = output_path / f"{pdf_name}_step7_line_removal.png"
    line_removed_img.save(step7_path)
    
    ocr_text = pytesseract.image_to_string(line_removed_img, lang='eng')
    conf_data = pytesseract.image_to_data(line_removed_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 7 - Line removal: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 8: Contrast enhancement
    enhancer = ImageEnhance.Contrast(line_removed_img)
    contrast_img = enhancer.enhance(1.5)
    
    step8_path = output_path / f"{pdf_name}_step8_contrast.png"
    contrast_img.save(step8_path)
    
    ocr_text = pytesseract.image_to_string(contrast_img, lang='eng')
    conf_data = pytesseract.image_to_data(contrast_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 8 - Contrast: {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    # Step 9: Sharpening
    final_img = contrast_img.filter(ImageFilter.SHARPEN)
    
    step9_path = output_path / f"{pdf_name}_step9_final.png"
    final_img.save(step9_path)
    
    ocr_text = pytesseract.image_to_string(final_img, lang='eng')
    conf_data = pytesseract.image_to_data(final_img, lang='eng', output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in conf_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"\nStep 9 - Final (with sharpening): {len(ocr_text)} chars, confidence: {avg_conf:.2f}")
    print(f"Preview: {repr(ocr_text[:100])}")
    
    pdf_doc.close()

if __name__ == "__main__":
    pdf_path = "/home/nick/Projects/CLaiM/tests/test_data/Mixed_Document_Contract_Amendment.pdf"
    if os.path.exists(pdf_path):
        test_preprocessing_steps(pdf_path)
    else:
        print(f"PDF not found: {pdf_path}")