"""
OCR handler for processing scanned documents in construction litigation files.

This module handles optical character recognition for scanned pages commonly found
in construction documents like faxed RFIs, scanned change orders, and old permits.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import fitz  # PyMuPDF

from backend.shared.exceptions import OCRError

logger = logging.getLogger(__name__)


class OCRHandler:
    """
    Handles OCR processing for scanned document pages.
    
    Optimized for construction litigation documents which often include:
    - Faxed documents with poor quality
    - Scanned blueprints and drawings
    - Handwritten notes on printed forms
    - Multi-generation photocopies
    """
    
    def __init__(self, language: str = "eng", min_confidence: float = 0.6):
        """
        Initialize OCR handler.
        
        Args:
            language: Tesseract language code (default: "eng" for English)
            min_confidence: Minimum confidence threshold for OCR results (0-1)
        """
        self.language = language
        self.min_confidence = min_confidence
        self._verify_tesseract()
        
    def _verify_tesseract(self) -> None:
        """Verify Tesseract is installed and accessible."""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise OCRError(
                "Tesseract is not installed or not in PATH. "
                "Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
            )
    
    def process_page(self, page: fitz.Page, dpi: int = 300) -> Tuple[str, float]:
        """
        Process a single PDF page with OCR.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering PDF to image
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Convert PDF page to image
            image = self._page_to_image(page, dpi)
            
            # Preprocess image for better OCR accuracy
            processed_image = self._preprocess_image(image)
            
            # Perform OCR with confidence scoring
            text, confidence = self._perform_ocr(processed_image)
            
            # Post-process text for construction documents
            text = self._post_process_text(text)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise OCRError(f"Failed to process page with OCR: {str(e)}")
    
    def _page_to_image(self, page: fitz.Page, dpi: int) -> Image.Image:
        """Convert PDF page to PIL Image."""
        # Render page to pixmap
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        return Image.open(io.BytesIO(img_data))
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Applies various image enhancement techniques optimized for
        construction documents which are often:
        - Faxed (low resolution, noisy)
        - Photocopied multiple times
        - Contains stamps and handwriting
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert PIL to OpenCV format for advanced preprocessing
        cv_image = np.array(image)
        
        # 1. Denoise - especially important for faxed documents
        cv_image = cv2.fastNlMeansDenoising(cv_image, h=30)
        
        # 2. Deskew - fix tilted scans
        cv_image = self._deskew_image(cv_image)
        
        # 3. Binarization - convert to pure black and white
        # Use adaptive thresholding for documents with varying lighting
        cv_image = cv2.adaptiveThreshold(
            cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Remove small noise particles
        kernel = np.ones((2, 2), np.uint8)
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        processed = Image.fromarray(cv_image)
        
        # 5. Enhance contrast for faded documents
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.5)
        
        # 6. Slight sharpening for blurry scans
        processed = processed.filter(ImageFilter.SHARPEN)
        
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image skew."""
        # Find all white pixels
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) > 100:  # Need enough points
            # Find minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct angle
            if angle < -45:
                angle = 90 + angle
            
            # Rotate image to deskew
            if abs(angle) > 0.5:  # Only rotate if skew is significant
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        return image
    
    def _perform_ocr(self, image: Image.Image) -> Tuple[str, float]:
        """
        Perform OCR with confidence scoring.
        
        Returns:
            Tuple of (text, average_confidence)
        """
        # Configure Tesseract for construction documents
        custom_config = r'--oem 3 --psm 3'  # Use best OCR engine mode and auto page segmentation
        
        # Get detailed OCR data including confidence scores
        ocr_data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate confidence
        words = []
        confidences = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # Only include words with positive confidence
            if text and conf > 0:
                words.append(text)
                confidences.append(conf / 100.0)  # Convert to 0-1 range
        
        # Join words with appropriate spacing
        full_text = ' '.join(words)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Log low confidence results
        if avg_confidence < self.min_confidence:
            logger.warning(
                f"Low OCR confidence ({avg_confidence:.2f}) - "
                f"text may be unreliable"
            )
        
        return full_text, avg_confidence
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-process OCR text for construction document specifics.
        
        Handles common OCR errors in construction documents:
        - Fix common abbreviations (RFI, CO, ASI, etc.)
        - Correct date formats
        - Fix common construction terms
        """
        if not text:
            return text
        
        # Common construction abbreviation fixes
        replacements = {
            r'\bRF[Il1]\b': 'RFI',  # Request for Information
            r'\bC[O0]\b': 'CO',     # Change Order
            r'\bAS[Il1]\b': 'ASI',  # Architect's Supplemental Instruction
            r'\bPCO\b': 'PCO',      # Proposed Change Order
            r'\bCCD\b': 'CCD',      # Construction Change Directive
            r'\bSUBM[Il1]TTAL\b': 'SUBMITTAL',
            r'\b[Il1]NVOICE\b': 'INVOICE',
        }
        
        import re
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix common date OCR errors (e.g., "0" read as "O")
        # Match dates with potential OCR errors and fix them
        def fix_date_ocr(match):
            date_str = match.group(0)
            # Replace common OCR errors
            fixed = date_str.replace('O', '0').replace('l', '1').replace('I', '1')
            return fixed
        
        # Match various date formats with potential OCR errors
        date_pattern = r'\b(\d{0,2}[O0Il1]?\d?)[/\-](\d{0,2}[O0Il1]?\d?)[/\-](\d{2,4})\b'
        text = re.sub(date_pattern, fix_date_ocr, text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def process_image_file(self, image_path: str) -> Tuple[str, float]:
        """
        Process a standalone image file with OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            image = Image.open(image_path)
            processed_image = self._preprocess_image(image)
            return self._perform_ocr(processed_image)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            raise OCRError(f"Failed to process image: {str(e)}")
    
    def get_supported_languages(self) -> list[str]:
        """Get list of available Tesseract languages."""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang not in ['osd', 'snum']]
        except Exception as e:
            logger.error(f"Failed to get languages: {str(e)}")
            return ['eng']  # Default to English


# Import for BytesIO
import io