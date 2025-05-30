"""
Hybrid text extractor that intelligently chooses between PyMuPDF and OCR.

This module implements a smart text extraction strategy:
1. First tries PyMuPDF text extraction
2. Evaluates quality of extracted text
3. Falls back to OCR only when needed
4. Supports multiple OCR engines as fallback
"""

import logging
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
from enum import Enum
import re
import fitz  # PyMuPDF

from .improved_ocr_handler import ImprovedOCRHandler
from shared.exceptions import OCRError

logger = logging.getLogger(__name__)


class TextExtractionMethod(Enum):
    """Text extraction methods used."""
    PYMUPDF = "pymupdf"
    OCR_TESSERACT = "ocr_tesseract"
    OCR_EASYOCR = "ocr_easyocr"
    OCR_PADDLEOCR = "ocr_paddleocr"
    HYBRID = "hybrid"


class TextQuality(Enum):
    """Text quality assessment."""
    EXCELLENT = "excellent"      # > 95% confidence, clean text
    GOOD = "good"               # > 80% confidence, minor issues
    POOR = "poor"               # > 50% confidence, significant issues
    UNUSABLE = "unusable"       # < 50% confidence, mostly garbage


class HybridTextExtractor:
    """
    Hybrid text extractor that combines PyMuPDF and OCR intelligently.
    
    This extractor:
    1. Always tries PyMuPDF first (fastest, most accurate for native PDFs)
    2. Evaluates text quality using multiple heuristics
    3. Falls back to OCR only when PyMuPDF fails or produces poor results
    4. Supports multiple OCR engines for maximum compatibility
    """
    
    def __init__(self, 
                 language: str = "eng",
                 min_confidence: float = 0.6,
                 enable_easyocr: bool = False,
                 enable_paddleocr: bool = False,
                 ocr_handler: Optional[ImprovedOCRHandler] = None):
        """
        Initialize hybrid text extractor.
        
        Args:
            language: Language code for OCR (default: "eng")
            min_confidence: Minimum confidence threshold for OCR results
            enable_easyocr: Enable EasyOCR as fallback (requires installation)
            enable_paddleocr: Enable PaddleOCR as fallback (requires installation)
            ocr_handler: Existing OCR handler to reuse (for cache sharing)
        """
        self.language = language
        self.min_confidence = min_confidence
        
        # Initialize OCR handlers
        if ocr_handler is not None:
            self.tesseract_handler = ocr_handler
        else:
            self.tesseract_handler = ImprovedOCRHandler(language, min_confidence)
        
        # Optional OCR engines (lazy initialization)
        self.easyocr_handler = None
        self.paddleocr_handler = None
        self.enable_easyocr = enable_easyocr
        self.enable_paddleocr = enable_paddleocr
        
        # Text quality thresholds
        self.min_text_length = 20  # Minimum characters for meaningful text
        self.min_word_count = 5    # Minimum words for meaningful text
        
    def extract_text(self, page: fitz.Page, dpi: int = 300) -> Tuple[str, float, TextExtractionMethod]:
        """
        Extract text using the best available method.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for OCR if needed
            
        Returns:
            Tuple of (text, confidence, method_used)
        """
        try:
            # Step 1: Try PyMuPDF extraction
            pymupdf_text = self._extract_with_pymupdf(page)
            pymupdf_quality = self._assess_text_quality(pymupdf_text)
            
            logger.debug(f"PyMuPDF: {len(pymupdf_text)} chars, quality: {pymupdf_quality.value}")
            
            # If PyMuPDF gives excellent or good results, use it
            if pymupdf_quality in [TextQuality.EXCELLENT, TextQuality.GOOD]:
                confidence = self._estimate_pymupdf_confidence(pymupdf_text, pymupdf_quality)
                logger.info(f"Using PyMuPDF extraction: {len(pymupdf_text)} chars, confidence: {confidence:.3f}")
                return pymupdf_text, confidence, TextExtractionMethod.PYMUPDF
            
            # Step 2: PyMuPDF failed or gave poor results - try OCR
            logger.info(f"PyMuPDF quality is {pymupdf_quality.value}, trying OCR")
            
            # Try OCR engines in order of preference
            ocr_results = []
            
            # Try Tesseract (always available)
            try:
                ocr_text, ocr_conf = self.tesseract_handler.process_page(page, dpi)
                ocr_results.append((ocr_text, ocr_conf, TextExtractionMethod.OCR_TESSERACT))
                logger.debug(f"Tesseract: {len(ocr_text)} chars, confidence: {ocr_conf:.3f}")
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            
            # Try EasyOCR if enabled
            if self.enable_easyocr:
                try:
                    easy_text, easy_conf = self._extract_with_easyocr(page, dpi)
                    ocr_results.append((easy_text, easy_conf, TextExtractionMethod.OCR_EASYOCR))
                    logger.debug(f"EasyOCR: {len(easy_text)} chars, confidence: {easy_conf:.3f}")
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Try PaddleOCR if enabled
            if self.enable_paddleocr:
                try:
                    paddle_text, paddle_conf = self._extract_with_paddleocr(page, dpi)
                    ocr_results.append((paddle_text, paddle_conf, TextExtractionMethod.OCR_PADDLEOCR))
                    logger.debug(f"PaddleOCR: {len(paddle_text)} chars, confidence: {paddle_conf:.3f}")
                except Exception as e:
                    logger.warning(f"PaddleOCR failed: {e}")
            
            # Step 3: Choose best OCR result
            if ocr_results:
                best_text, best_conf, best_method = max(ocr_results, key=lambda x: x[1])
                
                # Compare with PyMuPDF if it had some text
                if len(pymupdf_text.strip()) > 0 and pymupdf_quality != TextQuality.UNUSABLE:
                    # Hybrid approach: combine if both have useful content
                    combined_text = self._combine_text_sources(pymupdf_text, best_text)
                    if len(combined_text) > len(best_text):
                        logger.info(f"Using hybrid approach: PyMuPDF + {best_method.value}")
                        return combined_text, best_conf, TextExtractionMethod.HYBRID
                
                logger.info(f"Using {best_method.value}: {len(best_text)} chars, confidence: {best_conf:.3f}")
                return best_text, best_conf, best_method
            
            # Step 4: All OCR failed, return PyMuPDF even if poor quality
            if len(pymupdf_text.strip()) > 0:
                confidence = self._estimate_pymupdf_confidence(pymupdf_text, pymupdf_quality)
                logger.warning(f"All OCR failed, using PyMuPDF despite {pymupdf_quality.value} quality")
                return pymupdf_text, confidence, TextExtractionMethod.PYMUPDF
            
            # Step 5: Complete failure
            logger.error("All text extraction methods failed")
            return "", 0.0, TextExtractionMethod.PYMUPDF
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise OCRError(f"Text extraction failed: {e}")
    
    def _extract_with_pymupdf(self, page: fitz.Page) -> str:
        """Extract text using PyMuPDF with optimized parameters."""
        try:
            # Try different extraction methods and pick the best
            methods = [
                lambda: page.get_text(),              # Default
                lambda: page.get_text("text"),        # Text with layout
                lambda: page.get_text("blocks"),      # Block-based
            ]
            
            best_text = ""
            
            for method in methods:
                try:
                    if callable(method):
                        text = method()
                    else:
                        text = str(method)  # Handle blocks format
                    
                    # For blocks, join text
                    if isinstance(text, list):
                        text = " ".join([block[4] for block in text if len(block) > 4])
                    
                    # Pick longest reasonable text
                    if len(text) > len(best_text):
                        best_text = text
                        
                except Exception as e:
                    logger.debug(f"PyMuPDF method failed: {e}")
                    continue
            
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _assess_text_quality(self, text: str) -> TextQuality:
        """
        Assess the quality of extracted text using multiple heuristics.
        
        Args:
            text: Text to assess
            
        Returns:
            TextQuality enum value
        """
        if not text or len(text.strip()) < self.min_text_length:
            return TextQuality.UNUSABLE
        
        # Calculate various quality metrics
        metrics = self._calculate_text_metrics(text)
        
        # Scoring system
        score = 0
        max_score = 100
        
        # Length score (0-20 points)
        if len(text) >= 100:
            score += 20
        elif len(text) >= 50:
            score += 15
        elif len(text) >= self.min_text_length:
            score += 10
        
        # Word count score (0-15 points)
        if metrics['word_count'] >= 20:
            score += 15
        elif metrics['word_count'] >= 10:
            score += 10
        elif metrics['word_count'] >= self.min_word_count:
            score += 5
        
        # Character diversity score (0-15 points)
        if metrics['char_diversity'] >= 0.3:
            score += 15
        elif metrics['char_diversity'] >= 0.2:
            score += 10
        elif metrics['char_diversity'] >= 0.1:
            score += 5
        
        # Readability score (0-20 points)
        if metrics['readable_ratio'] >= 0.9:
            score += 20
        elif metrics['readable_ratio'] >= 0.7:
            score += 15
        elif metrics['readable_ratio'] >= 0.5:
            score += 10
        elif metrics['readable_ratio'] >= 0.3:
            score += 5
        
        # Structure score (0-15 points)
        if metrics['has_structure']:
            score += 15
        elif metrics['has_dates'] or metrics['has_numbers']:
            score += 10
        
        # Garbage detection penalty (0-15 points)
        if metrics['garbage_ratio'] < 0.1:
            score += 15
        elif metrics['garbage_ratio'] < 0.3:
            score += 10
        elif metrics['garbage_ratio'] < 0.5:
            score += 5
        
        # Convert score to quality
        percentage = (score / max_score) * 100
        
        if percentage >= 85:
            return TextQuality.EXCELLENT
        elif percentage >= 65:
            return TextQuality.GOOD
        elif percentage >= 40:
            return TextQuality.POOR
        else:
            return TextQuality.UNUSABLE
    
    def _calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate various text quality metrics."""
        if not text:
            return {
                'word_count': 0, 'char_diversity': 0, 'readable_ratio': 0,
                'garbage_ratio': 1, 'has_structure': False, 'has_dates': False,
                'has_numbers': False
            }
        
        words = text.split()
        chars = set(text.lower())
        
        # Character diversity (unique chars / total chars)
        char_diversity = len(chars) / len(text) if text else 0
        
        # Readable characters ratio
        readable_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,;:!?()-[]{}/"\'')
        readable_ratio = readable_chars / len(text) if text else 0
        
        # Garbage ratio (non-printable or weird characters)
        garbage_chars = sum(1 for c in text if ord(c) > 127 or c in '§©®°±µ¶·¸¹º»¼½¾¿×÷')
        garbage_ratio = garbage_chars / len(text) if text else 0
        
        # Structure detection
        has_structure = bool(re.search(r'[A-Z][A-Z\s]{10,}', text))  # Headers/titles
        has_dates = bool(re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text))
        has_numbers = bool(re.search(r'\$[\d,]+\.?\d*|\d+\.?\d*%', text))  # Money/percentages
        
        return {
            'word_count': len(words),
            'char_diversity': char_diversity,
            'readable_ratio': readable_ratio,
            'garbage_ratio': garbage_ratio,
            'has_structure': has_structure,
            'has_dates': has_dates,
            'has_numbers': has_numbers,
        }
    
    def _estimate_pymupdf_confidence(self, text: str, quality: TextQuality) -> float:
        """Estimate confidence for PyMuPDF extracted text."""
        base_confidence = {
            TextQuality.EXCELLENT: 0.98,
            TextQuality.GOOD: 0.85,
            TextQuality.POOR: 0.65,
            TextQuality.UNUSABLE: 0.30,
        }
        
        conf = base_confidence.get(quality, 0.50)
        
        # Adjust based on text length
        if len(text) > 500:
            conf += 0.02
        elif len(text) < 50:
            conf -= 0.10
        
        return min(1.0, max(0.0, conf))
    
    def _combine_text_sources(self, pymupdf_text: str, ocr_text: str) -> str:
        """
        Intelligently combine text from multiple sources.
        
        This method merges PyMuPDF and OCR results when both have useful content.
        """
        if not pymupdf_text:
            return ocr_text
        if not ocr_text:
            return pymupdf_text
        
        # Simple combination: use the longer text as base
        if len(ocr_text) > len(pymupdf_text) * 1.2:
            return ocr_text
        else:
            return pymupdf_text
    
    def _extract_with_easyocr(self, page: fitz.Page, dpi: int) -> Tuple[str, float]:
        """Extract text using EasyOCR (if available)."""
        if self.easyocr_handler is None:
            try:
                import easyocr
                self.easyocr_handler = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                raise OCRError("EasyOCR not installed. Install with: pip install easyocr")
        
        # Convert page to image
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        import numpy as np
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, 3)
        
        # Run EasyOCR
        results = self.easyocr_handler.readtext(img_array)
        
        # Extract text and calculate confidence
        text_parts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            if text.strip() and conf > 0.3:  # Filter low confidence
                text_parts.append(text)
                confidences.append(conf)
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    def _extract_with_paddleocr(self, page: fitz.Page, dpi: int) -> Tuple[str, float]:
        """Extract text using PaddleOCR (if available)."""
        if self.paddleocr_handler is None:
            try:
                from paddleocr import PaddleOCR
                self.paddleocr_handler = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except ImportError:
                raise OCRError("PaddleOCR not installed. Install with: pip install paddleocr")
        
        # Convert page to image
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        import numpy as np
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, 3)
        
        # Run PaddleOCR
        results = self.paddleocr_handler.ocr(img_array, cls=True)
        
        # Extract text and calculate confidence
        text_parts = []
        confidences = []
        
        for line in results[0] if results and results[0] else []:
            if line and len(line) >= 2:
                text = line[1][0] if len(line[1]) >= 1 else ""
                conf = line[1][1] if len(line[1]) >= 2 else 0.0
                
                if text.strip() and conf > 0.3:
                    text_parts.append(text)
                    confidences.append(conf)
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence