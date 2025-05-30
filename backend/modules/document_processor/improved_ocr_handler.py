"""
Improved OCR handler that avoids destructive preprocessing.

This module implements a smarter OCR approach that:
1. Tests multiple preprocessing strategies
2. Picks the best result based on confidence
3. Avoids destructive operations on clean documents
4. Only applies heavy preprocessing when needed
5. Caches OCR results to avoid duplicate processing
"""

import io
import logging
import hashlib
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import fitz  # PyMuPDF
import diskcache

from shared.exceptions import OCRError

logger = logging.getLogger(__name__)


class ImprovedOCRHandler:
    """
    Improved OCR handler that uses adaptive preprocessing strategies.
    
    Key improvements:
    - Tests multiple preprocessing approaches
    - Selects best result based on confidence
    - Avoids destructive operations on clean documents
    - Supports multiple OCR engines
    """
    
    def __init__(self, language: str = "eng", min_confidence: float = 0.4, cache_dir: str = ".ocr_cache"):
        """
        Initialize improved OCR handler.
        
        Args:
            language: Tesseract language code (default: "eng" for English)
            min_confidence: Minimum confidence threshold for OCR results (0-1)
            cache_dir: Directory for OCR result cache
        """
        self.language = language
        self.min_confidence = min_confidence
        self.cache_dir = cache_dir
        self._verify_tesseract()
        
        # Initialize disk cache with 1GB memory limit
        self.cache = diskcache.Cache(
            directory=cache_dir,
            size_limit=10 * 1024**3,  # 10GB disk limit
            eviction_policy='least-recently-used'
        )
        
        # Set memory limit to 1GB before using disk
        # Note: diskcache handles this automatically with its memory mapping
        
        # OCR strategies to try (in order of preference)
        self.strategies = [
            "minimal",      # Minimal preprocessing for clean documents
            "standard",     # Standard preprocessing without deskewing
            "aggressive",   # Full preprocessing for difficult documents
        ]
        
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
        Process a single PDF page with OCR using adaptive strategies.
        
        Uses disk cache to avoid duplicate OCR processing.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering PDF to image
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Generate cache key based on page content and processing parameters
            cache_key = self._generate_cache_key(page, dpi)
            
            # Check cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                text, confidence = cached_result
                logger.info(f"✓ OCR cache hit: {len(text)} chars, confidence: {confidence:.3f}")
                return text, confidence
            
            logger.info(f"⚡ OCR cache miss - processing page...")
            
            # Convert PDF page to image
            image = self._page_to_image(page, dpi)
            
            # Try different preprocessing strategies
            best_text = ""
            best_confidence = 0.0
            best_strategy = "none"
            
            for strategy in self.strategies:
                try:
                    text, confidence = self._try_strategy(image, strategy)
                    
                    logger.info(f"Strategy '{strategy}': {len(text)} chars, confidence: {confidence:.3f}")
                    
                    # If this strategy gives good results, use it
                    if confidence > best_confidence:
                        best_text = text
                        best_confidence = confidence
                        best_strategy = strategy
                    
                    # If we get excellent results with minimal processing, stop here
                    if strategy == "minimal" and confidence > 0.9:
                        logger.info(f"Excellent results with minimal preprocessing: {confidence:.3f}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Strategy '{strategy}' failed: {e}")
                    continue
            
            # Post-process the best result
            if best_text:
                best_text = self._post_process_text(best_text)
                logger.info(f"Best strategy: '{best_strategy}' with confidence {best_confidence:.3f}")
                
                # Cache the result
                self.cache.set(cache_key, (best_text, best_confidence))
                logger.info(f"💾 Cached OCR result (key: {cache_key[:16]}...)")
            else:
                logger.error("All OCR strategies failed")
                raise OCRError("All OCR strategies failed")
            
            return best_text, best_confidence
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise OCRError(f"Failed to process page with OCR: {str(e)}")
    
    def _page_to_image(self, page: fitz.Page, dpi: int) -> Image.Image:
        """Convert PDF page to PIL Image with optimized parameters."""
        try:
            # Use higher quality rendering
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            
            # Try grayscale first (often better for OCR)
            try:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            except:
                # Fallback to RGB
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if img.mode != 'L':
                    img = img.convert('L')
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to convert page to image: {str(e)}")
            raise
    
    def _try_strategy(self, image: Image.Image, strategy: str) -> Tuple[str, float]:
        """
        Try a specific preprocessing strategy.
        
        Args:
            image: PIL Image to process
            strategy: Strategy name ("minimal", "standard", "aggressive")
            
        Returns:
            Tuple of (text, confidence)
        """
        if strategy == "minimal":
            return self._minimal_preprocessing(image)
        elif strategy == "standard":
            return self._standard_preprocessing(image)
        elif strategy == "aggressive":
            return self._aggressive_preprocessing(image)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _minimal_preprocessing(self, image: Image.Image) -> Tuple[str, float]:
        """
        Minimal preprocessing - just ensure grayscale and test OCR.
        
        Best for: Clean documents, "print as PDF" files
        """
        # Ensure grayscale
        if image.mode != 'L':
            processed = image.convert('L')
        else:
            processed = image
        
        # Try multiple Tesseract configurations
        configs = [
            '--oem 3 --psm 3',  # Automatic page segmentation
            '--oem 3 --psm 6',  # Uniform block of text
            '--oem 3 --psm 4',  # Single column text
        ]
        
        best_text = ""
        best_conf = 0.0
        
        for config in configs:
            text, conf = self._perform_ocr(processed, config)
            if conf > best_conf:
                best_text = text
                best_conf = conf
        
        return best_text, best_conf
    
    def _standard_preprocessing(self, image: Image.Image) -> Tuple[str, float]:
        """
        Standard preprocessing without destructive operations.
        
        Best for: Scanned documents, faxes, slight quality issues
        """
        # Ensure grayscale
        if image.mode != 'L':
            processed = image.convert('L')
        else:
            processed = image
        
        # Convert to numpy for OpenCV processing
        cv_image = np.array(processed)
        
        # Light denoising
        cv_image = cv2.fastNlMeansDenoising(cv_image, h=30)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cv_image = clahe.apply(cv_image)
        
        # Light binarization (try both methods)
        _, otsu = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Test both and pick the better one
        otsu_img = Image.fromarray(otsu)
        adaptive_img = Image.fromarray(adaptive)
        
        otsu_text, otsu_conf = self._perform_ocr(otsu_img, '--oem 3 --psm 6')
        adaptive_text, adaptive_conf = self._perform_ocr(adaptive_img, '--oem 3 --psm 6')
        
        if otsu_conf > adaptive_conf:
            return otsu_text, otsu_conf
        else:
            return adaptive_text, adaptive_conf
    
    def _aggressive_preprocessing(self, image: Image.Image) -> Tuple[str, float]:
        """
        Aggressive preprocessing for very poor quality documents.
        
        Best for: Multiple-generation photocopies, damaged documents
        WARNING: Can destroy good documents - only use as last resort
        """
        # Ensure grayscale
        if image.mode != 'L':
            processed = image.convert('L')
        else:
            processed = image
        
        # Convert to numpy
        cv_image = np.array(processed)
        
        # Strong denoising
        cv_image = cv2.fastNlMeansDenoising(cv_image, h=40)
        
        # Careful deskewing (only if really needed)
        cv_image = self._careful_deskew(cv_image)
        
        # Strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        cv_image = clahe.apply(cv_image)
        
        # Binarization
        _, binary = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        processed = Image.fromarray(binary)
        
        # Final enhancements
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.2)
        
        return self._perform_ocr(processed, '--oem 3 --psm 6')
    
    def _careful_deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Careful deskewing that checks if it's actually needed.
        
        Only deskews if:
        1. Significant skew is detected (> 2 degrees)
        2. The result improves OCR confidence
        """
        # Find skew angle
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 1000:  # Not enough points for reliable skew detection
            return image
        
        try:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            
            # Only deskew if angle is significant
            if abs(angle) < 2.0:  # Less than 2 degrees - don't bother
                return image
            
            # Test deskewing
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Quick OCR test to see if deskewing helps
            original_img = Image.fromarray(image)
            deskewed_img = Image.fromarray(deskewed)
            
            try:
                _, orig_conf = self._perform_ocr(original_img, '--oem 3 --psm 6')
                _, desk_conf = self._perform_ocr(deskewed_img, '--oem 3 --psm 6')
                
                # Only use deskewed if it's significantly better
                if desk_conf > orig_conf + 0.1:  # At least 10% improvement
                    logger.info(f"Deskewing improved confidence: {orig_conf:.3f} -> {desk_conf:.3f}")
                    return deskewed
                else:
                    logger.info(f"Deskewing didn't help: {orig_conf:.3f} -> {desk_conf:.3f}")
                    return image
            except:
                # If testing fails, don't deskew
                return image
                
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _perform_ocr(self, image: Image.Image, config: str = '--oem 3 --psm 6') -> Tuple[str, float]:
        """
        Perform OCR with confidence scoring.
        
        Returns:
            Tuple of (text, average_confidence)
        """
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
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
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "", 0.0
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-process OCR text for construction document specifics.
        
        Handles common OCR errors in construction documents.
        """
        if not text:
            return text
        
        import re
        
        # Common construction abbreviation fixes
        replacements = {
            # Document types
            r'\bRF[Il1]\b': 'RFI',
            r'\bC[O0]\b': 'CO',
            r'\bAS[Il1]\b': 'ASI',
            r'\bPCO\b': 'PCO',
            r'\bCCD\b': 'CCD',
            r'\bSUBM[Il1]TTAL\b': 'SUBMITTAL',
            r'\b[Il1]NVOICE\b': 'INVOICE',
            r'\bP[O0]\b': 'PO',
            
            # Common OCR errors
            r'\b[O0]([0-9])\b': r'0\1',  # Fix O/0 confusion in numbers
            r'\b([0-9])[O0]\b': r'\g<1>0',
            r'\b[Il1]([0-9])\b': r'1\1',  # Fix I/l/1 confusion
            
            # Date fixes
            r'\b2[O0]2([0-9])\b': r'202\1',  # Common year OCR errors
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _generate_cache_key(self, page: fitz.Page, dpi: int) -> str:
        """
        Generate a unique cache key for a page.
        
        Based on:
        - Page content hash (text + images)
        - DPI setting
        - Language setting
        - OCR strategies
        """
        try:
            # Get page content for hashing
            page_text = page.get_text()
            page_images = page.get_images()
            
            # Create content hash
            content_data = {
                'text': page_text,
                'image_count': len(page_images),
                'image_info': [img[:4] for img in page_images[:5]],  # First 5 images, basic info only
                'dpi': dpi,
                'language': self.language,
                'strategies': self.strategies,
                'min_confidence': self.min_confidence
            }
            
            # Convert to string and hash
            content_str = str(sorted(content_data.items()))
            cache_key = hashlib.md5(content_str.encode()).hexdigest()
            
            return f"ocr_{cache_key}"
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to timestamp-based key (no caching benefit but won't crash)
            import time
            return f"ocr_fallback_{int(time.time() * 1000000)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get OCR cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            stats = {
                'cache_size_items': len(self.cache),
                'cache_size_bytes': self.cache.volume(),
                'cache_directory': str(self.cache.directory),
                'cache_hits': getattr(self.cache, 'hits', 'unknown'),
                'cache_misses': getattr(self.cache, 'misses', 'unknown')
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """
        Clear the OCR cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cache.clear()
            logger.info("OCR cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of available Tesseract languages."""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang not in ['osd', 'snum']]
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return ['eng']  # Default to English