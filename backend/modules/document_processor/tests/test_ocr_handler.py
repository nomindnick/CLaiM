"""Tests for OCR handler module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from backend.modules.document_processor.ocr_handler import OCRHandler
from backend.shared.exceptions import OCRError


class TestOCRHandler:
    """Test suite for OCR handler."""
    
    @pytest.fixture
    def ocr_handler(self):
        """Create OCR handler instance."""
        with patch('backend.modules.document_processor.ocr_handler.pytesseract.get_tesseract_version'):
            return OCRHandler()
    
    def test_init_default_params(self):
        """Test OCR handler initialization with default parameters."""
        with patch('backend.modules.document_processor.ocr_handler.pytesseract.get_tesseract_version'):
            handler = OCRHandler()
            assert handler.language == "eng"
            assert handler.min_confidence == 0.6
    
    def test_init_custom_params(self):
        """Test OCR handler initialization with custom parameters."""
        with patch('backend.modules.document_processor.ocr_handler.pytesseract.get_tesseract_version'):
            handler = OCRHandler(language="spa", min_confidence=0.8)
            assert handler.language == "spa"
            assert handler.min_confidence == 0.8
    
    def test_verify_tesseract_not_found(self):
        """Test error when Tesseract is not installed."""
        with patch('backend.modules.document_processor.ocr_handler.pytesseract.get_tesseract_version') as mock_version:
            mock_version.side_effect = Exception("Tesseract not found")
            
            with pytest.raises(OCRError) as exc_info:
                OCRHandler()
            
            assert "Tesseract is not installed" in str(exc_info.value)
    
    @patch('backend.modules.document_processor.ocr_handler.pytesseract.image_to_data')
    def test_perform_ocr_success(self, mock_ocr, ocr_handler):
        """Test successful OCR processing."""
        # Mock OCR result
        mock_ocr.return_value = {
            'text': ['Invoice', 'Number:', '12345', 'Date:', '2024-01-15'],
            'conf': [95, 92, 98, 90, 88],
        }
        
        # Create dummy image
        image = Image.new('L', (100, 100), color=255)
        
        text, confidence = ocr_handler._perform_ocr(image)
        
        assert text == "Invoice Number: 12345 Date: 2024-01-15"
        assert 0.8 < confidence < 1.0  # Average of confidences
    
    @patch('backend.modules.document_processor.ocr_handler.pytesseract.image_to_data')
    def test_perform_ocr_low_confidence(self, mock_ocr, ocr_handler, caplog):
        """Test OCR with low confidence results."""
        # Mock low confidence OCR result
        mock_ocr.return_value = {
            'text': ['fuzzy', 'text'],
            'conf': [45, 50],
        }
        
        image = Image.new('L', (100, 100), color=255)
        
        text, confidence = ocr_handler._perform_ocr(image)
        
        assert text == "fuzzy text"
        assert confidence < 0.6
        assert "Low OCR confidence" in caplog.text
    
    def test_post_process_text_construction_terms(self, ocr_handler):
        """Test post-processing fixes construction-specific terms."""
        # Test RFI variations
        assert ocr_handler._post_process_text("RF1 #123") == "RFI #123"
        assert ocr_handler._post_process_text("RFl document") == "RFI document"
        
        # Test Change Order variations
        assert ocr_handler._post_process_text("C0 approved") == "CO approved"
        
        # Test common word fixes
        assert ocr_handler._post_process_text("SUBM1TTAL package") == "SUBMITTAL package"
        assert ocr_handler._post_process_text("1NVOICE due") == "INVOICE due"
    
    def test_post_process_text_date_fixes(self, ocr_handler):
        """Test date OCR error corrections."""
        # Test date with O instead of 0
        assert ocr_handler._post_process_text("Date: O1/15/2024") == "Date: 01/15/2024"
        assert ocr_handler._post_process_text("Due: 12/3O/2024") == "Due: 12/30/2024"
        
        # Test date with l/I instead of 1
        assert ocr_handler._post_process_text("Date: 0l/15/2024") == "Date: 01/15/2024"
        assert ocr_handler._post_process_text("Date: I2/15/2024") == "Date: 12/15/2024"
    
    def test_post_process_text_whitespace(self, ocr_handler):
        """Test whitespace normalization."""
        text = "Multiple   spaces    and\n\nnewlines"
        result = ocr_handler._post_process_text(text)
        assert result == "Multiple spaces and newlines"
    
    def test_deskew_image(self, ocr_handler):
        """Test image deskewing."""
        # Create a simple test image
        image = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Add some content to detect skew
        image[40:60, 20:80] = 0  # Horizontal line
        
        result = ocr_handler._deskew_image(image)
        
        # Should return an image of same shape
        assert result.shape == image.shape
    
    @patch('backend.modules.document_processor.ocr_handler.pytesseract.get_languages')
    def test_get_supported_languages(self, mock_langs, ocr_handler):
        """Test getting supported languages."""
        mock_langs.return_value = ['eng', 'spa', 'fra', 'osd', 'snum']
        
        languages = ocr_handler.get_supported_languages()
        
        # Should filter out osd and snum
        assert 'eng' in languages
        assert 'spa' in languages
        assert 'fra' in languages
        assert 'osd' not in languages
        assert 'snum' not in languages
    
    def test_preprocess_image_grayscale_conversion(self, ocr_handler):
        """Test image preprocessing converts to grayscale."""
        # Create RGB image
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        
        with patch('cv2.fastNlMeansDenoising') as mock_denoise:
            with patch('cv2.adaptiveThreshold') as mock_threshold:
                with patch('cv2.morphologyEx') as mock_morph:
                    # Mock CV2 functions to return dummy arrays
                    mock_denoise.return_value = np.ones((100, 100), dtype=np.uint8)
                    mock_threshold.return_value = np.ones((100, 100), dtype=np.uint8)
                    mock_morph.return_value = np.ones((100, 100), dtype=np.uint8)
                    
                    result = ocr_handler._preprocess_image(image)
                    
                    # Should return a PIL Image
                    assert isinstance(result, Image.Image)
                    assert result.mode == 'L'  # Grayscale
    
    @patch('backend.modules.document_processor.ocr_handler.Image.open')
    def test_process_image_file_success(self, mock_open, ocr_handler):
        """Test processing standalone image file."""
        # Mock image
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        
        with patch.object(ocr_handler, '_preprocess_image') as mock_preprocess:
            with patch.object(ocr_handler, '_perform_ocr') as mock_perform:
                mock_preprocess.return_value = mock_image
                mock_perform.return_value = ("Test text", 0.95)
                
                text, confidence = ocr_handler.process_image_file("/path/to/image.png")
                
                assert text == "Test text"
                assert confidence == 0.95
                mock_open.assert_called_once_with("/path/to/image.png")
    
    @patch('backend.modules.document_processor.ocr_handler.Image.open')
    def test_process_image_file_error(self, mock_open, ocr_handler):
        """Test error handling for image file processing."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(OCRError) as exc_info:
            ocr_handler.process_image_file("/nonexistent/image.png")
        
        assert "Failed to process image" in str(exc_info.value)