"""Tests for the document classifier."""

import pytest
from unittest.mock import Mock, patch

from ..classifier import DocumentClassifier
from ..models import ClassificationRequest, DocumentType


class TestDocumentClassifier:
    """Tests for DocumentClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = DocumentClassifier()
    
    def test_feature_extraction_email(self):
        """Test feature extraction for email documents."""
        email_text = """
        From: john@contractor.com
        To: mary@owner.com
        Subject: Project Update
        
        Dear Mary,
        
        Please find attached the latest progress report.
        
        Best regards,
        John
        """
        
        features = self.classifier._extract_features(email_text)
        
        assert features.word_count > 0
        assert features.text_length > 0
        assert "from:" in features.key_phrases
        assert "dear" in features.key_phrases
        assert "best regards" in features.key_phrases
    
    def test_feature_extraction_rfi(self):
        """Test feature extraction for RFI documents."""
        rfi_text = """
        REQUEST FOR INFORMATION
        RFI #123
        
        Project: School Construction
        Date: 05/30/2025
        
        Please provide clarification on the following specification requirements:
        1. Material specifications for exterior walls
        2. Required fire ratings
        
        Response required by: 06/05/2025
        """
        
        features = self.classifier._extract_features(rfi_text)
        
        assert "request for information" in features.key_phrases
        assert "clarification" in features.key_phrases
        assert "response required" in features.key_phrases
        assert any("rfi_123" in ref for ref in features.reference_numbers)
        assert features.has_dates
        assert features.has_numbers
    
    def test_feature_extraction_invoice(self):
        """Test feature extraction for invoice documents."""
        invoice_text = """
        INVOICE #INV-001
        
        Bill To: ABC School District
        Amount Due: $15,000.00
        Payment Terms: Net 30
        
        Description: Concrete work Phase 1
        Total Amount: $15,000.00
        
        Please remit payment by 06/30/2025
        """
        
        features = self.classifier._extract_features(invoice_text)
        
        assert "invoice" in features.key_phrases
        assert "payment" in features.key_phrases
        assert "amount due" in features.key_phrases
        assert features.has_amounts
        assert features.has_dates
        assert any("invoice_001" in ref for ref in features.reference_numbers)
    
    def test_feature_extraction_change_order(self):
        """Test feature extraction for change order documents."""
        co_text = """
        CHANGE ORDER #007
        
        Project: Elementary School Renovation
        Original Contract Amount: $500,000.00
        This Change: +$25,000.00
        New Contract Amount: $525,000.00
        
        Description: Additional electrical work required due to scope change.
        Time Extension: 5 days
        
        Contractor Signature: ________________
        Owner Signature: ________________
        """
        
        features = self.classifier._extract_features(co_text)
        
        assert "change order" in features.key_phrases
        assert "additional work" in features.key_phrases
        assert "scope change" in features.key_phrases
        assert features.has_amounts
        assert features.has_signature_area
        assert any("change_order_007" in ref for ref in features.reference_numbers)
    
    def test_fallback_classification_email(self):
        """Test fallback classification for email."""
        email_features = self.classifier._extract_features("""
            From: test@example.com
            To: recipient@example.com
            Subject: Test Email
            Dear Recipient,
            Thank you for your email.
            Best regards,
            Sender
        """)
        
        result = self.classifier._fallback_classification(email_features)
        
        assert result["document_type"] == DocumentType.EMAIL
        assert result["confidence"] > 0
        assert "rule-based" in result["reasoning"]
    
    def test_fallback_classification_rfi(self):
        """Test fallback classification for RFI."""
        rfi_features = self.classifier._extract_features("""
            REQUEST FOR INFORMATION
            RFI #456
            Please provide clarification on the specification requirements.
            Response required by next week.
        """)
        
        result = self.classifier._fallback_classification(rfi_features)
        
        assert result["document_type"] == DocumentType.RFI
        assert result["confidence"] > 0
    
    def test_fallback_classification_unknown(self):
        """Test fallback classification for unknown document."""
        unknown_features = self.classifier._extract_features("Random text with no patterns.")
        
        result = self.classifier._fallback_classification(unknown_features)
        
        assert result["document_type"] == DocumentType.UNKNOWN
        assert result["confidence"] == 0.0
    
    @patch('backend.modules.ai_classifier.classifier.model_manager')
    def test_classify_with_model_success(self, mock_model_manager):
        """Test classification with successful model inference."""
        # Mock model manager response
        mock_model_manager.classify_text.return_value = {
            "document_type": DocumentType.EMAIL,
            "confidence": 0.85,
            "alternatives": [{"label": "email", "score": 0.85}],
            "reasoning": "Model classification"
        }
        
        request = ClassificationRequest(
            text="From: test@example.com\nTo: recipient@example.com\nSubject: Test",
            require_reasoning=True
        )
        
        result = self.classifier.classify(request)
        
        assert result.document_type == DocumentType.EMAIL
        assert result.confidence > 0.8
        assert result.reasoning is not None
        assert result.processing_time > 0
    
    @patch('backend.modules.ai_classifier.classifier.model_manager')
    def test_classify_with_model_failure(self, mock_model_manager):
        """Test classification with model failure fallback."""
        # Mock model manager to raise exception
        mock_model_manager.classify_text.side_effect = Exception("Model error")
        
        request = ClassificationRequest(
            text="RFI #123\nRequest for Information\nPlease provide clarification.",
            require_reasoning=True
        )
        
        result = self.classifier.classify(request)
        
        # Should fallback to rule-based classification
        assert result.document_type == DocumentType.RFI
        assert result.confidence > 0
        assert "error" in result.reasoning.lower()
    
    def test_enhance_with_features_reference_boost(self):
        """Test confidence enhancement with matching reference numbers."""
        mock_result = {
            "document_type": DocumentType.RFI,
            "confidence": 0.6,
            "alternatives": [],
            "reasoning": "Base classification"
        }
        
        features = self.classifier._extract_features("RFI #123\nRequest for information")
        
        enhanced = self.classifier._enhance_with_features(mock_result, features)
        
        # Should boost confidence due to matching RFI reference
        assert enhanced["confidence"] > 0.6
        assert "reference numbers" in enhanced["reasoning"]
    
    def test_enhance_with_features_amount_boost(self):
        """Test confidence enhancement for invoice with amounts."""
        mock_result = {
            "document_type": DocumentType.INVOICE,
            "confidence": 0.7,
            "alternatives": [],
            "reasoning": "Base classification"
        }
        
        features = self.classifier._extract_features("Invoice #001\nAmount Due: $1,000.00")
        
        enhanced = self.classifier._enhance_with_features(mock_result, features)
        
        # Should boost confidence due to monetary amounts
        assert enhanced["confidence"] > 0.7
        assert "amounts" in enhanced["reasoning"]
    
    def test_classify_empty_text(self):
        """Test classification with empty text."""
        request = ClassificationRequest(text="")
        
        result = self.classifier.classify(request)
        
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0
    
    def test_classify_minimum_confidence_threshold(self):
        """Test minimum confidence threshold enforcement."""
        request = ClassificationRequest(
            text="Some ambiguous text",
            min_confidence=0.8
        )
        
        result = self.classifier.classify(request)
        
        # If confidence is below threshold, should return UNKNOWN
        if result.confidence < 0.8:
            assert result.document_type == DocumentType.UNKNOWN
    
    def test_pattern_matching_case_insensitive(self):
        """Test that pattern matching is case insensitive."""
        features = self.classifier._extract_features("INVOICE #123\nAMOUNT DUE: $500")
        
        assert "invoice" in features.key_phrases
        assert features.has_amounts
        assert any("invoice_123" in ref for ref in features.reference_numbers)