"""Main document classifier with feature extraction and classification logic."""

import re
import time
from typing import List, Dict, Optional
import logging

from .models import (
    ClassificationResult, 
    ClassificationRequest, 
    ClassificationFeatures
)
from .model_manager import model_manager
from ..document_processor.models import DocumentType
from api.privacy_manager import privacy_manager, PrivacyContext

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Main document classifier for construction litigation documents."""
    
    def __init__(self):
        self.model_manager = model_manager
        
        # Pattern definitions for feature extraction
        self._amount_pattern = re.compile(r'\$[\d,]+\.?\d*')
        self._date_patterns = [
            re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}'),
            re.compile(r'\d{1,2}-\d{1,2}-\d{2,4}'),
            re.compile(r'\b\w+\s+\d{1,2},?\s+\d{4}\b'),
        ]
        self._reference_patterns = {
            'rfi': re.compile(r'\brfi\s*#?\s*(\d+)', re.IGNORECASE),
            'change_order': re.compile(r'\b(?:co|change\s*order)\s*#?\s*(\d+)', re.IGNORECASE),
            'invoice': re.compile(r'\binvoice\s*#?\s*(\d+)', re.IGNORECASE),
            'contract': re.compile(r'\bcontract\s*#?\s*(\d+)', re.IGNORECASE),
        }
        
        # Key phrases for document types
        self._key_phrases = {
            DocumentType.EMAIL: [
                'from:', 'to:', 'subject:', 'sent:', 'dear', 'sincerely',
                'regards', 'best regards', 'thank you'
            ],
            DocumentType.RFI: [
                'request for information', 'clarification', 'please provide',
                'response required', 'submittal', 'specification'
            ],
            DocumentType.CHANGE_ORDER: [
                'change order', 'modification', 'amendment', 'additional work',
                'scope change', 'cost adjustment', 'time extension'
            ],
            DocumentType.INVOICE: [
                'invoice', 'billing', 'payment', 'amount due', 'remittance',
                'pay', 'total amount', 'balance due'
            ],
            DocumentType.DAILY_REPORT: [
                'daily report', 'field report', 'weather conditions',
                'crew size', 'equipment', 'progress', 'work performed'
            ],
            DocumentType.CONTRACT: [
                'contract', 'agreement', 'terms and conditions',
                'contractor', 'owner', 'shall', 'whereas'
            ],
            DocumentType.LETTER: [
                'letterhead', 'correspondence', 'formal letter',
                'official communication'
            ],
            DocumentType.MEETING_MINUTES: [
                'meeting minutes', 'attendees', 'agenda', 'action items',
                'discussed', 'meeting notes', 'decisions made'
            ]
        }
    
    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify a document based on its text content.
        
        Args:
            request: Classification request with text and options
            
        Returns:
            Classification result with type, confidence, and features
        """
        start_time = time.time()
        
        try:
            # Extract features from the document
            features = self._extract_features(request.text, request.title)
            
            # Check privacy mode and classify accordingly
            privacy_context = PrivacyContext(
                mode=privacy_manager.get_mode(),
                has_sensitive_content=privacy_manager.check_sensitive_content(request.text),
                operation_type="classification"
            )
            
            # Perform classification
            if privacy_manager.should_use_local(privacy_context):
                classification_result = self._classify_local(request.text, features)
            else:
                # For now, always use local - cloud classification not implemented
                classification_result = self._classify_local(request.text, features)
            
            # Build final result
            result = ClassificationResult(
                document_type=classification_result["document_type"],
                confidence=classification_result["confidence"],
                alternatives=classification_result.get("alternatives", []),
                features=features,
                reasoning=classification_result.get("reasoning") if request.require_reasoning else None,
                processing_time=time.time() - start_time
            )
            
            logger.info(
                f"Classified document as {result.document_type.value} "
                f"(confidence: {result.confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                features=self._extract_features(request.text, request.title),
                reasoning=f"Classification error: {str(e)}" if request.require_reasoning else None,
                processing_time=time.time() - start_time
            )
    
    def _extract_features(self, text: str, title: Optional[str] = None) -> ClassificationFeatures:
        """Extract features from document text for classification.
        
        Args:
            text: Document text
            title: Optional document title
            
        Returns:
            Extracted features
        """
        text_lower = text.lower()
        
        # Basic text statistics
        text_length = len(text)
        word_count = len(text.split())
        
        # Pattern-based features
        has_amounts = bool(self._amount_pattern.search(text))
        has_dates = any(pattern.search(text) for pattern in self._date_patterns)
        has_numbers = bool(re.search(r'\d+', text))
        
        # Structural features
        has_signature_area = any(phrase in text_lower for phrase in [
            'signature', 'signed', 'authorized by', 'approved by'
        ])
        has_forms = any(phrase in text_lower for phrase in [
            'form', 'application', 'check one', '☐', '□', 'yes/no'
        ])
        has_tables = any(phrase in text_lower for phrase in [
            'table', '|', '\t\t', 'column', 'row'
        ]) or text.count('\n') > 10
        
        # Extract reference numbers
        reference_numbers = []
        for ref_type, pattern in self._reference_patterns.items():
            matches = pattern.findall(text)
            reference_numbers.extend([f"{ref_type}_{match}" for match in matches])
        
        # Extract key phrases
        key_phrases = []
        for doc_type, phrases in self._key_phrases.items():
            found_phrases = [phrase for phrase in phrases if phrase in text_lower]
            key_phrases.extend(found_phrases)
        
        return ClassificationFeatures(
            text_length=text_length,
            word_count=word_count,
            has_numbers=has_numbers,
            has_amounts=has_amounts,
            has_dates=has_dates,
            has_signature_area=has_signature_area,
            has_forms=has_forms,
            has_tables=has_tables,
            subject_line=title,
            reference_numbers=reference_numbers[:10],  # Limit to first 10
            key_phrases=key_phrases[:20]  # Limit to first 20
        )
    
    def _classify_local(self, text: str, features: ClassificationFeatures) -> Dict:
        """Perform local classification using loaded model.
        
        Args:
            text: Document text
            features: Extracted features
            
        Returns:
            Classification result dictionary
        """
        try:
            # Use model manager for classification
            model_result = self.model_manager.classify_text(text)
            
            # Enhance with feature-based adjustments
            enhanced_result = self._enhance_with_features(model_result, features)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Local classification failed: {e}")
            # Fallback to pure rule-based classification
            return self._fallback_classification(features)
    
    def _enhance_with_features(self, model_result: Dict, features: ClassificationFeatures) -> Dict:
        """Enhance model results with feature-based adjustments.
        
        Args:
            model_result: Result from model classification
            features: Extracted features
            
        Returns:
            Enhanced classification result
        """
        doc_type = model_result["document_type"]
        confidence = model_result["confidence"]
        
        # Feature-based confidence adjustments
        confidence_adjustment = 0.0
        reasoning_parts = [model_result.get("reasoning", "")]
        
        # Reference number boost
        if features.reference_numbers:
            ref_types = [ref.split('_')[0] for ref in features.reference_numbers]
            if doc_type.value in ref_types or any(
                rt in doc_type.value for rt in ref_types
            ):
                confidence_adjustment += 0.1
                reasoning_parts.append(f"Found matching reference numbers: {features.reference_numbers[:3]}")
        
        # Structural feature boost
        if doc_type == DocumentType.INVOICE and features.has_amounts:
            confidence_adjustment += 0.05
            reasoning_parts.append("Contains monetary amounts (invoice indicator)")
        
        if doc_type == DocumentType.CONTRACT and features.has_signature_area:
            confidence_adjustment += 0.05
            reasoning_parts.append("Contains signature area (contract indicator)")
        
        if doc_type == DocumentType.DAILY_REPORT and features.has_dates:
            confidence_adjustment += 0.05
            reasoning_parts.append("Contains dates (daily report indicator)")
        
        # Key phrase boost
        doc_phrases = self._key_phrases.get(doc_type, [])
        matching_phrases = [p for p in features.key_phrases if p in doc_phrases]
        if matching_phrases:
            phrase_boost = min(len(matching_phrases) * 0.02, 0.1)
            confidence_adjustment += phrase_boost
            reasoning_parts.append(f"Found key phrases: {matching_phrases[:3]}")
        
        # Apply adjustment
        final_confidence = min(confidence + confidence_adjustment, 1.0)
        
        return {
            "document_type": doc_type,
            "confidence": final_confidence,
            "alternatives": model_result.get("alternatives", []),
            "reasoning": " | ".join(filter(None, reasoning_parts))
        }
    
    def _fallback_classification(self, features: ClassificationFeatures) -> Dict:
        """Pure rule-based classification fallback.
        
        Args:
            features: Extracted features
            
        Returns:
            Classification result dictionary
        """
        scores = {}
        reasoning_parts = []
        
        # Score based on reference numbers
        for ref in features.reference_numbers:
            ref_type = ref.split('_')[0]
            if ref_type == 'rfi':
                scores[DocumentType.RFI] = scores.get(DocumentType.RFI, 0) + 0.4
            elif ref_type == 'change_order':
                scores[DocumentType.CHANGE_ORDER] = scores.get(DocumentType.CHANGE_ORDER, 0) + 0.4
            elif ref_type == 'invoice':
                scores[DocumentType.INVOICE] = scores.get(DocumentType.INVOICE, 0) + 0.4
            elif ref_type == 'contract':
                scores[DocumentType.CONTRACT] = scores.get(DocumentType.CONTRACT, 0) + 0.4
        
        # Score based on key phrases
        for doc_type, phrases in self._key_phrases.items():
            phrase_matches = [p for p in features.key_phrases if p in phrases]
            if phrase_matches:
                score = len(phrase_matches) / len(phrases)
                scores[doc_type] = scores.get(doc_type, 0) + score * 0.3
        
        # Structural scoring
        if features.has_amounts:
            scores[DocumentType.INVOICE] = scores.get(DocumentType.INVOICE, 0) + 0.2
            scores[DocumentType.CHANGE_ORDER] = scores.get(DocumentType.CHANGE_ORDER, 0) + 0.1
        
        if features.has_signature_area:
            scores[DocumentType.CONTRACT] = scores.get(DocumentType.CONTRACT, 0) + 0.15
            scores[DocumentType.CHANGE_ORDER] = scores.get(DocumentType.CHANGE_ORDER, 0) + 0.1
        
        if features.has_forms:
            scores[DocumentType.RFI] = scores.get(DocumentType.RFI, 0) + 0.1
            scores[DocumentType.SUBMITTAL] = scores.get(DocumentType.SUBMITTAL, 0) + 0.1
        
        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            alternatives = [
                {"label": dtype.value, "score": score}
                for dtype, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ][:3]
            
            reasoning = f"Fallback rule-based classification (score: {best_score:.2f})"
        else:
            best_type = DocumentType.UNKNOWN
            best_score = 0.0
            alternatives = []
            reasoning = "No classification patterns matched"
        
        return {
            "document_type": best_type,
            "confidence": min(best_score, 1.0),
            "alternatives": alternatives,
            "reasoning": reasoning
        }
    
    def get_model_status(self):
        """Get current model status."""
        return self.model_manager.get_status()
    
    def preload_model(self) -> bool:
        """Preload the classification model."""
        return self.model_manager.load_model()


# Global classifier instance
document_classifier = DocumentClassifier()