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
        """Perform local classification using loaded model with rule-based input.
        
        Args:
            text: Document text
            features: Extracted features
            
        Returns:
            Classification result dictionary
        """
        try:
            # STEP 1: Always run rule-based classification first
            rule_based_result = self._fallback_classification(features)
            
            # STEP 2: Use AI model with rule-based context (no confidence threshold yet)
            model_result = self.model_manager.classify_text(
                text, 
                min_confidence=0.0,  # Don't apply threshold yet - ensemble will decide
                rule_based_suggestion=rule_based_result
            )
            
            # STEP 3: Create ensemble result combining both approaches
            ensemble_result = self._create_ensemble_result(
                ai_result=model_result,
                rule_result=rule_based_result,
                features=features
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Local classification failed: {e}")
            # Fallback to pure rule-based classification
            return self._fallback_classification(features)
    
    
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
                # EMAIL patterns get higher weight since they're structural
                weight = 0.6 if doc_type == DocumentType.EMAIL else 0.3
                scores[doc_type] = scores.get(doc_type, 0) + score * weight
        
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
    
    def _create_ensemble_result(
        self, 
        ai_result: Dict, 
        rule_result: Dict, 
        features: ClassificationFeatures
    ) -> Dict:
        """Create ensemble result combining AI and rule-based classifications.
        
        Args:
            ai_result: Result from AI model
            rule_result: Result from rule-based classification
            features: Extracted features
            
        Returns:
            Combined classification result
        """
        ai_type = ai_result["document_type"]
        ai_confidence = ai_result["confidence"]
        rule_type = rule_result["document_type"]
        rule_confidence = rule_result["confidence"]
        
        reasoning_parts = []
        
        # If both methods agree, boost confidence
        if ai_type == rule_type:
            final_confidence = min(ai_confidence * 1.2, 1.0)  # 20% boost
            final_type = ai_type
            reasoning_parts.append(f"AI and rules agree on {ai_type.value}")
            reasoning_parts.append(f"AI: {ai_confidence:.2f}, Rules: {rule_confidence:.2f}")
        
        # If they disagree, use weighted decision based on confidence
        else:
            # Give slight preference to rule-based for high-confidence rule matches
            if rule_confidence > 0.7 and ai_confidence < 0.6:
                final_type = rule_type
                final_confidence = rule_confidence * 0.9  # Slight penalty for disagreement
                reasoning_parts.append(f"Rules override AI: {rule_type.value} (high rule confidence)")
            
            # If AI is very confident, trust it
            elif ai_confidence > 0.8:
                final_type = ai_type
                final_confidence = ai_confidence * 0.9  # Slight penalty for disagreement
                reasoning_parts.append(f"AI override rules: {ai_type.value} (high AI confidence)")
            
            # Otherwise, weighted average based on confidence
            else:
                if ai_confidence >= rule_confidence:
                    final_type = ai_type
                    final_confidence = (ai_confidence * 0.7) + (rule_confidence * 0.3)
                else:
                    final_type = rule_type
                    final_confidence = (rule_confidence * 0.7) + (ai_confidence * 0.3)
                
                reasoning_parts.append(f"Weighted decision: {final_type.value}")
                reasoning_parts.append(f"AI suggested {ai_type.value} ({ai_confidence:.2f})")
                reasoning_parts.append(f"Rules suggested {rule_type.value} ({rule_confidence:.2f})")
        
        # Combine alternatives from both methods
        all_alternatives = {}
        for alt in ai_result.get("alternatives", []):
            all_alternatives[alt["label"]] = alt["score"] * 0.6  # Weight AI alternatives
        
        for alt in rule_result.get("alternatives", []):
            label = alt["label"]
            score = alt["score"] * 0.4  # Weight rule alternatives
            all_alternatives[label] = all_alternatives.get(label, 0) + score
        
        # Sort and take top 3
        final_alternatives = [
            {"label": label, "score": score}
            for label, score in sorted(all_alternatives.items(), key=lambda x: x[1], reverse=True)
        ][:3]
        
        return {
            "document_type": final_type,
            "confidence": final_confidence,
            "alternatives": final_alternatives,
            "reasoning": " | ".join(reasoning_parts)
        }
    
    def get_model_status(self):
        """Get current model status."""
        return self.model_manager.get_status()
    
    def preload_model(self) -> bool:
        """Preload the classification model."""
        return self.model_manager.load_model()


# Global classifier instance
document_classifier = DocumentClassifier()