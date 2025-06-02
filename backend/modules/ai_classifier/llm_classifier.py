"""LLM-based document classifier for construction litigation documents.

This module replaces the existing rule-based + DistilBERT ensemble with a pure
LLM approach for dramatically improved accuracy on real-world documents.
"""

import time
import re
from typing import Dict, List, Optional, Any
from loguru import logger

from .models import (
    ClassificationResult, 
    ClassificationRequest, 
    ClassificationFeatures
)
from ..document_processor.models import DocumentType
from ..llm_client.router import LLMRouter, PrivacyMode
from ..llm_client.base_client import LLMRequest, LLMTaskType, LLMError, LLMServiceUnavailable
from ..llm_client.prompt_templates import PromptTemplates


class LLMDocumentClassifier:
    """LLM-based document classifier for construction litigation documents."""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "llama3:8b-instruct-q5_K_M",
                 openai_model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None,
                 default_privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE):
        """Initialize LLM classifier.
        
        Args:
            ollama_host: Ollama service host
            ollama_model: Ollama model name
            openai_model: OpenAI model name  
            openai_api_key: OpenAI API key
            default_privacy_mode: Default privacy mode for classification
        """
        self.router = LLMRouter(
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            openai_model=openai_model,
            openai_api_key=openai_api_key
        )
        self.default_privacy_mode = default_privacy_mode
        
        # Construction document categories for LLM
        self.document_categories = [
            "Email",
            "Contract Document",
            "Change Order", 
            "Payment Application",
            "Inspection Report",
            "Plans and Specifications",
            "Meeting Minutes",
            "Request for Information (RFI)",
            "Submittal",
            "Daily Report",
            "Invoice",
            "Letter",
            "Cost Proposal",
            "Schedule of Values",
            "Other"
        ]
        
        # Mapping from LLM categories to DocumentType enum
        self.category_mapping = {
            "Email": DocumentType.EMAIL,
            "Contract Document": DocumentType.CONTRACT,
            "Change Order": DocumentType.CHANGE_ORDER,
            "Payment Application": DocumentType.PAYMENT_APPLICATION,
            "Inspection Report": DocumentType.INSPECTION_REPORT,
            "Plans and Specifications": DocumentType.PLANS_SPECIFICATIONS,
            "Meeting Minutes": DocumentType.MEETING_MINUTES,
            "Request for Information (RFI)": DocumentType.RFI,
            "Submittal": DocumentType.SUBMITTAL,
            "Daily Report": DocumentType.DAILY_REPORT,
            "Invoice": DocumentType.INVOICE,
            "Letter": DocumentType.LETTER,
            "Cost Proposal": DocumentType.CHANGE_ORDER,  # Map to change order for now
            "Schedule of Values": DocumentType.PAYMENT_APPLICATION,  # Map to payment app
            "Other": DocumentType.OTHER
        }
        
        # Chunking parameters for large documents
        self.max_chunk_tokens = 6000  # Leave room for prompt
        self.chunk_overlap_tokens = 200
        
    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify a document using LLM.
        
        Args:
            request: Classification request with text and options
            
        Returns:
            Classification result with type, confidence, and features
        """
        start_time = time.time()
        
        try:
            # Extract basic features for compatibility
            features = self._extract_basic_features(request.text, request.title)
            
            # Determine privacy mode
            privacy_mode = self._determine_privacy_mode(request.text)
            
            # Check if document needs chunking
            if self._needs_chunking(request.text):
                logger.info("Document requires chunking for LLM processing")
                classification_result = self._classify_chunked_document(
                    request.text, 
                    request.title,
                    privacy_mode,
                    request.require_reasoning
                )
            else:
                # Single classification
                classification_result = self._classify_single_text(
                    request.text,
                    request.title, 
                    privacy_mode,
                    request.require_reasoning
                )
            
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
                f"LLM classified document as {result.document_type.value} "
                f"(confidence: {result.confidence:.3f}, time: {result.processing_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            
            # Fallback to legacy classifier
            try:
                from .legacy_classifier import document_classifier as legacy_classifier
                logger.warning("Falling back to legacy classifier")
                return legacy_classifier.classify(request)
            except Exception as fallback_error:
                logger.error(f"Legacy fallback also failed: {fallback_error}")
                
                # Ultimate fallback
                return ClassificationResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                    features=self._extract_basic_features(request.text, request.title),
                    reasoning=f"Classification error: {str(e)}" if request.require_reasoning else None,
                    processing_time=time.time() - start_time
                )
    
    def _classify_single_text(self, 
                             text: str, 
                             title: Optional[str],
                             privacy_mode: PrivacyMode,
                             require_reasoning: bool) -> Dict[str, Any]:
        """Classify a single text using LLM.
        
        Args:
            text: Document text to classify
            title: Optional document title
            privacy_mode: Privacy mode for processing
            require_reasoning: Whether to include reasoning
            
        Returns:
            Classification result dictionary
        """
        try:
            # Use router's convenience method for classification
            response = self.router.classify_document(
                text=text,
                categories=self.document_categories,
                context=title,
                privacy_mode=privacy_mode
            )
            
            # Parse LLM response
            parsed = PromptTemplates.parse_classification_response(response.content)
            
            if not parsed:
                raise LLMError("Failed to parse LLM classification response")
            
            # Map category to DocumentType
            category = parsed["document_type"]
            document_type = self.category_mapping.get(category, DocumentType.OTHER)
            confidence = parsed["confidence"]  # Already converted to 0-1 range in parser
            
            # Build alternatives (for now, just the main result)
            alternatives = [{"label": document_type.value, "score": confidence}]
            
            reasoning = None
            if require_reasoning:
                reasoning = f"LLM Classification: {category} ({confidence*100:.0f}%) - {parsed['reasoning']}"
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "alternatives": alternatives,
                "reasoning": reasoning
            }
            
        except (LLMError, LLMServiceUnavailable) as e:
            logger.error(f"LLM service error: {e}")
            raise
        except Exception as e:
            logger.error(f"Classification processing error: {e}")
            raise LLMError(f"Classification failed: {str(e)}")
    
    def _classify_chunked_document(self,
                                  text: str,
                                  title: Optional[str],
                                  privacy_mode: PrivacyMode,
                                  require_reasoning: bool) -> Dict[str, Any]:
        """Classify a large document by chunking and using majority voting.
        
        Args:
            text: Document text to classify
            title: Optional document title
            privacy_mode: Privacy mode for processing
            require_reasoning: Whether to include reasoning
            
        Returns:
            Classification result dictionary
        """
        logger.info("Classifying large document using chunking strategy")
        
        # Split document into chunks
        chunks = self._chunk_document(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Classify each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"Classifying chunk {i+1}/{len(chunks)}")
                result = self._classify_single_text(
                    chunk, 
                    title, 
                    privacy_mode, 
                    require_reasoning=False  # Skip reasoning for chunks
                )
                chunk_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to classify chunk {i+1}: {e}")
                continue
        
        if not chunk_results:
            raise LLMError("Failed to classify any chunks")
        
        # Aggregate results using majority voting with confidence weighting
        return self._aggregate_chunk_results(chunk_results, require_reasoning)
    
    def _chunk_document(self, text: str) -> List[str]:
        """Split document into chunks for LLM processing.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple chunking by sentences with overlap
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Rough token estimation (4 chars per token)
            if len(current_chunk + sentence) * 0.25 > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk.split('. ')[-5:]  # Last 5 sentences
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                else:
                    # Single sentence is too long, truncate
                    max_chars = self.max_chunk_tokens * 4
                    chunks.append(sentence[:max_chars])
                    current_chunk = ""
            else:
                current_chunk += sentence + '. '
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _aggregate_chunk_results(self, 
                                chunk_results: List[Dict[str, Any]], 
                                require_reasoning: bool) -> Dict[str, Any]:
        """Aggregate classification results from multiple chunks.
        
        Args:
            chunk_results: List of classification results from chunks
            require_reasoning: Whether to include reasoning
            
        Returns:
            Aggregated classification result
        """
        # Count votes weighted by confidence
        vote_scores = {}
        total_confidence = 0.0
        
        for result in chunk_results:
            doc_type = result["document_type"]
            confidence = result["confidence"]
            
            if doc_type not in vote_scores:
                vote_scores[doc_type] = 0.0
            
            vote_scores[doc_type] += confidence
            total_confidence += confidence
        
        # Normalize scores
        if total_confidence > 0:
            for doc_type in vote_scores:
                vote_scores[doc_type] /= len(chunk_results)
        
        # Find winner
        if vote_scores:
            winner_type = max(vote_scores, key=vote_scores.get)
            winner_confidence = vote_scores[winner_type]
            
            # Build alternatives
            alternatives = [
                {"label": doc_type.value, "score": score}
                for doc_type, score in sorted(vote_scores.items(), 
                                            key=lambda x: x[1], reverse=True)
            ][:3]
            
            reasoning = None
            if require_reasoning:
                reasoning = f"Chunk aggregation: {len(chunk_results)} chunks, winner: {winner_type.value} (avg confidence: {winner_confidence:.3f})"
            
            return {
                "document_type": winner_type,
                "confidence": winner_confidence,
                "alternatives": alternatives,
                "reasoning": reasoning
            }
        else:
            # No valid results
            return {
                "document_type": DocumentType.UNKNOWN,
                "confidence": 0.0,
                "alternatives": [],
                "reasoning": "No valid classification from chunks" if require_reasoning else None
            }
    
    def _needs_chunking(self, text: str) -> bool:
        """Check if document needs chunking based on size.
        
        Args:
            text: Document text
            
        Returns:
            True if chunking is needed
        """
        # Rough token estimation (4 characters per token)
        estimated_tokens = len(text) / 4
        return estimated_tokens > self.max_chunk_tokens
    
    def _determine_privacy_mode(self, text: str) -> PrivacyMode:
        """Determine appropriate privacy mode for text.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Privacy mode to use
        """
        # Use router's sensitive content detection
        has_sensitive = self.router.has_sensitive_content(text)
        
        if has_sensitive:
            logger.debug("Sensitive content detected, using local processing")
            return PrivacyMode.FULL_LOCAL
        else:
            return self.default_privacy_mode
    
    def _extract_basic_features(self, text: str, title: Optional[str] = None) -> ClassificationFeatures:
        """Extract basic features for compatibility with existing system.
        
        Args:
            text: Document text
            title: Optional document title
            
        Returns:
            Basic classification features
        """
        # Simple feature extraction for compatibility
        text_length = len(text)
        word_count = len(text.split())
        
        # Basic pattern detection
        has_numbers = bool(re.search(r'\d+', text))
        has_amounts = bool(re.search(r'\$[\d,]+\.?\d*', text))
        has_dates = bool(re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text))
        
        return ClassificationFeatures(
            text_length=text_length,
            word_count=word_count,
            has_numbers=has_numbers,
            has_amounts=has_amounts,
            has_dates=has_dates,
            has_signature_area=False,  # Not implemented for LLM version
            has_forms=False,  # Not implemented for LLM version
            has_tables=False,  # Not implemented for LLM version
            subject_line=title,
            reference_numbers=[],  # Not implemented for LLM version
            key_phrases=[]  # Not implemented for LLM version
        )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current LLM model status.
        
        Returns:
            Dictionary with model status information
        """
        return self.router.get_status()
    
    def preload_model(self) -> bool:
        """Preload the LLM model.
        
        Returns:
            True if model is available
        """
        status = self.router.get_status()
        return status["clients"]["ollama"] or status["clients"]["openai"]
    
    def is_available(self) -> bool:
        """Check if LLM classification is available.
        
        Returns:
            True if at least one LLM client is available
        """
        status = self.router.get_status()
        return status["clients"]["ollama"] or status["clients"]["openai"]


# Global LLM classifier instance
llm_document_classifier = LLMDocumentClassifier()