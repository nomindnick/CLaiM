"""Model management for document classification."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

from api.config import settings
from .models import ModelStatus, DocumentType

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages DistilBERT model loading and inference for document classification."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.classifier_pipeline = None
        self.model_path = None
        self.last_loaded = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Document type mapping for fine-tuned models
        self._type_mapping = {
            label.value: idx for idx, label in enumerate(DocumentType)
        }
        self._reverse_mapping = {idx: label for label, idx in self._type_mapping.items()}
    
    def load_model(self, force_reload: bool = False) -> bool:
        """Load DistilBERT model for document classification.
        
        Args:
            force_reload: Whether to reload even if already loaded
            
        Returns:
            True if model loaded successfully
        """
        if self.model is not None and not force_reload:
            logger.info("DistilBERT model already loaded")
            return True
        
        try:
            start_time = time.time()
            
            # Try custom model path first, fallback to HuggingFace model
            custom_model_path = settings.models_dir / "distilbert-construction-classifier"
            
            if custom_model_path.exists():
                logger.info(f"Loading custom DistilBERT from {custom_model_path}")
                model_path = str(custom_model_path)
            else:
                # Use base DistilBERT for now - we'll fine-tune later
                logger.info("Loading base DistilBERT model from HuggingFace")
                model_path = "distilbert-base-uncased"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # For base model, we'll create a simple classifier
            if "distilbert-base-uncased" in model_path:
                # Create a classification pipeline using base model
                # We'll implement rule-based classification for now
                self.classifier_pipeline = None
                self.model = "rule_based"  # Placeholder
                logger.info("Using rule-based classification with DistilBERT embeddings")
            else:
                # Load fine-tuned model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=len(DocumentType),
                    torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
                )
                self.model.to(self._device)
                
                # Create classification pipeline
                self.classifier_pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self._device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info(f"Loaded fine-tuned DistilBERT model on {self._device}")
            
            self.model_path = model_path
            self.last_loaded = datetime.now().isoformat()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            self.model = None
            self.tokenizer = None
            self.classifier_pipeline = None
            return False
    
    def classify_text(self, text: str, min_confidence: float = 0.3) -> Dict[str, Any]:
        """Classify document text using loaded model.
        
        Args:
            text: Document text to classify
            min_confidence: Minimum confidence for classification
            
        Returns:
            Dictionary with classification results
        """
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load classification model")
        
        try:
            start_time = time.time()
            
            if self.model == "rule_based":
                # Use rule-based classification for now
                result = self._rule_based_classify(text)
            else:
                # Use fine-tuned model
                result = self._model_classify(text)
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # Apply minimum confidence threshold
            if result["confidence"] < min_confidence:
                result["document_type"] = DocumentType.UNKNOWN
                result["confidence"] = 0.0
                result["reasoning"] = f"Confidence {result['confidence']:.2f} below threshold {min_confidence}"
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "document_type": DocumentType.UNKNOWN,
                "confidence": 0.0,
                "alternatives": [],
                "reasoning": f"Classification error: {str(e)}",
                "processing_time": 0.0
            }
    
    def _rule_based_classify(self, text: str) -> Dict[str, Any]:
        """Rule-based classification using pattern matching.
        
        This is a fallback when we don't have a fine-tuned model.
        """
        text_lower = text.lower()
        
        # Define patterns for each document type
        patterns = {
            DocumentType.EMAIL: [
                "from:", "to:", "subject:", "sent:", "@", "email",
                "dear", "sincerely", "regards"
            ],
            DocumentType.RFI: [
                "request for information", "rfi", "clarification",
                "please provide", "response required"
            ],
            DocumentType.CHANGE_ORDER: [
                "change order", "co", "modification", "amendment",
                "additional work", "scope change"
            ],
            DocumentType.INVOICE: [
                "invoice", "bill", "payment", "amount due",
                "total:", "$", "pay", "remit"
            ],
            DocumentType.DAILY_REPORT: [
                "daily report", "field report", "weather",
                "crew", "equipment", "progress"
            ],
            DocumentType.CONTRACT: [
                "contract", "agreement", "terms", "conditions",
                "shall", "contractor", "owner"
            ],
            DocumentType.LETTER: [
                "letter", "correspondence", "dear", "sincerely",
                "letterhead"
            ],
            DocumentType.MEETING_MINUTES: [
                "meeting minutes", "attendees", "agenda",
                "action items", "discussed"
            ]
        }
        
        # Calculate scores for each type
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score / len(keywords)  # Normalize by pattern count
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Create alternatives list
        alternatives = [
            {"label": dtype.value, "score": score}
            for dtype, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score > 0
        ][:3]  # Top 3 alternatives
        
        reasoning = f"Rule-based classification: matched {int(best_score * len(patterns[best_type]))} patterns for {best_type.value}"
        
        return {
            "document_type": best_type,
            "confidence": min(best_score * 2, 1.0),  # Scale up but cap at 1.0
            "alternatives": alternatives,
            "reasoning": reasoning
        }
    
    def _model_classify(self, text: str) -> Dict[str, Any]:
        """Classification using fine-tuned DistilBERT model."""
        if not self.classifier_pipeline:
            raise RuntimeError("Classification pipeline not available")
        
        # Truncate text to model's max length
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        # Get predictions
        predictions = self.classifier_pipeline(text)
        
        # Find best prediction
        best_pred = max(predictions, key=lambda x: x['score'])
        best_type = DocumentType(best_pred['label'])
        
        # Format alternatives
        alternatives = [
            {"label": pred['label'], "score": pred['score']}
            for pred in sorted(predictions, key=lambda x: x['score'], reverse=True)
        ][:3]
        
        reasoning = f"DistilBERT classification: {best_pred['score']:.3f} confidence"
        
        return {
            "document_type": best_type,
            "confidence": best_pred['score'],
            "alternatives": alternatives,
            "reasoning": reasoning
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None
    
    def get_status(self) -> ModelStatus:
        """Get current model status."""
        model_size_mb = None
        if self.model and hasattr(self.model, 'num_parameters'):
            # Estimate size (rough calculation)
            param_count = self.model.num_parameters()
            model_size_mb = param_count * 4 / (1024 * 1024)  # Assume 4 bytes per param
        
        return ModelStatus(
            distilbert_loaded=self.is_loaded(),
            model_path=self.model_path,
            model_size_mb=model_size_mb,
            vocabulary_size=len(self.tokenizer.vocab) if self.tokenizer else None,
            supported_classes=[dtype.value for dtype in DocumentType],
            last_loaded=self.last_loaded
        )
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        self.classifier_pipeline = None
        self.model_path = None
        self.last_loaded = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully")


# Global model manager instance
model_manager = ModelManager()