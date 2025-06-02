"""LLM-enhanced boundary detection for construction litigation documents.

This module provides LLM-based validation and refinement of document boundaries
detected through traditional methods, dramatically improving accuracy on real-world
construction documents.
"""

import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from dataclasses import dataclass

from ..llm_client.router import LLMRouter, PrivacyMode
from ..llm_client.base_client import LLMError, LLMServiceUnavailable
from ..llm_client.prompt_templates import PromptTemplates


@dataclass
class BoundaryCandidate:
    """Represents a potential document boundary."""
    page_number: int
    position: int  # Character position in full text
    confidence: float
    method: str  # 'pattern', 'visual', 'llm'
    reasoning: Optional[str] = None


@dataclass
class BoundaryValidationResult:
    """Result of LLM boundary validation."""
    is_boundary: bool
    confidence: float
    reasoning: str
    processing_time: float


class LLMBoundaryDetector:
    """LLM-enhanced boundary detection for document processing."""
    
    def __init__(self,
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "llama3:8b-instruct-q5_K_M",
                 openai_model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None,
                 default_privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE):
        """Initialize LLM boundary detector.
        
        Args:
            ollama_host: Ollama service host
            ollama_model: Ollama model name
            openai_model: OpenAI model name
            openai_api_key: OpenAI API key
            default_privacy_mode: Default privacy mode
        """
        self.router = LLMRouter(
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            openai_model=openai_model,
            openai_api_key=openai_api_key
        )
        self.default_privacy_mode = default_privacy_mode
        
        # Text windowing parameters
        self.window_size = 300  # Words before/after boundary candidate
        self.overlap_size = 50   # Words of overlap between windows
        self.min_segment_length = 100  # Minimum words in a document segment
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.3
        
        # Simple in-memory cache for boundary validation results
        self._boundary_cache = {}
        self._cache_max_size = 1000  # Maximum number of cached results
    
    def validate_boundaries(self, 
                          full_text: str,
                          boundary_candidates: List[BoundaryCandidate],
                          privacy_mode: Optional[PrivacyMode] = None) -> List[BoundaryCandidate]:
        """Validate boundary candidates using LLM analysis.
        
        Args:
            full_text: Complete document text
            boundary_candidates: List of potential boundaries to validate
            privacy_mode: Privacy mode for LLM processing
            
        Returns:
            List of validated boundary candidates with updated confidence scores
        """
        if not boundary_candidates:
            return []
        
        privacy_mode = privacy_mode or self._determine_privacy_mode(full_text)
        validated_boundaries = []
        
        logger.info(f"Validating {len(boundary_candidates)} boundary candidates with LLM")
        
        for i, candidate in enumerate(boundary_candidates):
            try:
                logger.debug(f"Validating boundary {i+1}/{len(boundary_candidates)} at position {candidate.position}")
                
                # Extract text windows around the boundary
                current_window, next_window = self._extract_boundary_windows(
                    full_text, candidate.position
                )
                
                if not current_window or not next_window:
                    logger.warning(f"Could not extract windows for boundary at position {candidate.position}")
                    validated_boundaries.append(candidate)
                    continue
                
                # Get LLM validation
                validation_result = self._validate_single_boundary(
                    current_window, next_window, privacy_mode
                )
                
                # Update candidate with LLM results
                updated_candidate = self._merge_boundary_confidence(
                    candidate, validation_result
                )
                
                validated_boundaries.append(updated_candidate)
                
            except Exception as e:
                logger.warning(f"Failed to validate boundary {i+1}: {e}")
                # Keep original candidate if validation fails
                validated_boundaries.append(candidate)
        
        # Sort by confidence and filter low-confidence boundaries
        validated_boundaries = sorted(
            validated_boundaries, 
            key=lambda x: x.confidence, 
            reverse=True
        )
        
        # Filter out very low confidence boundaries
        filtered_boundaries = [
            b for b in validated_boundaries 
            if b.confidence >= self.low_confidence_threshold
        ]
        
        logger.info(f"LLM validation complete: {len(filtered_boundaries)}/{len(boundary_candidates)} boundaries retained")
        return filtered_boundaries
    
    def detect_boundaries_from_text(self,
                                   full_text: str,
                                   privacy_mode: Optional[PrivacyMode] = None) -> List[BoundaryCandidate]:
        """Detect boundaries directly from text using LLM analysis.
        
        Args:
            full_text: Complete document text
            privacy_mode: Privacy mode for processing
            
        Returns:
            List of detected boundaries
        """
        privacy_mode = privacy_mode or self._determine_privacy_mode(full_text)
        
        logger.info("Detecting boundaries using LLM analysis")
        
        words = full_text.split()
        total_words = len(words)
        
        # For short documents, use pattern-based candidate detection
        if total_words < self.window_size:
            logger.info(f"Short document ({total_words} words), using pattern-based approach")
            return self._detect_boundaries_short_document(full_text, privacy_mode)
        else:
            logger.info(f"Long document ({total_words} words), using sliding window approach")
            return self._detect_boundaries_sliding_window(full_text, privacy_mode)
    
    def _detect_boundaries_short_document(self,
                                        full_text: str,
                                        privacy_mode: PrivacyMode) -> List[BoundaryCandidate]:
        """Detect boundaries in short documents using pattern-based candidate detection."""
        # Look for clear document boundary markers
        boundary_patterns = [
            r'\n---+\n',  # Horizontal line separators
            r'\n={3,}\n',  # Equal sign separators
            r'(?i)\n\s*REQUEST FOR INFORMATION',  # RFI headers
            r'(?i)\n\s*INVOICE',  # Invoice headers  
            r'(?i)\n\s*CHANGE ORDER',  # Change order headers
            r'(?i)\n\s*FROM:\s*\S+@\S+',  # Email headers
            r'(?i)\n\s*DAILY REPORT',  # Daily report headers
            r'(?i)\n\s*MEETING MINUTES',  # Meeting minutes headers
        ]
        
        import re
        boundaries = []
        
        for pattern in boundary_patterns:
            matches = list(re.finditer(pattern, full_text))
            for match in matches:
                # Get context around the potential boundary
                match_pos = match.start()
                
                # Extract before and after segments
                before_start = max(0, match_pos - 300)  # 300 chars before
                after_end = min(len(full_text), match_pos + 300)  # 300 chars after
                
                before_text = full_text[before_start:match_pos]
                after_text = full_text[match_pos:after_end]
                
                # Validate with LLM
                try:
                    validation_result = self._validate_single_boundary(
                        before_text, after_text, privacy_mode
                    )
                    
                    if validation_result.is_boundary and validation_result.confidence > self.low_confidence_threshold:
                        boundary = BoundaryCandidate(
                            page_number=self._estimate_page_number(match_pos, full_text),
                            position=match_pos,
                            confidence=validation_result.confidence,
                            method='llm_pattern',
                            reasoning=f"Pattern: {pattern[:20]}... - {validation_result.reasoning}"
                        )
                        boundaries.append(boundary)
                        
                except Exception as e:
                    logger.warning(f"Failed to validate pattern boundary: {e}")
                    continue
        
        # Remove duplicate boundaries (too close together)
        boundaries = self._remove_duplicate_boundaries(boundaries)
        
        logger.info(f"Pattern-based detection found {len(boundaries)} boundaries")
        return boundaries
    
    def _detect_boundaries_sliding_window(self,
                                        full_text: str,
                                        privacy_mode: PrivacyMode) -> List[BoundaryCandidate]:
        """Detect boundaries in long documents using sliding window approach."""
        # Split text into overlapping windows
        windows = self._create_sliding_windows(full_text)
        boundaries = []
        
        for i, (window_start, window_text) in enumerate(windows[:-1]):
            try:
                # Get text segments for boundary detection
                current_end = self._get_window_end(window_text)
                next_start = self._get_window_start(windows[i+1][1])
                
                # Validate potential boundary
                validation_result = self._validate_single_boundary(
                    current_end, next_start, privacy_mode
                )
                
                if validation_result.is_boundary and validation_result.confidence > self.low_confidence_threshold:
                    boundary = BoundaryCandidate(
                        page_number=self._estimate_page_number(window_start, full_text),
                        position=window_start + len(window_text),
                        confidence=validation_result.confidence,
                        method='llm_window',
                        reasoning=validation_result.reasoning
                    )
                    boundaries.append(boundary)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze window {i+1}: {e}")
                continue
        
        logger.info(f"Sliding window detection found {len(boundaries)} boundaries")
        return boundaries
    
    def _remove_duplicate_boundaries(self, boundaries: List[BoundaryCandidate]) -> List[BoundaryCandidate]:
        """Remove boundaries that are too close together."""
        if not boundaries:
            return boundaries
        
        # Sort by position
        sorted_boundaries = sorted(boundaries, key=lambda b: b.position)
        
        # Remove duplicates within 100 characters of each other
        filtered = [sorted_boundaries[0]]
        for boundary in sorted_boundaries[1:]:
            if boundary.position - filtered[-1].position > 100:
                filtered.append(boundary)
            elif boundary.confidence > filtered[-1].confidence:
                # Replace with higher confidence boundary
                filtered[-1] = boundary
        
        return filtered
    
    def _validate_single_boundary(self,
                                 current_segment: str,
                                 next_segment: str,
                                 privacy_mode: PrivacyMode) -> BoundaryValidationResult:
        """Validate a single boundary using LLM with caching.
        
        Args:
            current_segment: Text ending current document
            next_segment: Text starting next potential document
            privacy_mode: Privacy mode for processing
            
        Returns:
            Boundary validation result
        """
        start_time = time.time()
        
        # Create cache key from text segments
        cache_key = self._create_cache_key(current_segment, next_segment, privacy_mode)
        
        # Check cache first
        if cache_key in self._boundary_cache:
            cached_result = self._boundary_cache[cache_key]
            logger.debug("Using cached boundary validation result")
            # Return cached result with updated processing time
            return BoundaryValidationResult(
                is_boundary=cached_result["is_boundary"],
                confidence=cached_result["confidence"],
                reasoning=f"[CACHED] {cached_result['reasoning']}",
                processing_time=time.time() - start_time  # Very fast for cached results
            )
        
        try:
            # Use router's boundary detection method
            response = self.router.detect_boundary(
                current_segment=current_segment,
                next_segment=next_segment,
                privacy_mode=privacy_mode
            )
            
            # Parse LLM response
            parsed = PromptTemplates.parse_boundary_response(response.content)
            
            if not parsed:
                raise LLMError("Failed to parse LLM boundary response")
            
            processing_time = time.time() - start_time
            
            result = BoundaryValidationResult(
                is_boundary=parsed["is_boundary"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                processing_time=processing_time
            )
            
            # Cache the result
            self._cache_boundary_result(cache_key, parsed)
            
            return result
            
        except (LLMError, LLMServiceUnavailable) as e:
            logger.error(f"LLM boundary validation failed: {e}")
            # Return conservative result
            return BoundaryValidationResult(
                is_boundary=False,
                confidence=0.0,
                reasoning=f"LLM validation failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _extract_boundary_windows(self, 
                                 full_text: str, 
                                 boundary_position: int) -> Tuple[str, str]:
        """Extract text windows around a boundary position.
        
        Args:
            full_text: Complete document text
            boundary_position: Character position of potential boundary
            
        Returns:
            Tuple of (current_window, next_window)
        """
        # Convert to word-based positions for more accurate windowing
        words = full_text.split()
        chars_so_far = 0
        boundary_word_index = 0
        
        # Find word index corresponding to character position
        for i, word in enumerate(words):
            if chars_so_far >= boundary_position:
                boundary_word_index = i
                break
            chars_so_far += len(word) + 1  # +1 for space
        
        # Extract windows around boundary
        start_index = max(0, boundary_word_index - self.window_size)
        end_index = min(len(words), boundary_word_index + self.window_size)
        
        # Current segment: words before boundary
        current_start = max(0, boundary_word_index - self.window_size // 2)
        current_words = words[current_start:boundary_word_index]
        
        # Next segment: words after boundary
        next_end = min(len(words), boundary_word_index + self.window_size // 2)
        next_words = words[boundary_word_index:next_end]
        
        # Handle edge cases where we don't have enough context
        if not current_words and boundary_word_index == 0:
            # At start of document - use beginning as "current" for context
            current_words = words[:min(50, len(words))]  # First 50 words as context
        
        if not next_words:
            # At end of document - use ending as "next" for context  
            next_words = words[max(0, len(words) - 50):]  # Last 50 words as context
        
        current_window = " ".join(current_words)
        next_window = " ".join(next_words)
        
        return current_window, next_window
    
    def _create_sliding_windows(self, text: str) -> List[Tuple[int, str]]:
        """Create overlapping text windows for boundary detection.
        
        Args:
            text: Full document text
            
        Returns:
            List of (start_position, window_text) tuples
        """
        words = text.split()
        windows = []
        
        window_step = self.window_size - self.overlap_size
        
        for i in range(0, len(words), window_step):
            end_index = min(i + self.window_size, len(words))
            window_words = words[i:end_index]
            
            if len(window_words) < self.min_segment_length:
                break
                
            window_text = " ".join(window_words)
            
            # Calculate character position
            char_position = len(" ".join(words[:i]))
            if i > 0:
                char_position += 1  # Add space
                
            windows.append((char_position, window_text))
        
        return windows
    
    def _get_window_end(self, window_text: str) -> str:
        """Get the ending portion of a text window.
        
        Args:
            window_text: Window text
            
        Returns:
            Last ~150 words of the window
        """
        words = window_text.split()
        if len(words) > 150:
            return " ".join(words[-150:])
        return window_text
    
    def _get_window_start(self, window_text: str) -> str:
        """Get the starting portion of a text window.
        
        Args:
            window_text: Window text
            
        Returns:
            First ~150 words of the window
        """
        words = window_text.split()
        if len(words) > 150:
            return " ".join(words[:150])
        return window_text
    
    def _merge_boundary_confidence(self,
                                  original_candidate: BoundaryCandidate,
                                  llm_result: BoundaryValidationResult) -> BoundaryCandidate:
        """Merge original boundary detection with LLM validation.
        
        Args:
            original_candidate: Original boundary candidate
            llm_result: LLM validation result
            
        Returns:
            Updated boundary candidate
        """
        # Weighted average of original and LLM confidence
        if llm_result.is_boundary:
            # If LLM confirms boundary, weight LLM higher
            combined_confidence = (0.3 * original_candidate.confidence + 
                                 0.7 * llm_result.confidence)
        else:
            # If LLM rejects boundary, weight LLM much higher
            combined_confidence = (0.1 * original_candidate.confidence + 
                                 0.9 * (1.0 - llm_result.confidence))
        
        # Update reasoning
        updated_reasoning = f"{original_candidate.method}: {original_candidate.confidence:.2f}, LLM: {llm_result.reasoning}"
        
        return BoundaryCandidate(
            page_number=original_candidate.page_number,
            position=original_candidate.position,
            confidence=combined_confidence,
            method=f"{original_candidate.method}+llm",
            reasoning=updated_reasoning
        )
    
    def _estimate_page_number(self, char_position: int, full_text: str) -> int:
        """Estimate page number from character position.
        
        Args:
            char_position: Character position in text
            full_text: Full document text
            
        Returns:
            Estimated page number (1-based)
        """
        # Rough estimation: 2000 characters per page
        chars_per_page = 2000
        return max(1, int(char_position / chars_per_page) + 1)
    
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status information.
        
        Returns:
            Status dictionary
        """
        return {
            "llm_available": self.router.get_status(),
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
            "confidence_thresholds": {
                "high": self.high_confidence_threshold,
                "low": self.low_confidence_threshold
            },
            "cache_stats": {
                "cache_size": len(self._boundary_cache),
                "cache_max_size": self._cache_max_size
            }
        }
    
    def _create_cache_key(self, current_segment: str, next_segment: str, privacy_mode: PrivacyMode) -> str:
        """Create a cache key for boundary validation."""
        # Normalize segments to reduce cache misses from minor variations
        current_normalized = " ".join(current_segment.split())[:200]  # First 200 chars
        next_normalized = " ".join(next_segment.split())[:200]  # First 200 chars
        
        # Create hash of the key components
        key_data = f"{current_normalized}|{next_normalized}|{privacy_mode}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_boundary_result(self, cache_key: str, parsed_result: Dict[str, Any]) -> None:
        """Cache a boundary validation result."""
        # Implement simple LRU eviction if cache is full
        if len(self._boundary_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._boundary_cache))
            del self._boundary_cache[oldest_key]
            logger.debug("Cache full, evicted oldest entry")
        
        # Store the result
        self._boundary_cache[cache_key] = {
            "is_boundary": parsed_result["is_boundary"],
            "confidence": parsed_result["confidence"],
            "reasoning": parsed_result["reasoning"]
        }
        
        logger.debug(f"Cached boundary result (cache size: {len(self._boundary_cache)})")
    
    def clear_cache(self) -> None:
        """Clear the boundary validation cache."""
        self._boundary_cache.clear()
        logger.info("Boundary validation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._boundary_cache),
            "cache_max_size": self._cache_max_size,
            "cache_hit_potential": len(self._boundary_cache) / max(1, self._cache_max_size)
        }


# Global instance for use by other modules
llm_boundary_detector = LLMBoundaryDetector()