"""Text windowing and boundary validation utilities for LLM-enhanced boundary detection.

This module provides sophisticated text windowing and context preservation
strategies for optimal LLM boundary detection performance.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TextWindow:
    """Represents a text window with context information."""
    start_position: int
    end_position: int
    text: str
    word_count: int
    has_document_markers: bool
    quality_score: float


@dataclass
class BoundaryContext:
    """Context information around a potential boundary."""
    before_text: str
    after_text: str
    boundary_position: int
    confidence_indicators: List[str]
    quality_metrics: Dict[str, float]


class TextWindowManager:
    """Manages text windowing for boundary detection."""
    
    def __init__(self,
                 base_window_size: int = 300,
                 min_window_size: int = 100,
                 max_window_size: int = 500,
                 overlap_ratio: float = 0.2):
        """Initialize text window manager.
        
        Args:
            base_window_size: Base window size in words
            min_window_size: Minimum window size
            max_window_size: Maximum window size
            overlap_ratio: Overlap ratio between windows
        """
        self.base_window_size = base_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.overlap_ratio = overlap_ratio
        
        # Document boundary indicators
        self.strong_boundary_patterns = [
            r'(?i)^from:\s*\S+@\S+',  # Email headers
            r'(?i)^to:\s*\S+@\S+',
            r'(?i)^subject:\s*',
            r'(?i)^date:\s*\d',
            r'(?i)^request for information',
            r'(?i)^rfi\s*#?\d+',
            r'(?i)^change order\s*#?\d+',
            r'(?i)^invoice\s*#?\w+',
            r'(?i)^bill\s+to:',
            r'(?i)^contract\s+for\s+',
            r'(?i)^meeting minutes',
            r'(?i)^daily report',
            r'(?i)^submittal\s*#?\d+'
        ]
        
        self.weak_boundary_patterns = [
            r'(?i)sincerely,?\s*$',
            r'(?i)best regards,?\s*$',
            r'(?i)thank you,?\s*$',
            r'(?i)yours truly,?\s*$',
            r'(?i)\[end of \w+\]',
            r'(?i)page \d+ of \d+',
            r'(?i)continued on next page',
            r'\f',  # Form feed
            r'={3,}',  # Multiple equals signs
            r'-{3,}',  # Multiple dashes
        ]
        
        # Content quality indicators
        self.quality_indicators = {
            'has_sentences': r'[.!?]+\s+[A-Z]',
            'has_numbers': r'\d+',
            'has_dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'has_amounts': r'\$[\d,]+\.?\d*',
            'has_emails': r'\S+@\S+\.\S+',
            'has_proper_nouns': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'coherent_structure': r'^\s*[A-Z].*[.!?]\s*$'
        }
    
    def create_adaptive_windows(self, 
                               full_text: str,
                               boundary_candidates: List[int]) -> List[BoundaryContext]:
        """Create adaptive text windows around boundary candidates.
        
        Args:
            full_text: Complete document text
            boundary_candidates: List of character positions for potential boundaries
            
        Returns:
            List of boundary contexts with optimal windowing
        """
        contexts = []
        
        for position in boundary_candidates:
            try:
                context = self._create_boundary_context(full_text, position)
                if context:
                    contexts.append(context)
            except Exception as e:
                logger.warning(f"Failed to create context for boundary at {position}: {e}")
                continue
        
        return contexts
    
    def _create_boundary_context(self, 
                                full_text: str, 
                                boundary_position: int) -> Optional[BoundaryContext]:
        """Create optimized context around a boundary position.
        
        Args:
            full_text: Complete document text
            boundary_position: Character position of potential boundary
            
        Returns:
            Boundary context or None if creation fails
        """
        # Convert to word-based processing
        words = full_text.split()
        word_positions = self._calculate_word_positions(full_text)
        
        # Find word index for boundary position
        boundary_word_index = self._find_word_index(boundary_position, word_positions)
        
        if boundary_word_index is None:
            return None
        
        # Determine optimal window size based on content quality
        window_size = self._determine_window_size(words, boundary_word_index)
        
        # Extract before and after windows
        before_start = max(0, boundary_word_index - window_size)
        before_words = words[before_start:boundary_word_index]
        
        after_end = min(len(words), boundary_word_index + window_size)
        after_words = words[boundary_word_index:after_end]
        
        before_text = " ".join(before_words)
        after_text = " ".join(after_words)
        
        # Analyze context quality and indicators
        confidence_indicators = self._analyze_boundary_indicators(before_text, after_text)
        quality_metrics = self._calculate_quality_metrics(before_text, after_text)
        
        return BoundaryContext(
            before_text=before_text,
            after_text=after_text,
            boundary_position=boundary_position,
            confidence_indicators=confidence_indicators,
            quality_metrics=quality_metrics
        )
    
    def _calculate_word_positions(self, text: str) -> List[int]:
        """Calculate character positions for each word.
        
        Args:
            text: Input text
            
        Returns:
            List of character positions for each word start
        """
        positions = []
        current_pos = 0
        
        for word in text.split():
            # Find word in text starting from current position
            word_start = text.find(word, current_pos)
            if word_start != -1:
                positions.append(word_start)
                current_pos = word_start + len(word)
            else:
                # Fallback: estimate position
                positions.append(current_pos)
                current_pos += len(word) + 1
        
        return positions
    
    def _find_word_index(self, char_position: int, word_positions: List[int]) -> Optional[int]:
        """Find word index corresponding to character position.
        
        Args:
            char_position: Character position to find
            word_positions: List of word start positions
            
        Returns:
            Word index or None if not found
        """
        for i, pos in enumerate(word_positions):
            if pos >= char_position:
                return i
        
        # Return last word index if position is beyond text
        return len(word_positions) - 1 if word_positions else None
    
    def _determine_window_size(self, words: List[str], boundary_index: int) -> int:
        """Determine optimal window size based on content analysis.
        
        Args:
            words: List of all words
            boundary_index: Index of boundary word
            
        Returns:
            Optimal window size in words
        """
        # Analyze content density around boundary
        start_analysis = max(0, boundary_index - 50)
        end_analysis = min(len(words), boundary_index + 50)
        
        analysis_text = " ".join(words[start_analysis:end_analysis])
        
        # Base window size
        window_size = self.base_window_size
        
        # Adjust based on content characteristics
        if self._has_dense_content(analysis_text):
            window_size = min(self.max_window_size, int(window_size * 1.2))
        elif self._has_sparse_content(analysis_text):
            window_size = max(self.min_window_size, int(window_size * 0.8))
        
        # Ensure we don't exceed text boundaries
        available_before = boundary_index
        available_after = len(words) - boundary_index
        
        max_window = min(available_before, available_after)
        return min(window_size, max_window)
    
    def _has_dense_content(self, text: str) -> bool:
        """Check if text has dense, information-rich content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if content is dense
        """
        # Look for indicators of dense content
        indicators = [
            len(re.findall(r'\d+', text)) > 10,  # Many numbers
            len(re.findall(r'\$[\d,]+', text)) > 5,  # Multiple amounts
            len(re.findall(r'\b[A-Z]{2,}\b', text)) > 10,  # Many acronyms
            len(re.findall(r'\w+@\w+', text)) > 3,  # Multiple emails
            text.count('.') / len(text.split()) > 0.15  # High sentence density
        ]
        
        return sum(indicators) >= 2
    
    def _has_sparse_content(self, text: str) -> bool:
        """Check if text has sparse, low-information content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if content is sparse
        """
        if not text.strip():
            return True
        
        # Look for indicators of sparse content
        indicators = [
            len(text.split()) < 20,  # Very short
            text.count('\n') / len(text.split()) > 0.5,  # Many line breaks
            len(set(text.lower().split())) / len(text.split()) < 0.5,  # Low vocabulary diversity
            re.search(r'^[\s\n]*$', text) is not None,  # Mostly whitespace
            len(re.findall(r'\w+', text)) < 10  # Very few words
        ]
        
        return sum(indicators) >= 2
    
    def _analyze_boundary_indicators(self, before_text: str, after_text: str) -> List[str]:
        """Analyze text for boundary indicators.
        
        Args:
            before_text: Text before potential boundary
            after_text: Text after potential boundary
            
        Returns:
            List of detected boundary indicators
        """
        indicators = []
        
        # Check for strong boundary patterns in after_text
        for pattern in self.strong_boundary_patterns:
            if re.search(pattern, after_text[:200]):  # Check first 200 chars
                indicators.append(f"strong_start: {pattern}")
        
        # Check for weak boundary patterns in before_text
        for pattern in self.weak_boundary_patterns:
            if re.search(pattern, before_text[-200:]):  # Check last 200 chars
                indicators.append(f"weak_end: {pattern}")
        
        # Topic change detection
        if self._detect_topic_change(before_text, after_text):
            indicators.append("topic_change")
        
        # Format change detection
        if self._detect_format_change(before_text, after_text):
            indicators.append("format_change")
        
        return indicators
    
    def _detect_topic_change(self, before_text: str, after_text: str) -> bool:
        """Detect if there's a significant topic change.
        
        Args:
            before_text: Text before boundary
            after_text: Text after boundary
            
        Returns:
            True if topic change is detected
        """
        # Simple keyword-based topic change detection
        before_keywords = self._extract_keywords(before_text)
        after_keywords = self._extract_keywords(after_text)
        
        if not before_keywords or not after_keywords:
            return False
        
        # Calculate keyword overlap
        overlap = len(before_keywords & after_keywords)
        total_unique = len(before_keywords | after_keywords)
        
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        
        # Topic change if low overlap
        return overlap_ratio < 0.3
    
    def _detect_format_change(self, before_text: str, after_text: str) -> bool:
        """Detect if there's a significant format change.
        
        Args:
            before_text: Text before boundary
            after_text: Text after boundary
            
        Returns:
            True if format change is detected
        """
        # Analyze formatting characteristics
        before_metrics = {
            'avg_line_length': self._avg_line_length(before_text),
            'capitalization_ratio': self._capitalization_ratio(before_text),
            'punctuation_density': self._punctuation_density(before_text),
            'number_density': self._number_density(before_text)
        }
        
        after_metrics = {
            'avg_line_length': self._avg_line_length(after_text),
            'capitalization_ratio': self._capitalization_ratio(after_text),
            'punctuation_density': self._punctuation_density(after_text),
            'number_density': self._number_density(after_text)
        }
        
        # Check for significant differences
        differences = []
        for key in before_metrics:
            before_val = before_metrics[key]
            after_val = after_metrics[key]
            
            if before_val > 0 and after_val > 0:
                ratio = abs(before_val - after_val) / max(before_val, after_val)
                differences.append(ratio)
        
        # Format change if multiple significant differences
        significant_differences = sum(1 for diff in differences if diff > 0.5)
        return significant_differences >= 2
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'has', 'have', 'been', 'will', 'this', 'that', 'with', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        
        keywords = {word for word in words if word not in stopwords and len(word) > 3}
        return keywords
    
    def _calculate_quality_metrics(self, before_text: str, after_text: str) -> Dict[str, float]:
        """Calculate quality metrics for boundary context.
        
        Args:
            before_text: Text before boundary
            after_text: Text after boundary
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Length metrics
        metrics['before_length'] = len(before_text.split())
        metrics['after_length'] = len(after_text.split())
        metrics['length_balance'] = min(metrics['before_length'], metrics['after_length']) / max(metrics['before_length'], metrics['after_length']) if max(metrics['before_length'], metrics['after_length']) > 0 else 0
        
        # Content quality metrics
        for name, pattern in self.quality_indicators.items():
            before_count = len(re.findall(pattern, before_text))
            after_count = len(re.findall(pattern, after_text))
            metrics[f'before_{name}'] = before_count
            metrics[f'after_{name}'] = after_count
        
        # Overall quality score
        quality_factors = [
            metrics['length_balance'],
            min(1.0, metrics['before_has_sentences'] / 5.0),
            min(1.0, metrics['after_has_sentences'] / 5.0),
            min(1.0, (metrics['before_has_proper_nouns'] + metrics['after_has_proper_nouns']) / 10.0)
        ]
        
        metrics['overall_quality'] = sum(quality_factors) / len(quality_factors)
        
        return metrics
    
    def _avg_line_length(self, text: str) -> float:
        """Calculate average line length."""
        lines = text.split('\n')
        if not lines:
            return 0.0
        return sum(len(line) for line in lines) / len(lines)
    
    def _capitalization_ratio(self, text: str) -> float:
        """Calculate ratio of capitalized words."""
        words = text.split()
        if not words:
            return 0.0
        capitalized = sum(1 for word in words if word and word[0].isupper())
        return capitalized / len(words)
    
    def _punctuation_density(self, text: str) -> float:
        """Calculate punctuation density."""
        if not text:
            return 0.0
        punctuation = len(re.findall(r'[.!?,:;]', text))
        return punctuation / len(text)
    
    def _number_density(self, text: str) -> float:
        """Calculate number density."""
        if not text:
            return 0.0
        numbers = len(re.findall(r'\d+', text))
        words = len(text.split())
        return numbers / words if words > 0 else 0.0


# Global instance
text_window_manager = TextWindowManager()