"""Hybrid boundary detection combining pattern matching and visual analysis.

This module orchestrates multiple boundary detection methods for maximum accuracy,
using a progressive approach from fast heuristics to deep visual analysis.
"""

import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF
from loguru import logger

from .boundary_detector import BoundaryDetector
from .improved_boundary_detector import ImprovedBoundaryDetector
from .visual_boundary_detector import VisualBoundaryDetector, BoundaryScore, PageFeatures
from .ocr_handler import OCRHandler
from .layoutlm_boundary_detector import LayoutLMBoundaryDetector
from .llm_boundary_detector import LLMBoundaryDetector
from ..llm_client.router import PrivacyMode


class DetectionLevel(Enum):
    """Detection levels for progressive analysis."""
    HEURISTIC = 1  # Fast pattern matching
    VISUAL = 2     # Visual similarity analysis
    LLM = 3        # LLM-enhanced validation
    DEEP = 4       # LayoutLM integration


@dataclass
class HybridBoundaryResult:
    """Result from hybrid boundary detection."""
    boundaries: List[Tuple[int, int]]
    detection_level: DetectionLevel
    confidence_scores: Dict[int, float]  # Page number -> confidence
    processing_time: float
    visual_scores: Optional[List[BoundaryScore]] = None


class HybridBoundaryDetector:
    """Progressive boundary detection using multiple methods."""
    
    def __init__(self,
                 ocr_handler: Optional[OCRHandler] = None,
                 visual_model: str = "clip-ViT-B-32",
                 cache_dir: str = ".boundary_cache",
                 device: str = None,
                 privacy_mode: PrivacyMode = PrivacyMode.HYBRID_SAFE):
        """Initialize hybrid detector.
        
        Args:
            ocr_handler: OCR handler for scanned pages
            visual_model: Visual embedding model name
            cache_dir: Directory for caching embeddings
            device: Device for model inference
            privacy_mode: Privacy mode for LLM operations
        """
        self.ocr_handler = ocr_handler
        self.privacy_mode = privacy_mode
        
        # Initialize detectors with improved boundary detector
        self.pattern_detector = ImprovedBoundaryDetector(ocr_handler)
        self.visual_detector = None  # Lazy load
        self.llm_detector = None  # Lazy load for LLM analysis
        self.layoutlm_detector = None  # Lazy load for deep analysis
        
        # Configuration
        self.visual_model = visual_model
        self.cache_dir = cache_dir
        self.device = device
        
        # Thresholds
        self.pattern_confidence_threshold = 0.7
        self.visual_confidence_threshold = 0.6
        self.llm_confidence_threshold = 0.7
        self.min_visual_pages = 10  # Don't use visual for very small PDFs
        self.min_llm_pages = 5  # Use LLM for smaller documents too
    
    def detect_boundaries(self,
                         pdf_doc: fitz.Document,
                         max_level: DetectionLevel = DetectionLevel.LLM,
                         force_visual: bool = False,
                         force_llm: bool = False,
                         progress_callback=None) -> HybridBoundaryResult:
        """Detect document boundaries using progressive approach.
        
        Args:
            pdf_doc: PyMuPDF document
            max_level: Maximum detection level to use
            force_visual: Force visual detection regardless of pattern results
            force_llm: Force LLM validation regardless of other results
            
        Returns:
            Hybrid boundary detection result
        """
        start_time = time.time()
        logger.info(f"Starting hybrid boundary detection for {pdf_doc.page_count} pages")
        
        # Level 1: Pattern-based detection
        pattern_result = self._run_pattern_detection(pdf_doc)
        
        # Check if we need higher level detection
        needs_visual = (
            force_visual or
            pdf_doc.page_count >= self.min_visual_pages and (
                self._has_low_confidence_boundaries(pattern_result) or
                self._has_suspiciously_many_documents(pattern_result, pdf_doc.page_count) or
                self._is_likely_scanned(pdf_doc)
            )
        )
        
        current_result = pattern_result
        current_level = DetectionLevel.HEURISTIC
        
        if needs_visual and max_level.value >= DetectionLevel.VISUAL.value:
            logger.info("Running visual boundary detection for improved accuracy")
            visual_result = self._run_visual_detection(pdf_doc)
            current_result = self._merge_detection_results(pattern_result, visual_result, pdf_doc)
            current_level = DetectionLevel.VISUAL
        
        # Check if we need LLM validation
        needs_llm = (
            force_llm or
            max_level.value >= DetectionLevel.LLM.value and (
                pdf_doc.page_count >= self.min_llm_pages and (
                    self._has_low_confidence_boundaries(current_result) or
                    self._has_suspiciously_many_documents(current_result, pdf_doc.page_count) or
                    current_level == DetectionLevel.HEURISTIC  # Always validate pattern-only results
                )
            )
        )
        
        if needs_llm:
            logger.info("Running LLM boundary validation for enhanced accuracy")
            llm_result = self._run_llm_validation(pdf_doc, current_result)
            current_result = llm_result
            current_level = DetectionLevel.LLM
        
        # Check if we need deep LayoutLM analysis
        needs_deep = (
            max_level.value >= DetectionLevel.DEEP.value and (
                self._has_very_low_confidence(current_result) or
                self._has_ambiguous_boundaries(current_result)
            )
        )
        
        if needs_deep:
            logger.info("Running deep LayoutLM analysis for complex document")
            deep_result = self._run_deep_detection(pdf_doc, current_result)
            current_result = deep_result
            current_level = DetectionLevel.DEEP
        
        # Set final detection level
        current_result.detection_level = current_level
        
        # Record processing time
        current_result.processing_time = time.time() - start_time
        
        logger.info(
            f"Hybrid detection completed in {current_result.processing_time:.2f}s "
            f"using {current_result.detection_level.name.lower()} level, "
            f"found {len(current_result.boundaries)} documents"
        )
        
        return current_result
    
    def _run_pattern_detection(self, pdf_doc: fitz.Document) -> HybridBoundaryResult:
        """Run pattern-based boundary detection."""
        boundaries = self.pattern_detector.detect_boundaries(pdf_doc)
        
        # Estimate confidence based on detection patterns
        confidence_scores = {}
        for start, end in boundaries:
            # Higher confidence for multi-page documents
            base_confidence = 0.8 if (end - start) > 0 else 0.6
            for page_num in range(start, end + 1):
                confidence_scores[page_num] = base_confidence
        
        return HybridBoundaryResult(
            boundaries=boundaries,
            detection_level=DetectionLevel.HEURISTIC,
            confidence_scores=confidence_scores,
            processing_time=0.0
        )
    
    def _run_visual_detection(self, pdf_doc: fitz.Document) -> HybridBoundaryResult:
        """Run visual boundary detection."""
        # Lazy load visual detector
        if self.visual_detector is None:
            self.visual_detector = VisualBoundaryDetector(
                cache_dir=self.cache_dir,
                model_name=self.visual_model,
                device=self.device
            )
        
        # Get visual boundary scores
        visual_scores = self.visual_detector.detect_boundaries(
            pdf_doc, self.ocr_handler
        )
        
        # Convert to boundaries
        boundaries = self.visual_detector.convert_to_boundaries(
            visual_scores, min_confidence=self.visual_confidence_threshold
        )
        
        # Extract confidence scores
        confidence_scores = {
            score.page_num: score.confidence 
            for score in visual_scores
        }
        
        return HybridBoundaryResult(
            boundaries=boundaries,
            detection_level=DetectionLevel.VISUAL,
            confidence_scores=confidence_scores,
            processing_time=0.0,
            visual_scores=visual_scores
        )
    
    def _run_llm_validation(self, 
                           pdf_doc: fitz.Document,
                           current_result: HybridBoundaryResult) -> HybridBoundaryResult:
        """Run LLM validation of boundary candidates.
        
        Args:
            pdf_doc: PyMuPDF document
            current_result: Current boundary detection result to validate
            
        Returns:
            LLM-validated boundary result
        """
        # Lazy load LLM detector
        if self.llm_detector is None:
            self.llm_detector = LLMBoundaryDetector(
                default_privacy_mode=self.privacy_mode
            )
        
        # Extract full text from PDF and track page positions
        full_text = ""
        page_positions = {}  # page_num -> char_position
        
        for page_num in range(pdf_doc.page_count):
            page_positions[page_num + 1] = len(full_text)  # Store position of page start
            page_text = pdf_doc[page_num].get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
        
        # Convert boundaries to boundary candidates
        boundary_candidates = []
        for start_page, end_page in current_result.boundaries:
            # Get actual character position for the page boundary
            char_position = page_positions.get(start_page, 0)
            
            confidence = current_result.confidence_scores.get(start_page, 0.5)
            method = current_result.detection_level.name.lower()
            
            candidate = BoundaryCandidate(
                page_number=start_page,
                position=char_position,
                confidence=confidence,
                method=method
            )
            boundary_candidates.append(candidate)
        
        # Run LLM validation
        validated_candidates = self.llm_detector.validate_boundaries(
            full_text=full_text,
            boundary_candidates=boundary_candidates,
            privacy_mode=self.privacy_mode
        )
        
        # Convert validated candidates back to boundaries
        validated_boundaries = []
        updated_confidence_scores = {}
        
        for candidate in validated_candidates:
            # Convert back to page-based boundaries
            start_page = candidate.page_number
            
            # Find end page (next boundary or end of document)
            end_page = pdf_doc.page_count - 1
            for other_candidate in validated_candidates:
                if other_candidate.page_number > start_page:
                    end_page = other_candidate.page_number - 1
                    break
            
            if start_page <= end_page:
                validated_boundaries.append((start_page, end_page))
                
                # Update confidence scores for all pages in this document
                for page_num in range(start_page, end_page + 1):
                    updated_confidence_scores[page_num] = candidate.confidence
        
        # Sort boundaries
        validated_boundaries = sorted(validated_boundaries)
        
        logger.info(
            f"LLM validation: {len(current_result.boundaries)} → {len(validated_boundaries)} boundaries"
        )
        
        return HybridBoundaryResult(
            boundaries=validated_boundaries,
            detection_level=DetectionLevel.LLM,
            confidence_scores=updated_confidence_scores,
            processing_time=0.0,
            visual_scores=current_result.visual_scores
        )
    
    def _merge_detection_results(self,
                                pattern_result: HybridBoundaryResult,
                                visual_result: HybridBoundaryResult,
                                pdf_doc: fitz.Document) -> HybridBoundaryResult:
        """Merge pattern and visual detection results.
        
        Strategy:
        1. Check average confidence of pattern detection
        2. If pattern detection has high confidence, prefer it
        3. If pattern detection is poor, trust visual more
        4. Merge high-confidence boundaries from both methods
        """
        pattern_boundaries = set(pattern_result.boundaries)
        visual_boundaries = set(visual_result.boundaries)
        
        # Calculate overlap
        overlap = pattern_boundaries & visual_boundaries
        only_pattern = pattern_boundaries - visual_boundaries
        only_visual = visual_boundaries - pattern_boundaries
        
        logger.info(
            f"Boundary comparison: {len(overlap)} common, "
            f"{len(only_pattern)} pattern-only, {len(only_visual)} visual-only"
        )
        
        # Calculate pattern detection quality
        pattern_avg_conf = 0.0
        if pattern_result.confidence_scores:
            pattern_avg_conf = sum(pattern_result.confidence_scores.values()) / len(pattern_result.confidence_scores)
        
        # If pattern detection has high confidence and reasonable number of boundaries, prefer it
        # For CPRA/construction documents, many small documents are normal, so be more lenient
        if (pattern_avg_conf > 0.7 and 
            len(pattern_boundaries) > 0 and
            len(pattern_boundaries) <= pdf_doc.page_count):  # Allow up to 1 document per page
            logger.info(f"High-confidence pattern detection (avg: {pattern_avg_conf:.3f}), preferring pattern results")
            return pattern_result
        
        # If high agreement, trust visual
        if len(overlap) / max(len(pattern_boundaries), 1) > 0.7:
            logger.info("High agreement between methods, using visual results")
            return visual_result
        
        # Otherwise, merge intelligently
        merged_boundaries = list(overlap)
        
        # Add high-confidence pattern boundaries (lowered threshold since pattern detection is working well)
        for boundary in only_pattern:
            start, end = boundary
            avg_confidence = sum(
                pattern_result.confidence_scores.get(p, 0) 
                for p in range(start, end + 1)
            ) / max(1, end - start + 1)
            
            # Lower threshold for pattern boundaries when pattern detection is generally good
            threshold = 0.6 if pattern_avg_conf > 0.7 else 0.8
            if avg_confidence > threshold:
                merged_boundaries.append(boundary)
                logger.debug(f"Keeping pattern boundary {boundary} with confidence {avg_confidence:.2f}")
        
        # Add visual boundaries with good scores (stricter threshold since visual disagreed with patterns)
        if visual_result.visual_scores:
            for boundary in only_visual:
                start, _ = boundary
                if start < len(visual_result.visual_scores):
                    score = visual_result.visual_scores[start]
                    if score.total_score > 0.8 and score.confidence > 0.7:  # Higher threshold
                        merged_boundaries.append(boundary)
                        logger.debug(
                            f"Adding visual boundary {boundary} with score "
                            f"{score.total_score:.2f}, confidence {score.confidence:.2f}"
                        )
        
        # Sort and clean up boundaries
        merged_boundaries = sorted(merged_boundaries)
        merged_boundaries = self._cleanup_overlapping_boundaries(merged_boundaries)
        
        # Combine confidence scores - weight pattern more heavily if it's performing well
        pattern_weight = 0.7 if pattern_avg_conf > 0.7 else 0.3
        visual_weight = 1.0 - pattern_weight
        
        merged_confidence = {}
        for page_num in range(pdf_doc.page_count):
            pattern_conf = pattern_result.confidence_scores.get(page_num, 0)
            visual_conf = visual_result.confidence_scores.get(page_num, 0)
            merged_confidence[page_num] = pattern_weight * pattern_conf + visual_weight * visual_conf
        
        return HybridBoundaryResult(
            boundaries=merged_boundaries,
            detection_level=DetectionLevel.VISUAL,
            confidence_scores=merged_confidence,
            processing_time=0.0,
            visual_scores=visual_result.visual_scores
        )
    
    def _has_low_confidence_boundaries(self, result: HybridBoundaryResult) -> bool:
        """Check if result has low confidence boundaries."""
        if not result.confidence_scores:
            return True
        
        avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores)
        return avg_confidence < self.pattern_confidence_threshold
    
    def _has_suspiciously_many_documents(self, 
                                        result: HybridBoundaryResult,
                                        page_count: int) -> bool:
        """Check if we detected too many documents (possible over-splitting)."""
        if page_count < 10:
            return False
        
        # For construction litigation documents, many boundaries are normal
        # Only consider suspicious if more than 1 document per 1.5 pages (very aggressive splitting)
        return len(result.boundaries) > page_count / 1.5
    
    def _is_likely_scanned(self, pdf_doc: fitz.Document) -> bool:
        """Quick check if PDF is likely scanned."""
        # Sample first few pages
        sample_size = min(5, pdf_doc.page_count)
        low_text_pages = 0
        
        for i in range(sample_size):
            text = pdf_doc[i].get_text().strip()
            if len(text) < 50:
                low_text_pages += 1
        
        return low_text_pages >= sample_size * 0.6
    
    def _cleanup_overlapping_boundaries(self, 
                                       boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Clean up overlapping or adjacent boundaries."""
        if not boundaries:
            return boundaries
        
        cleaned = []
        current_start, current_end = boundaries[0]
        
        for start, end in boundaries[1:]:
            if start <= current_end + 1:
                # Overlapping or adjacent, merge
                current_end = max(current_end, end)
            else:
                # Gap found, save current and start new
                cleaned.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Don't forget the last one
        cleaned.append((current_start, current_end))
        
        return cleaned
    
    def _run_deep_detection(self,
                           pdf_doc: fitz.Document,
                           visual_result: HybridBoundaryResult) -> HybridBoundaryResult:
        """Run deep LayoutLM-based detection."""
        # Lazy load LayoutLM detector
        if self.layoutlm_detector is None:
            self.layoutlm_detector = LayoutLMBoundaryDetector(
                device=self.device,
                cache_embeddings=True
            )
        
        # Extract page features from visual result if available
        page_features = []
        if visual_result.visual_scores:
            # Convert visual scores to page features (simplified)
            for score in visual_result.visual_scores:
                features = PageFeatures(page_num=score.page_num)
                page_features.append(features)
        
        # Run LayoutLM detection
        layoutlm_scores = self.layoutlm_detector.detect_boundaries(
            pdf_doc, page_features, self.ocr_handler
        )
        
        # Convert to boundaries
        boundaries = []
        current_start = 0
        
        for score in layoutlm_scores[1:]:  # Skip first page
            if score.total_score > 0.7 and score.confidence > 0.6:
                boundaries.append((current_start, score.page_num - 1))
                current_start = score.page_num
        
        # Add last document
        if current_start < len(layoutlm_scores):
            boundaries.append((current_start, len(layoutlm_scores) - 1))
        
        # Build confidence scores
        confidence_scores = {
            score.page_num: score.confidence
            for score in layoutlm_scores
        }
        
        return HybridBoundaryResult(
            boundaries=boundaries,
            detection_level=DetectionLevel.DEEP,
            confidence_scores=confidence_scores,
            processing_time=0.0,
            visual_scores=layoutlm_scores
        )
    
    def _has_very_low_confidence(self, result: HybridBoundaryResult) -> bool:
        """Check if result has very low confidence requiring deep analysis."""
        if not result.confidence_scores:
            return True
        
        avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores)
        return avg_confidence < 0.5
    
    def _has_ambiguous_boundaries(self, result: HybridBoundaryResult) -> bool:
        """Check if boundaries are ambiguous (many uncertain scores)."""
        if not result.visual_scores:
            return False
        
        # Count ambiguous scores (between 0.4 and 0.6)
        ambiguous_count = sum(
            1 for score in result.visual_scores
            if 0.4 <= score.total_score <= 0.6
        )
        
        # If more than 30% are ambiguous, needs deep analysis
        return ambiguous_count > len(result.visual_scores) * 0.3
    
    def get_boundary_explanations(self, 
                                 result: HybridBoundaryResult) -> Dict[int, List[str]]:
        """Get human-readable explanations for boundary decisions.
        
        Args:
            result: Hybrid boundary result
            
        Returns:
            Dict mapping page numbers to explanation lists
        """
        explanations = {}
        
        if result.visual_scores:
            for score in result.visual_scores:
                if score.reasons:
                    explanations[score.page_num] = score.reasons
        
        # Add detection level info
        for boundary_start, _ in result.boundaries:
            if boundary_start not in explanations:
                explanations[boundary_start] = []
            explanations[boundary_start].insert(
                0, f"Detected using {result.detection_level.name.lower()} analysis"
            )
        
        return explanations