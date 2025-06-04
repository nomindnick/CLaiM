"""Two-stage LLM boundary detection for performance optimization.

This module implements a fast pre-filtering approach using a small model (phi-3-mini)
followed by detailed analysis with a larger model for uncertain areas.
"""

import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

import fitz
from loguru import logger

from ..llm_client.base_client import LLMClient
from ..llm_client.ollama_client import OllamaClient
from .improved_ocr_handler import ImprovedOCRHandler
from .models import DocumentType
from .llm_boundary_detector import BoundaryAnalysis, WindowAnalysis, ConfidenceLevel


@dataclass
class FastScreenResult:
    """Result from fast pre-screening pass."""
    page_num: int
    likely_boundary: bool
    confidence: float
    needs_deep_analysis: bool
    document_hint: Optional[str] = None


@dataclass
class BatchWindow:
    """Window for batch processing."""
    window_id: int
    start_page: int
    end_page: int
    pages: List[Dict[str, Any]]
    priority: float = 1.0  # Higher priority for uncertain regions


class TwoStageDetector:
    """Two-stage document boundary detection for improved performance."""
    
    def __init__(
        self,
        fast_model: str = "phi3:mini",
        deep_model: str = "llama3:8b-instruct-q4_0",
        ocr_handler: Optional[ImprovedOCRHandler] = None,
        window_size: int = 3,
        confidence_threshold: float = 0.7,
        batch_size: int = 5,
        keep_alive_minutes: int = 10
    ):
        """Initialize two-stage detector.
        
        Args:
            fast_model: Small model for quick screening (phi-3-mini)
            deep_model: Larger model for detailed analysis
            ocr_handler: OCR handler for scanned pages
            window_size: Number of pages to analyze together
            confidence_threshold: Minimum confidence for boundary detection
            batch_size: Number of windows to process in parallel
            keep_alive_minutes: Minutes to keep models loaded in memory
        """
        # Initialize LLM clients with keep-alive
        self.fast_client = OllamaClient(
            model_name=fast_model,
            timeout=120,  # Increased timeout for fast model
            keep_alive=f"{keep_alive_minutes}m"
        )
        
        self.deep_client = OllamaClient(
            model_name=deep_model,
            timeout=300,
            keep_alive=f"{keep_alive_minutes}m"
        )
        
        self.ocr_handler = ocr_handler or ImprovedOCRHandler()
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Cache for OCR and page analysis
        self._ocr_cache = {}
        self._page_features_cache = {}
        
        # Smart overlap settings
        self.min_overlap = 1
        self.max_overlap = 2
        
    def detect_boundaries(self, pdf_doc: fitz.Document) -> List[Tuple[int, int]]:
        """Detect document boundaries using two-stage approach.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            List of (start_page, end_page) tuples for each document
        """
        logger.info(f"Starting two-stage boundary detection for {pdf_doc.page_count} pages")
        
        # Pre-load models to ensure they're ready
        self._preload_models()
        
        # Stage 1: Fast screening of all pages
        start_time = time.time()
        screening_results = self._fast_screening_pass(pdf_doc)
        stage1_time = time.time() - start_time
        logger.info(f"Stage 1 (fast screening) completed in {stage1_time:.2f}s")
        
        # Stage 2: Create smart windows based on screening results
        windows = self._create_smart_windows(screening_results, pdf_doc)
        logger.info(f"Created {len(windows)} analysis windows")
        
        # Stage 3: Batch process windows with deep analysis where needed
        start_time = time.time()
        window_analyses = self._batch_process_windows(windows, screening_results)
        stage2_time = time.time() - start_time
        logger.info(f"Stage 2 (deep analysis) completed in {stage2_time:.2f}s")
        
        # Stage 4: Consolidate results
        boundaries = self._consolidate_results(window_analyses, screening_results, pdf_doc.page_count)
        
        total_time = stage1_time + stage2_time
        logger.info(f"Detected {len(boundaries)} documents in {total_time:.2f}s total")
        logger.info(f"Performance: {pdf_doc.page_count / total_time:.1f} pages/second")
        
        return boundaries
    
    def _preload_models(self):
        """Pre-load models to avoid cold start delays."""
        logger.info("Pre-loading models...")
        
        # Warm up fast model
        try:
            self.fast_client.complete("Hello", timeout=30)
            logger.info("✓ Fast model (phi-3-mini) ready")
        except Exception as e:
            logger.warning(f"Fast model not ready: {e}")
            
        # Warm up deep model
        try:
            self.deep_client.complete("Hello", timeout=30)
            logger.info("✓ Deep model ready")
        except Exception as e:
            logger.warning(f"Deep model not ready: {e}")
    
    def _fast_screening_pass(self, pdf_doc: fitz.Document) -> List[FastScreenResult]:
        """Fast screening pass using small model.
        
        Args:
            pdf_doc: PDF document
            
        Returns:
            List of screening results for each page
        """
        results = []
        
        # Process pages in small batches for efficiency
        batch_size = 5  # Reduced batch size for faster processing
        for batch_start in range(0, pdf_doc.page_count, batch_size):
            batch_end = min(batch_start + batch_size, pdf_doc.page_count)
            
            # Build batch context
            batch_pages = []
            for page_num in range(batch_start, batch_end):
                page = pdf_doc[page_num]
                page_features = self._extract_page_features(page, page_num)
                batch_pages.append(page_features)
            
            # Fast analysis prompt
            prompt = self._build_fast_screening_prompt(batch_pages)
            
            try:
                response = self.fast_client.complete(prompt, timeout=120)
                batch_results = self._parse_screening_response(response, batch_start)
                results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"Fast screening failed for batch {batch_start}-{batch_end}: {e}")
                # Fallback: mark all as needing deep analysis
                for page_num in range(batch_start, batch_end):
                    results.append(FastScreenResult(
                        page_num=page_num,
                        likely_boundary=page_num % 5 == 0,  # Simple heuristic
                        confidence=0.3,
                        needs_deep_analysis=True
                    ))
        
        return results
    
    def _extract_page_features(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Extract page features with caching."""
        if page_num in self._page_features_cache:
            return self._page_features_cache[page_num]
        
        # Get text (with OCR if needed)
        text = self._get_page_text(page, page_num)
        
        # Extract quick features
        features = {
            'page_num': page_num,
            'text_preview': text[:150],  # Reduced to 150 chars for faster processing
            'text_length': len(text.strip()),
            'has_images': len(page.get_images()) > 0,
            'has_tables': self._quick_table_check(text),
            'has_email_header': self._quick_email_check(text),
            'has_doc_header': self._quick_doc_header_check(text),
            'font_count': len(page.get_fonts()),
            'block_count': len(page.get_text("blocks"))
        }
        
        self._page_features_cache[page_num] = features
        return features
    
    def _get_page_text(self, page: fitz.Page, page_num: int) -> str:
        """Get text from page with caching."""
        if page_num in self._ocr_cache:
            return self._ocr_cache[page_num]
            
        text = page.get_text()
        
        # Use OCR for scanned pages
        if len(text.strip()) < 10 and self.ocr_handler:
            try:
                ocr_text, confidence = self.ocr_handler.process_page(page, dpi=150)  # Lower DPI for speed
                if confidence > 0.3:
                    text = ocr_text
            except Exception as e:
                logger.debug(f"OCR skipped for page {page_num}: {e}")
                
        self._ocr_cache[page_num] = text
        return text
    
    def _quick_table_check(self, text: str) -> bool:
        """Quick check for table-like content."""
        lines = text.split('\n')[:20]  # Check first 20 lines
        pipe_count = sum(1 for line in lines if '|' in line)
        tab_count = sum(1 for line in lines if '\t' in line)
        return pipe_count > 3 or tab_count > 5
    
    def _quick_email_check(self, text: str) -> bool:
        """Quick check for email headers."""
        text_lower = text[:500].lower()
        return any(pattern in text_lower for pattern in ['from:', 'to:', 'subject:', 'sent:'])
    
    def _quick_doc_header_check(self, text: str) -> bool:
        """Quick check for document headers."""
        text_lower = text[:500].lower()
        patterns = ['invoice', 'rfi', 'submittal', 'change order', 'daily report', 'notice']
        return any(pattern in text_lower for pattern in patterns)
    
    def _build_fast_screening_prompt(self, batch_pages: List[Dict]) -> str:
        """Build prompt for fast screening."""
        pages_text = []
        for page in batch_pages:
            # Simplified page info for faster processing
            pages_text.append(f"Page {page['page_num'] + 1}: Email={page['has_email_header']}, Doc={page['has_doc_header']}, Len={page['text_length']}")
        
        prompt = f"""RETURN ONLY JSON. NO EXPLANATION.

{chr(10).join(pages_text)}

Output JSON array marking document boundaries:
[{{"page": 0, "likely_boundary": true, "confidence": 0.8, "needs_deep_analysis": false}}]

Rules: Email=True means boundary. Doc=True means boundary."""
        
        return prompt
    
    def _parse_screening_response(self, response: str, batch_start: int) -> List[FastScreenResult]:
        """Parse fast screening response."""
        try:
            # Extract JSON more robustly
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # Find JSON array in response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            data = json.loads(response)
            
            results = []
            for item in data:
                results.append(FastScreenResult(
                    page_num=item['page'] + batch_start,
                    likely_boundary=item.get('likely_boundary', False),
                    confidence=item.get('confidence', 0.5),
                    needs_deep_analysis=item.get('needs_deep_analysis', True),
                    document_hint=item.get('hint')
                ))
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse screening response: {e}")
            return []
    
    def _create_smart_windows(
        self, 
        screening_results: List[FastScreenResult],
        pdf_doc: fitz.Document
    ) -> List[BatchWindow]:
        """Create analysis windows with smart overlap based on confidence."""
        windows = []
        window_id = 0
        
        # Group pages by uncertainty regions
        uncertainty_regions = []
        current_region = None
        
        for i, result in enumerate(screening_results):
            if result.needs_deep_analysis or result.confidence < 0.7:
                if current_region is None:
                    current_region = {'start': i, 'end': i, 'avg_confidence': result.confidence}
                else:
                    current_region['end'] = i
                    current_region['avg_confidence'] = (
                        current_region['avg_confidence'] + result.confidence
                    ) / 2
            else:
                if current_region is not None:
                    uncertainty_regions.append(current_region)
                    current_region = None
        
        if current_region is not None:
            uncertainty_regions.append(current_region)
        
        # Create windows with dynamic overlap
        i = 0
        while i < len(screening_results):
            # Check if we're in an uncertainty region
            in_uncertain_region = any(
                region['start'] <= i <= region['end'] 
                for region in uncertainty_regions
            )
            
            # Determine window size and overlap
            if in_uncertain_region:
                # Larger overlap for uncertain regions
                overlap = self.max_overlap
                priority = 2.0
            else:
                # Minimal overlap for confident regions
                overlap = self.min_overlap
                priority = 1.0
            
            # Create window
            window_end = min(i + self.window_size, len(screening_results))
            
            # Extract page features for window
            window_pages = []
            for page_num in range(i, window_end):
                if page_num < pdf_doc.page_count:
                    page = pdf_doc[page_num]
                    features = self._extract_page_features(page, page_num)
                    window_pages.append(features)
            
            windows.append(BatchWindow(
                window_id=window_id,
                start_page=i,
                end_page=window_end - 1,
                pages=window_pages,
                priority=priority
            ))
            
            window_id += 1
            
            # Move to next window with smart overlap
            i += self.window_size - overlap
        
        # Sort windows by priority (process uncertain regions first)
        windows.sort(key=lambda w: w.priority, reverse=True)
        
        return windows
    
    def _batch_process_windows(
        self,
        windows: List[BatchWindow],
        screening_results: List[FastScreenResult]
    ) -> List[WindowAnalysis]:
        """Process windows in batches for efficiency."""
        analyses = []
        
        # Process windows in batches
        for batch_start in range(0, len(windows), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(windows))
            batch_windows = windows[batch_start:batch_end]
            
            logger.info(f"Processing window batch {batch_start + 1}-{batch_end} of {len(windows)}")
            
            # Process each window in the batch
            # In a production system, these could be processed in parallel
            for window in batch_windows:
                # Check if deep analysis is needed
                window_screening = screening_results[window.start_page:window.end_page + 1]
                needs_deep = any(s.needs_deep_analysis for s in window_screening)
                
                if needs_deep or window.priority > 1.0:
                    # Use deep model for uncertain regions
                    analysis = self._deep_analyze_window(window, screening_results)
                else:
                    # Convert screening results to analysis
                    analysis = self._convert_screening_to_analysis(window, window_screening)
                
                analyses.append(analysis)
        
        # Sort analyses back to page order
        analyses.sort(key=lambda a: a.window_start)
        
        return analyses
    
    def _deep_analyze_window(
        self,
        window: BatchWindow,
        screening_results: List[FastScreenResult]
    ) -> WindowAnalysis:
        """Perform deep analysis on a window using the larger model."""
        # Build detailed context
        context = self._build_deep_analysis_context(window, screening_results)
        
        # Create focused prompt
        prompt = self._build_deep_analysis_prompt(context)
        
        try:
            response = self.deep_client.complete(prompt, timeout=120)
            analysis_data = self._parse_deep_response(response)
            
            # Convert to WindowAnalysis
            boundaries = []
            for boundary_data in analysis_data.get('boundaries', []):
                boundaries.append(BoundaryAnalysis(
                    page_num=boundary_data['page_num'],
                    is_boundary=boundary_data['is_boundary'],
                    confidence=boundary_data['confidence'],
                    document_type=boundary_data.get('document_type'),
                    reasoning=boundary_data.get('reasoning', ''),
                    continues_from_previous=boundary_data.get('continues_from_previous', False)
                ))
            
            avg_confidence = sum(b.confidence for b in boundaries) / len(boundaries) if boundaries else 0
            
            return WindowAnalysis(
                window_start=window.start_page,
                window_end=window.end_page,
                boundaries=boundaries,
                window_summary=analysis_data.get('summary', ''),
                avg_confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"Deep analysis failed for window {window.window_id}: {e}")
            # Fallback to screening results
            return self._convert_screening_to_analysis(
                window, 
                screening_results[window.start_page:window.end_page + 1]
            )
    
    def _build_deep_analysis_context(
        self,
        window: BatchWindow,
        screening_results: List[FastScreenResult]
    ) -> str:
        """Build context for deep analysis."""
        context_parts = []
        
        for page in window.pages:
            page_num = page['page_num']
            screening = screening_results[page_num]
            
            # Include screening hints
            hints = []
            if screening.likely_boundary:
                hints.append(f"likely boundary (confidence: {screening.confidence:.2f})")
            if screening.document_hint:
                hints.append(f"possible {screening.document_hint}")
            
            page_context = f"""
Page {page_num + 1}:
- Screening: {', '.join(hints) if hints else 'no clear indicators'}
- Text length: {page['text_length']} chars
- Has images: {page['has_images']}
- Text: {page['text_preview']}
"""
            context_parts.append(page_context)
        
        return '\n'.join(context_parts)
    
    def _build_deep_analysis_prompt(self, context: str) -> str:
        """Build prompt for deep analysis."""
        return f"""Detailed document boundary analysis for uncertain region.

{context}

Carefully analyze these pages to identify document boundaries.
Focus on:
1. Document headers and footers
2. Continuation patterns (page X of Y, continued...)
3. Format and style changes
4. Logical content flow

Return JSON:
{{
  "boundaries": [
    {{
      "page_num": 0,  // 0-based
      "is_boundary": true/false,
      "confidence": 0.0-1.0,
      "document_type": "type",
      "reasoning": "explanation",
      "continues_from_previous": true/false
    }}
  ],
  "summary": "Brief summary of findings"
}}"""
    
    def _parse_deep_response(self, response: str) -> Dict[str, Any]:
        """Parse deep analysis response."""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end]
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end]
            
            return json.loads(response.strip())
        except Exception as e:
            logger.error(f"Failed to parse deep response: {e}")
            return {"boundaries": [], "summary": ""}
    
    def _convert_screening_to_analysis(
        self,
        window: BatchWindow,
        screening_results: List[FastScreenResult]
    ) -> WindowAnalysis:
        """Convert screening results to window analysis."""
        boundaries = []
        
        for i, screening in enumerate(screening_results):
            boundaries.append(BoundaryAnalysis(
                page_num=window.start_page + i,
                is_boundary=screening.likely_boundary,
                confidence=screening.confidence,
                document_type=screening.document_hint,
                reasoning="Fast screening result",
                continues_from_previous=False
            ))
        
        avg_confidence = sum(b.confidence for b in boundaries) / len(boundaries) if boundaries else 0
        
        return WindowAnalysis(
            window_start=window.start_page,
            window_end=window.end_page,
            boundaries=boundaries,
            window_summary="Fast screening analysis",
            avg_confidence=avg_confidence
        )
    
    def _consolidate_results(
        self,
        window_analyses: List[WindowAnalysis],
        screening_results: List[FastScreenResult],
        total_pages: int
    ) -> List[Tuple[int, int]]:
        """Consolidate results from both stages."""
        # Aggregate boundary votes
        boundary_votes = defaultdict(list)
        
        # Add deep analysis votes
        for analysis in window_analyses:
            for boundary in analysis.boundaries:
                if boundary.is_boundary:
                    boundary_votes[boundary.page_num].append({
                        'confidence': boundary.confidence,
                        'type': boundary.document_type,
                        'source': 'deep'
                    })
        
        # Add screening votes for high-confidence results
        for screening in screening_results:
            if screening.likely_boundary and screening.confidence > 0.8:
                boundary_votes[screening.page_num].append({
                    'confidence': screening.confidence,
                    'type': screening.document_hint,
                    'source': 'screening'
                })
        
        # Determine final boundaries
        boundary_pages = []
        
        for page_num, votes in boundary_votes.items():
            # Weight deep analysis higher
            weighted_confidence = sum(
                v['confidence'] * (1.5 if v['source'] == 'deep' else 1.0)
                for v in votes
            ) / sum(1.5 if v['source'] == 'deep' else 1.0 for v in votes)
            
            if weighted_confidence >= self.confidence_threshold:
                boundary_pages.append(page_num)
                logger.info(f"Page {page_num + 1} confirmed as boundary (confidence: {weighted_confidence:.2f})")
        
        # Always include page 0
        if 0 not in boundary_pages:
            boundary_pages.append(0)
        
        boundary_pages.sort()
        
        # Convert to (start, end) tuples
        boundaries = []
        for i in range(len(boundary_pages)):
            start = boundary_pages[i]
            end = boundary_pages[i + 1] - 1 if i < len(boundary_pages) - 1 else total_pages - 1
            boundaries.append((start, end))
        
        return boundaries