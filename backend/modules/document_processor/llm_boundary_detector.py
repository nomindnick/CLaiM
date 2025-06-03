"""LLM-based document boundary detection.

This module implements a semantic, context-aware approach to document boundary
detection using LLMs instead of pattern matching.
"""

import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

import fitz
from loguru import logger

from ..llm_client.base_client import BaseLLMClient
from ..llm_client.ollama_client import OllamaClient
from .improved_ocr_handler import ImprovedOCRHandler
from .models import DocumentType


class ConfidenceLevel(Enum):
    """Confidence levels for boundary detection."""
    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.6 - 0.8  
    LOW = "low"  # < 0.6


@dataclass
class BoundaryAnalysis:
    """Result of boundary analysis for a page."""
    page_num: int
    is_boundary: bool
    confidence: float
    document_type: Optional[str]
    reasoning: str
    continues_from_previous: bool
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.6:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


@dataclass
class WindowAnalysis:
    """Analysis results for a window of pages."""
    window_start: int
    window_end: int
    boundaries: List[BoundaryAnalysis]
    window_summary: str
    avg_confidence: float


class LLMBoundaryDetector:
    """Document boundary detection using LLM for semantic understanding."""
    
    def __init__(
        self, 
        llm_client: Optional[BaseLLMClient] = None,
        ocr_handler: Optional[ImprovedOCRHandler] = None,
        window_size: int = 3,
        overlap: int = 1,
        confidence_threshold: float = 0.7
    ):
        """Initialize LLM boundary detector.
        
        Args:
            llm_client: LLM client for analysis
            ocr_handler: OCR handler for scanned pages
            window_size: Number of pages to analyze together
            overlap: Number of pages to overlap between windows
            confidence_threshold: Minimum confidence for boundary detection
        """
        self.llm_client = llm_client or OllamaClient()
        self.ocr_handler = ocr_handler or ImprovedOCRHandler()
        self.window_size = window_size
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        
        # Cache for OCR results
        self._ocr_cache = {}
        
    def detect_boundaries(self, pdf_doc: fitz.Document) -> List[Tuple[int, int]]:
        """Detect document boundaries in a PDF using LLM analysis.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            List of (start_page, end_page) tuples for each document
        """
        logger.info(f"Starting LLM boundary detection for {pdf_doc.page_count} pages")
        
        # Step 1: Extract page windows with overlap
        windows = self._extract_page_windows(pdf_doc)
        logger.info(f"Created {len(windows)} analysis windows")
        
        # Step 2: Analyze each window with LLM
        window_analyses = []
        previous_summary = None
        
        for i, window in enumerate(windows):
            logger.info(f"Analyzing window {i+1}/{len(windows)}")
            analysis = self._analyze_window(window, previous_summary)
            window_analyses.append(analysis)
            previous_summary = analysis.window_summary
            
        # Step 3: Consolidate boundaries from overlapping analyses
        boundaries = self._consolidate_boundaries(window_analyses, pdf_doc.page_count)
        
        # Step 4: Validate and refine boundaries
        boundaries = self._validate_boundaries(boundaries, pdf_doc)
        
        logger.info(f"Detected {len(boundaries)} documents with LLM analysis")
        return boundaries
    
    def _extract_page_windows(self, pdf_doc: fitz.Document) -> List[List[Dict]]:
        """Extract overlapping windows of pages for analysis.
        
        Args:
            pdf_doc: PDF document
            
        Returns:
            List of page windows, each containing page info
        """
        windows = []
        
        # Calculate window positions with overlap
        step = self.window_size - self.overlap
        
        for start in range(0, pdf_doc.page_count, step):
            window_pages = []
            end = min(start + self.window_size, pdf_doc.page_count)
            
            for page_num in range(start, end):
                page = pdf_doc[page_num]
                text = self._get_page_text(page, page_num)
                
                # Extract page features
                page_info = {
                    'page_num': page_num,
                    'text': text,
                    'text_length': len(text.strip()),
                    'has_images': len(page.get_images()) > 0,
                    'has_many_drawings': len(page.get_drawings()) > 50,
                    'font_info': self._extract_font_info(page),
                    'layout_info': self._extract_layout_info(page)
                }
                
                window_pages.append(page_info)
                
            windows.append(window_pages)
            
        return windows
    
    def _get_page_text(self, page: fitz.Page, page_num: int) -> str:
        """Get text from page, using OCR if needed.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number for caching
            
        Returns:
            Extracted text
        """
        # Check cache first
        if page_num in self._ocr_cache:
            return self._ocr_cache[page_num]
            
        text = page.get_text()
        
        # Use OCR for scanned pages
        if len(text.strip()) < 10 and self.ocr_handler:
            try:
                ocr_text, confidence = self.ocr_handler.process_page(page, dpi=200)
                if confidence > 0.3:
                    text = ocr_text
                    self._ocr_cache[page_num] = text
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                
        return text
    
    def _extract_font_info(self, page: fitz.Page) -> Dict[str, Any]:
        """Extract font information from page."""
        try:
            fonts = page.get_fonts()
            return {
                'num_fonts': len(fonts),
                'has_multiple_fonts': len(fonts) > 2
            }
        except:
            return {'num_fonts': 0, 'has_multiple_fonts': False}
    
    def _extract_layout_info(self, page: fitz.Page) -> Dict[str, Any]:
        """Extract layout information from page."""
        try:
            blocks = page.get_text("blocks")
            return {
                'num_blocks': len(blocks),
                'is_dense': len(blocks) > 20,
                'is_sparse': len(blocks) < 5
            }
        except:
            return {'num_blocks': 0, 'is_dense': False, 'is_sparse': True}
    
    def _analyze_window(
        self, 
        window_pages: List[Dict], 
        previous_summary: Optional[str] = None
    ) -> WindowAnalysis:
        """Analyze a window of pages for document boundaries.
        
        Args:
            window_pages: List of page information
            previous_summary: Summary from previous window
            
        Returns:
            Window analysis with boundary detections
        """
        # Build context for LLM
        window_text = self._build_window_context(window_pages)
        
        # Create analysis prompt
        prompt = self._build_analysis_prompt(window_text, previous_summary)
        
        # Get LLM analysis
        try:
            response = self.llm_client.complete(prompt)
            analysis_data = self._parse_llm_response(response)
            
            # Convert to BoundaryAnalysis objects
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
                
            # Calculate average confidence
            avg_confidence = sum(b.confidence for b in boundaries) / len(boundaries) if boundaries else 0
            
            return WindowAnalysis(
                window_start=window_pages[0]['page_num'],
                window_end=window_pages[-1]['page_num'],
                boundaries=boundaries,
                window_summary=analysis_data.get('window_summary', ''),
                avg_confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback to simple analysis
            return self._fallback_analysis(window_pages)
    
    def _build_window_context(self, window_pages: List[Dict]) -> str:
        """Build context string for window analysis."""
        context_parts = []
        
        for page in window_pages:
            page_num = page['page_num'] + 1  # Human-readable
            
            # Truncate text for context
            text_preview = page['text'][:1000] if len(page['text']) > 1000 else page['text']
            text_preview = text_preview.replace('\n', ' ')
            
            page_context = f"""
Page {page_num}:
- Text length: {page['text_length']} characters
- Has images: {page['has_images']}
- Dense layout: {page['layout_info']['is_dense']}
- Text preview: {text_preview}
"""
            context_parts.append(page_context)
            
        return "\n".join(context_parts)
    
    def _build_analysis_prompt(
        self, 
        window_text: str, 
        previous_summary: Optional[str] = None
    ) -> str:
        """Build prompt for LLM boundary analysis."""
        prompt = f"""You are analyzing pages from a construction litigation PDF to identify document boundaries.
Your task is to determine where one document ends and another begins.

IMPORTANT: Construction documents often span multiple pages:
- Emails with attachments/quotes can be 2-4 pages
- RFIs and submittals are typically multi-page
- Invoices often have multiple pages of line items
- Technical drawings come in sets
- Payment applications have continuation sheets

Window of pages to analyze:
{window_text}

{'Previous window summary: ' + previous_summary if previous_summary else 'This is the first window.'}

For EACH page in this window, analyze if it starts a new document.

Consider these indicators for NEW documents:
1. Email headers (From:, To:, Subject:) that are NOT quoted/forwarded
2. Document headers (RFI #, Invoice #, Submittal #)
3. Major format changes (email to drawing, invoice to RFI)
4. New letterhead or company info
5. Date jumps indicating different time periods

Consider these indicators for CONTINUED documents:
1. "Page X of Y" patterns
2. Continued item lists or tables
3. Email quotes/forwards (>, >>, etc.)
4. Consistent headers/footers
5. References to "continued from previous page"
6. Same invoice/RFI/submittal number

Return a JSON object with this structure:
{{
    "boundaries": [
        {{
            "page_num": 0,  // 0-based page number
            "is_boundary": true/false,  // Is this the start of a new document?
            "confidence": 0.0-1.0,  // How confident are you?
            "document_type": "email/rfi/invoice/submittal/drawing/etc",  // If new document
            "reasoning": "Brief explanation",
            "continues_from_previous": true/false  // Does this continue from previous window?
        }}
        // ... for each page
    ],
    "window_summary": "Brief summary of documents in this window"
}}

Be conservative - only mark clear document boundaries. When in doubt, assume continuation."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data."""
        try:
            # Try to extract JSON from response
            # Handle case where LLM includes markdown formatting
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
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")
            # Return empty analysis
            return {"boundaries": [], "window_summary": ""}
    
    def _fallback_analysis(self, window_pages: List[Dict]) -> WindowAnalysis:
        """Simple fallback analysis when LLM fails."""
        boundaries = []
        
        for page in window_pages:
            # Simple heuristics
            text = page['text'][:500].lower()
            is_boundary = any([
                'from:' in text and '@' in text,
                'invoice' in text and '#' in text,
                'rfi' in text and '#' in text,
                'submittal' in text
            ])
            
            boundaries.append(BoundaryAnalysis(
                page_num=page['page_num'],
                is_boundary=is_boundary,
                confidence=0.5,  # Low confidence for fallback
                document_type=None,
                reasoning="Fallback detection",
                continues_from_previous=False
            ))
            
        return WindowAnalysis(
            window_start=window_pages[0]['page_num'],
            window_end=window_pages[-1]['page_num'],
            boundaries=boundaries,
            window_summary="Fallback analysis",
            avg_confidence=0.5
        )
    
    def _consolidate_boundaries(
        self, 
        window_analyses: List[WindowAnalysis],
        total_pages: int
    ) -> List[Tuple[int, int]]:
        """Consolidate boundaries from overlapping window analyses.
        
        Args:
            window_analyses: List of window analysis results
            total_pages: Total pages in document
            
        Returns:
            List of (start_page, end_page) tuples
        """
        # Aggregate votes for each page
        boundary_votes = defaultdict(list)
        
        for analysis in window_analyses:
            for boundary in analysis.boundaries:
                if boundary.is_boundary:
                    boundary_votes[boundary.page_num].append({
                        'confidence': boundary.confidence,
                        'type': boundary.document_type,
                        'reasoning': boundary.reasoning
                    })
        
        # Determine final boundaries based on consensus
        boundary_pages = []
        
        for page_num, votes in boundary_votes.items():
            # Average confidence across all votes
            avg_confidence = sum(v['confidence'] for v in votes) / len(votes)
            
            if avg_confidence >= self.confidence_threshold:
                # Majority vote for document type
                if any(v['type'] for v in votes):
                    type_votes = Counter(v['type'] for v in votes if v['type'])
                    if type_votes:
                        most_common_type = type_votes.most_common(1)[0][0]
                    else:
                        most_common_type = None
                else:
                    most_common_type = None
                    
                boundary_pages.append(page_num)
                logger.info(
                    f"Page {page_num + 1} confirmed as boundary "
                    f"(confidence: {avg_confidence:.2f}, type: {most_common_type})"
                )
        
        # Always include page 0 as a boundary
        if 0 not in boundary_pages:
            boundary_pages.append(0)
            
        # Sort boundaries
        boundary_pages.sort()
        
        # Convert to (start, end) tuples
        boundaries = []
        for i in range(len(boundary_pages)):
            start = boundary_pages[i]
            if i < len(boundary_pages) - 1:
                end = boundary_pages[i + 1] - 1
            else:
                end = total_pages - 1
            boundaries.append((start, end))
            
        return boundaries
    
    def _validate_boundaries(
        self, 
        boundaries: List[Tuple[int, int]], 
        pdf_doc: fitz.Document
    ) -> List[Tuple[int, int]]:
        """Validate and refine detected boundaries.
        
        Args:
            boundaries: Initial boundary list
            pdf_doc: PDF document
            
        Returns:
            Refined boundary list
        """
        if len(boundaries) <= 1:
            return boundaries
            
        refined = []
        i = 0
        
        while i < len(boundaries):
            start, end = boundaries[i]
            
            # Check for very short documents that might be fragments
            if end - start < 2 and i < len(boundaries) - 1:
                # Analyze if this should be merged with next
                should_merge = self._should_merge_fragments(
                    pdf_doc, start, end, 
                    boundaries[i + 1][0], boundaries[i + 1][1]
                )
                
                if should_merge:
                    # Merge with next document
                    next_start, next_end = boundaries[i + 1]
                    refined.append((start, next_end))
                    i += 2
                    continue
                    
            refined.append((start, end))
            i += 1
            
        return refined
    
    def _should_merge_fragments(
        self,
        pdf_doc: fitz.Document,
        start1: int, end1: int,
        start2: int, end2: int
    ) -> bool:
        """Determine if two document fragments should be merged.
        
        Uses LLM to analyze if documents are related.
        """
        # Get text from both fragments
        text1 = ""
        for page_num in range(start1, end1 + 1):
            text1 += self._get_page_text(pdf_doc[page_num], page_num)
            
        text2 = ""
        for page_num in range(start2, min(start2 + 2, end2 + 1)):  # First 2 pages
            text2 += self._get_page_text(pdf_doc[page_num], page_num)
            
        # Ask LLM if these should be merged
        prompt = f"""Analyze if these two document fragments should be merged into one document:

Fragment 1 ({end1 - start1 + 1} pages):
{text1[:500]}

Fragment 2 (starts at page {start2 + 1}):
{text2[:500]}

Consider:
1. Is Fragment 2 a continuation of Fragment 1?
2. Do they share the same document type/purpose?
3. Are there references between them?

Return JSON: {{"should_merge": true/false, "confidence": 0.0-1.0, "reason": "..."}}"""

        try:
            response = self.llm_client.complete(prompt)
            result = self._parse_llm_response(response)
            return result.get('should_merge', False) and result.get('confidence', 0) > 0.7
        except:
            return False