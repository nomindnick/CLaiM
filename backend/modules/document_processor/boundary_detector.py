"""Document boundary detection for PDF splitting."""

import re
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from loguru import logger


class BoundaryDetector:
    """Detects document boundaries within large PDF files."""
    
    def __init__(self):
        # Common document start patterns
        self.start_patterns = [
            # Email patterns
            re.compile(r"^From:\s*", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Subject:\s*", re.IGNORECASE | re.MULTILINE),
            
            # Letter/memo patterns
            re.compile(r"^(MEMORANDUM|MEMO)\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Dear\s+", re.IGNORECASE),
            
            # RFI patterns
            re.compile(r"REQUEST\s+FOR\s+INFORMATION", re.IGNORECASE),
            re.compile(r"RFI\s*#?\s*\d+", re.IGNORECASE),
            
            # Change order patterns
            re.compile(r"CHANGE\s+ORDER", re.IGNORECASE),
            re.compile(r"C\.O\.\s*#?\s*\d+", re.IGNORECASE),
            
            # Invoice patterns
            re.compile(r"^INVOICE\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"Invoice\s*#?\s*\d+", re.IGNORECASE),
            
            # Submittal patterns
            re.compile(r"^SUBMITTAL", re.IGNORECASE | re.MULTILINE),
            
            # General document headers
            re.compile(r"^[A-Z\s]{3,50}$", re.MULTILINE),  # All caps headers
        ]
        
        # Page break indicators
        self.page_break_patterns = [
            re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
            re.compile(r"^\d+\s*$"),  # Just page numbers
            re.compile(r"-\s*\d+\s*-"),  # Centered page numbers
        ]
    
    def detect_boundaries(self, pdf_doc: fitz.Document) -> List[Tuple[int, int]]:
        """Detect document boundaries in a PDF.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            List of (start_page, end_page) tuples for each document
        """
        boundaries = []
        current_start = 0
        
        logger.debug(f"Detecting boundaries in {pdf_doc.page_count} pages")
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text = page.get_text()
            
            # Check if this page starts a new document
            if page_num > 0 and self._is_document_start(text, page):
                # End previous document
                boundaries.append((current_start, page_num - 1))
                current_start = page_num
        
        # Add the last document
        if current_start < pdf_doc.page_count:
            boundaries.append((current_start, pdf_doc.page_count - 1))
        
        # Post-process to merge single-page fragments
        boundaries = self._merge_fragments(boundaries)
        
        logger.debug(f"Found {len(boundaries)} documents")
        return boundaries
    
    def _is_document_start(self, text: str, page: fitz.Page) -> bool:
        """Check if a page represents the start of a new document.
        
        Args:
            text: Extracted text from the page
            page: PyMuPDF page object
            
        Returns:
            True if this appears to be a document start
        """
        # Skip mostly empty pages
        if len(text.strip()) < 50:
            return False
        
        # Check for document start patterns
        for pattern in self.start_patterns:
            if pattern.search(text[:500]):  # Check first 500 chars
                return True
        
        # Check for specific visual cues
        if self._has_letterhead(page):
            return True
        
        # Check for significant formatting changes
        if self._has_format_change(page):
            return True
        
        return False
    
    def _has_letterhead(self, page: fitz.Page) -> bool:
        """Check if page has a letterhead (logo/header).
        
        Simple heuristic: check for images in top portion of page.
        """
        page_rect = page.rect
        top_region = fitz.Rect(
            page_rect.x0,
            page_rect.y0,
            page_rect.x1,
            page_rect.y0 + (page_rect.height * 0.2)  # Top 20%
        )
        
        # Check for images in top region
        for img in page.get_images():
            img_rect = page.get_image_bbox(img[0])
            if img_rect and top_region.intersects(img_rect):
                return True
        
        return False
    
    def _has_format_change(self, page: fitz.Page) -> bool:
        """Check for significant formatting changes.
        
        This is a placeholder for more sophisticated analysis.
        """
        # TODO: Implement format change detection
        # - Font changes
        # - Layout changes
        # - Margin changes
        return False
    
    def _merge_fragments(self, boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge single-page fragments with adjacent documents.
        
        Args:
            boundaries: Initial boundary list
            
        Returns:
            Merged boundary list
        """
        if len(boundaries) <= 1:
            return boundaries
        
        merged = []
        i = 0
        
        while i < len(boundaries):
            start, end = boundaries[i]
            
            # If this is a single-page document, consider merging
            if end - start == 0 and i < len(boundaries) - 1:
                next_start, next_end = boundaries[i + 1]
                
                # If next document is also small, might be part of same document
                if next_end - next_start <= 2:
                    # Merge them
                    merged.append((start, next_end))
                    i += 2
                    continue
            
            merged.append((start, end))
            i += 1
        
        return merged