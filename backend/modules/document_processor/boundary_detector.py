"""Document boundary detection for PDF splitting."""

import re
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from loguru import logger

from .ocr_handler import OCRHandler
from .construction_patterns import detect_document_type, is_strong_document_start


class BoundaryDetector:
    """Detects document boundaries within large PDF files."""
    
    def __init__(self, ocr_handler: Optional[OCRHandler] = None):
        self.ocr_handler = ocr_handler
        # Common document start patterns - prioritized by strength
        self.start_patterns = [
            # Email patterns - very strong indicators
            re.compile(r"^From:\s*.*@", re.IGNORECASE | re.MULTILINE),  # From: with email
            re.compile(r"^To:\s*.*@", re.IGNORECASE | re.MULTILINE),    # To: with email
            re.compile(r"^Sent:\s*\w+,\s*\w+\s*\d+", re.IGNORECASE | re.MULTILINE),  # Sent: Monday, February 26
            re.compile(r"^Subject:\s*", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Date:\s*\w+,\s*\w+\s*\d+", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^(RE:|FW:|Fwd:)\s*", re.IGNORECASE | re.MULTILINE),
            
            # Submittal/Transmittal patterns
            re.compile(r"^\s*SUBMITTAL\s*TRANSMITTAL", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^SUBMITTAL\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^TRANSMITTAL\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"SHOP\s+DRAWING.*SUBMITTAL", re.IGNORECASE),
            re.compile(r"Reference\s*Number:\s*\d+", re.IGNORECASE),
            
            # Payment/SOV patterns
            re.compile(r"APPLICATION\s+AND\s+CERTIFICATE\s+FOR\s+PAYMENT", re.IGNORECASE),
            re.compile(r"SCHEDULE\s+OF\s+VALUES", re.IGNORECASE),
            re.compile(r"CONTINUATION\s+SHEET", re.IGNORECASE),
            re.compile(r"AIA\s+Document\s+G70[0-9]", re.IGNORECASE),  # AIA forms
            
            # Packing slip/shipping patterns
            re.compile(r"^\s*PACKING\s+SLIP", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^\s*SALES\s+ORDER", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^\s*SHIPPING", re.IGNORECASE | re.MULTILINE),
            re.compile(r"Ship\s+To:", re.IGNORECASE),
            re.compile(r"Sold\s+To:", re.IGNORECASE),
            
            # RFI patterns
            re.compile(r"REQUEST\s+FOR\s+INFORMATION", re.IGNORECASE),
            re.compile(r"^\s*RFI\s*#?\s*\d+", re.IGNORECASE | re.MULTILINE),
            re.compile(r"Request\s*#:\s*\d+", re.IGNORECASE),
            
            # Cost proposal patterns
            re.compile(r"^\s*COST\s+PROPOSAL", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^\s*PROPOSAL\s*#?\s*\d+", re.IGNORECASE | re.MULTILINE),
            re.compile(r"Quote\s+valid\s+for", re.IGNORECASE),
            
            # Change order patterns
            re.compile(r"CHANGE\s+ORDER\s*#?\s*\d+", re.IGNORECASE),
            re.compile(r"^C\.O\.\s*#?\s*\d+", re.IGNORECASE | re.MULTILINE),
            re.compile(r"PCO\s*#?\s*\d+", re.IGNORECASE),  # Proposed Change Order
            
            # Construction drawing patterns
            re.compile(r"^[A-Z]+\d+\.\d+\s*$", re.MULTILINE),  # Drawing numbers like P1.6, SI-1.1
            re.compile(r"SCALE:\s*\d+", re.IGNORECASE),
            re.compile(r"SHEET\s+\d+\s+OF\s+\d+", re.IGNORECASE),
            
            # Invoice patterns
            re.compile(r"^INVOICE\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"Invoice\s*#?\s*\d+", re.IGNORECASE),
            
            # Letter/memo patterns
            re.compile(r"^(MEMORANDUM|MEMO)\s*$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"^Dear\s+", re.IGNORECASE),
            
            # Construction-specific patterns
            re.compile(r"DAILY\s+REPORT", re.IGNORECASE),
            re.compile(r"FIELD\s+REPORT", re.IGNORECASE),
            re.compile(r"NOTICE\s+TO\s+PROCEED", re.IGNORECASE),
            re.compile(r"PUNCH\s+LIST", re.IGNORECASE),
            re.compile(r"CERTIFICATE\s+OF\s+SUBSTANTIAL\s+COMPLETION", re.IGNORECASE),
            re.compile(r"LIEN\s+WAIVER", re.IGNORECASE),
            re.compile(r"PAY\s+APPLICATION", re.IGNORECASE),
            
            # DSA (Division of State Architect) patterns for California schools
            re.compile(r"DSA\s+APPROVAL", re.IGNORECASE),
            re.compile(r"DSA\s+PROJECT", re.IGNORECASE),
            re.compile(r"DIVISION\s+OF\s+THE\s+STATE\s+ARCHITECT", re.IGNORECASE),
        ]
        
        # Page break indicators
        self.page_break_patterns = [
            re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
            re.compile(r"^\d+\s*$"),  # Just page numbers
            re.compile(r"-\s*\d+\s*-"),  # Centered page numbers
            re.compile(r"\d+/\d+"),  # Page numbers like 1/10
        ]
        
        # End of document indicators
        self.end_patterns = [
            re.compile(r"EXHIBIT\s+[A-Z]", re.IGNORECASE),  # Often starts attachments
            re.compile(r"ATTACHMENT\s+\d+", re.IGNORECASE),
            re.compile(r"APPENDIX", re.IGNORECASE),
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
        
        logger.info(f"Detecting boundaries in {pdf_doc.page_count} pages")
        
        # First pass: check if this is a fully scanned PDF
        scanned_pages = 0
        for i in range(min(5, pdf_doc.page_count)):  # Sample first 5 pages
            if len(pdf_doc[i].get_text().strip()) < 10:
                scanned_pages += 1
        
        is_scanned_pdf = scanned_pages >= min(3, pdf_doc.page_count)
        if is_scanned_pdf:
            logger.info("Detected scanned PDF - will use OCR for boundary detection")
        
        # Cache OCR results to avoid re-processing
        ocr_cache = {}
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text = page.get_text()
            
            # For scanned pages, try OCR if available
            if len(text.strip()) < 10 and self.ocr_handler:
                if page_num in ocr_cache:
                    text = ocr_cache[page_num]
                else:
                    try:
                        logger.debug(f"Performing OCR on page {page_num + 1} for boundary detection")
                        ocr_text, confidence = self.ocr_handler.process_page(page, dpi=200)  # Lower DPI for faster processing
                        if confidence > 0.3:  # Lower threshold for boundary detection
                            text = ocr_text
                            ocr_cache[page_num] = text
                            logger.debug(f"OCR text extracted with confidence {confidence:.2f}")
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            
            # Check if this page starts a new document
            try:
                if page_num > 0 and self._is_document_start(text, page, page_num):
                    # End previous document
                    boundaries.append((current_start, page_num - 1))
                    logger.info(f"Document boundary detected at page {page_num + 1}")
                    current_start = page_num
                # Also check for continuation patterns that might have been missed
                elif page_num > 0 and self._is_strong_boundary(text, page, pdf_doc, page_num, ocr_cache):
                    boundaries.append((current_start, page_num - 1))
                    logger.info(f"Strong boundary detected at page {page_num + 1}")
                    current_start = page_num
            except Exception as e:
                logger.error(f"Error checking document start for page {page_num + 1}: {e}")
                # Continue processing despite error
                pass
        
        # Add the last document
        if current_start < pdf_doc.page_count:
            boundaries.append((current_start, pdf_doc.page_count - 1))
        
        # Post-process to merge single-page fragments and validate boundaries
        boundaries = self._post_process_boundaries(boundaries, pdf_doc, ocr_cache)
        
        logger.info(f"Found {len(boundaries)} documents")
        return boundaries
    
    def _is_document_start(self, text: str, page: fitz.Page, page_num: int = 0) -> bool:
        """Check if a page represents the start of a new document.
        
        Args:
            text: Extracted text from the page
            page: PyMuPDF page object
            page_num: Page number for logging
            
        Returns:
            True if this appears to be a document start
        """
        # For scanned documents with OCR text, lower the threshold
        min_text_length = 20 if self.ocr_handler else 50
        
        # Skip mostly empty pages
        if len(text.strip()) < min_text_length:
            logger.debug(f"Page {page_num + 1}: Skipping - too little text ({len(text.strip())} chars)")
            return False
        
        # Check for very strong email indicators first
        strong_email_patterns = [
            re.compile(r"From:\s*.*@", re.IGNORECASE),
            re.compile(r"To:\s*.*@", re.IGNORECASE),
            re.compile(r"Sent:\s*\w+,\s*\w+\s*\d+", re.IGNORECASE),
            # More flexible email date patterns for OCR text
            re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\w+\s+\d{1,2},\s+\d{4}", re.IGNORECASE),
            re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)", re.IGNORECASE),
        ]
        
        for pattern in strong_email_patterns:
            if pattern.search(text[:800]):  # Check first 800 chars for email headers (OCR might be spread out)
                logger.info(f"Page {page_num + 1}: Strong email pattern detected")
                return True
        
        # Use construction-specific pattern detection
        doc_types = detect_document_type(text)
        if doc_types:
            logger.info(f"Page {page_num + 1}: Document types detected: {', '.join(doc_types)}")
            return True
        
        # Check for document start patterns
        for pattern in self.start_patterns:
            match = pattern.search(text[:1500])  # Check first 1500 chars (increased from 1000)
            if match:
                logger.info(f"Page {page_num + 1}: Document start pattern found: {pattern.pattern} -> '{match.group()}'")
                return True
        
        # Use construction-specific strong start detection
        if is_strong_document_start(text, page_num):
            logger.info(f"Page {page_num + 1}: Strong document start detected")
            return True
        
        # Check for page numbering that indicates a new document (Page 1 of X)
        page_one_pattern = re.compile(r"Page\s*1\s*of\s*\d+", re.IGNORECASE)
        if page_one_pattern.search(text):
            logger.info(f"Page {page_num + 1}: Found 'Page 1 of X' pattern")
            return True
        
        # Check for specific visual cues
        if self._has_letterhead(page):
            logger.info(f"Page {page_num + 1}: Letterhead detected")
            return True
        
        # Check if this looks like a form with specific structure
        if self._is_form_start(text):
            logger.info(f"Page {page_num + 1}: Form structure detected")
            return True
        
        # Check for significant blank space before this page (common in construction docs)
        # Disabled for now - Sample PDFs doesn't have blank pages
        # if page_num > 0 and self._previous_page_mostly_blank(page.parent, page_num - 1):
        #     logger.info(f"Page {page_num + 1}: Previous page was mostly blank - likely new document")
        #     return True
        
        # Check for significant formatting changes
        if self._has_format_change(page, page_num):
            logger.info(f"Page {page_num + 1}: Significant format change detected")
            return True
        
        return False
    
    def _has_letterhead(self, page: fitz.Page) -> bool:
        """Check if page has a letterhead (logo/header).
        
        Simple heuristic: check for images in top portion of page.
        """
        try:
            page_rect = page.rect
            top_region = fitz.Rect(
                page_rect.x0,
                page_rect.y0,
                page_rect.x1,
                page_rect.y0 + (page_rect.height * 0.2)  # Top 20%
            )
            
            # Check for images in top region
            images = page.get_images()
            if not images:
                return False
                
            for img in images:
                try:
                    # Some PDFs might have issues with get_image_bbox
                    img_list = page.get_image_bbox(img)
                    if img_list:
                        # get_image_bbox returns a list of rectangles
                        for img_rect in img_list:
                            if top_region.intersects(img_rect):
                                return True
                except Exception as e:
                    logger.debug(f"Could not get image bbox: {e}")
                    continue
            
            return False
        except Exception as e:
            logger.debug(f"Error checking letterhead: {e}")
            return False
    
    def _has_format_change(self, page: fitz.Page, page_num: int = 0) -> bool:
        """Check for significant formatting changes.
        
        Detect transitions between different document types based on layout.
        """
        try:
            # Get text blocks to analyze layout
            blocks = page.get_text("blocks")
            if not blocks:
                return False
                
            # Check for drawing/technical content
            drawings = page.get_drawings()
            if len(drawings) > 50:  # Many drawing elements suggest technical drawing
                if page_num > 0:
                    # Check if previous page had few drawings
                    prev_page = page.parent[page_num - 1]
                    prev_drawings = prev_page.get_drawings()
                    if len(prev_drawings) < 10:
                        logger.debug(f"Page {page_num + 1}: Transition to technical drawing")
                        return True
            
            # Check for form-like structure (many small text blocks in grid)
            if len(blocks) > 20:
                # Analyze block positions for grid-like arrangement
                y_positions = sorted([b[1] for b in blocks])  # y0 coordinates
                if len(set(y_positions)) > 15:  # Many different y positions
                    logger.debug(f"Page {page_num + 1}: Form-like structure detected")
                    return True
            
            return False
        except Exception as e:
            logger.debug(f"Error checking format change: {e}")
            return False
    
    def _is_form_start(self, text: str) -> bool:
        """Check if text indicates the start of a form.
        
        Forms often have specific field patterns.
        """
        form_indicators = [
            r"Date:\s*_{5,}",  # Date: ______
            r"Project\s*#:\s*",
            r"Transmitted\s+To:",
            r"Attention:",
            r"Project\s+Name:",
            r"Contract\s+No\.?:",
            r"P\.?O\.?\s+Box",
            r"Federal\s+Tax\s+ID",
        ]
        
        indicator_count = 0
        for indicator in form_indicators:
            if re.search(indicator, text[:1000], re.IGNORECASE):
                indicator_count += 1
        
        # If we find multiple form indicators, it's likely a form
        return indicator_count >= 2
    
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
    
    def _previous_page_mostly_blank(self, pdf_doc: fitz.Document, page_num: int) -> bool:
        """Check if the previous page is mostly blank.
        
        This is common in construction documents where documents are 
        separated by blank pages.
        """
        if page_num < 0 or page_num >= pdf_doc.page_count:
            return False
        
        page = pdf_doc[page_num]
        text = page.get_text().strip()
        
        # Check if page has very little text
        if len(text) < 100:
            # Also check if it's not just a page number
            if not re.search(r"^\s*\d+\s*$", text) and not re.search(r"Page\s+\d+", text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_strong_boundary(self, text: str, page: fitz.Page, pdf_doc: fitz.Document, page_num: int, ocr_cache: dict = None) -> bool:
        """Check for strong boundary indicators that might override normal detection.
        
        This catches transitions that pattern matching might miss.
        """
        # For scanned documents, need to check OCR cache
        prev_text = ""
        if page_num > 0:
            prev_page = pdf_doc[page_num - 1]
            prev_text = prev_page.get_text()
            
            # If previous page was scanned, get cached OCR text if available
            if len(prev_text.strip()) < 10 and ocr_cache:
                prev_text = ocr_cache.get(page_num - 1, prev_text)
        
        # Check for email timestamp patterns that indicate a new email thread
        email_timestamp = re.compile(
            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+(AM|PM)",
            re.IGNORECASE
        )
        if email_timestamp.search(text[:500]):
            # Check if previous page had different content type
            if page_num > 0 and not email_timestamp.search(prev_text[:500]):
                logger.debug(f"Page {page_num + 1}: Email timestamp detected - new email thread")
                return True
        
        # Check for major document type transitions
        # From email to form
        if page_num > 0:
            prev_is_email = any(marker in prev_text[:500] for marker in ["From:", "To:", "Subject:", "@"])
            curr_is_form = self._is_form_start(text)
            if prev_is_email and curr_is_form:
                logger.debug(f"Page {page_num + 1}: Transition from email to form")
                return True
        
        # Check for drastic change in text density (only for non-scanned pages)
        if page_num > 0 and len(prev_text.strip()) > 50:  # Only if we have real text
            curr_density = len(text.strip())
            prev_density = len(prev_text.strip())
            
            # If we go from very dense to very sparse or vice versa
            if prev_density > 1000 and curr_density < 200:
                logger.debug(f"Page {page_num + 1}: Text density change {prev_density} -> {curr_density}")
                return True
            elif prev_density < 200 and curr_density > 1000:
                logger.debug(f"Page {page_num + 1}: Text density change {prev_density} -> {curr_density}")
                return True
        
        return False
    
    def _post_process_boundaries(self, boundaries: List[Tuple[int, int]], 
                                 pdf_doc: fitz.Document, 
                                 ocr_cache: dict) -> List[Tuple[int, int]]:
        """Post-process boundaries to handle special cases and validate splits.
        
        Args:
            boundaries: Initial boundary list
            pdf_doc: PDF document
            ocr_cache: Cache of OCR results
            
        Returns:
            Refined boundary list
        """
        if len(boundaries) <= 1:
            return boundaries
        
        refined = []
        i = 0
        
        while i < len(boundaries):
            start, end = boundaries[i]
            
            # Check if this document is suspiciously short
            if end - start < 2:  # Less than 2 pages
                # Look at the content to decide if it should be merged
                should_merge = False
                
                # Get text for analysis
                doc_text = ""
                for p in range(start, end + 1):
                    if p in ocr_cache:
                        doc_text += ocr_cache[p]
                    else:
                        doc_text += pdf_doc[p].get_text()
                
                # Check if it's a continuation sheet or similar
                continuation_patterns = [
                    r"CONTINUATION\s+SHEET",
                    r"Page\s+\d+\s+of\s+\d+",
                    r"Continued from previous",
                    r"See attached",
                ]
                
                for pattern in continuation_patterns:
                    if re.search(pattern, doc_text, re.IGNORECASE):
                        should_merge = True
                        break
                
                # If we should merge and there's a next document
                if should_merge and i < len(boundaries) - 1:
                    next_start, next_end = boundaries[i + 1]
                    refined.append((start, next_end))
                    i += 2
                    continue
            
            # Check for multi-page emails that might have been split
            if i < len(boundaries) - 1:
                next_start, next_end = boundaries[i + 1]
                
                # Get first page of current and next document
                curr_first_text = ocr_cache.get(start, pdf_doc[start].get_text())
                next_first_text = ocr_cache.get(next_start, pdf_doc[next_start].get_text())
                
                # Check if both are emails from same thread
                if self._are_same_email_thread(curr_first_text, next_first_text):
                    refined.append((start, next_end))
                    i += 2
                    continue
            
            refined.append((start, end))
            i += 1
        
        return refined
    
    def _are_same_email_thread(self, text1: str, text2: str) -> bool:
        """Check if two pages are part of the same email thread."""
        # Extract subject from both texts
        subject_pattern = re.compile(r"Subject:\s*(.+?)(?:\n|$)", re.IGNORECASE)
        
        match1 = subject_pattern.search(text1[:1000])
        match2 = subject_pattern.search(text2[:1000])
        
        if match1 and match2:
            subject1 = match1.group(1).strip()
            subject2 = match2.group(1).strip()
            
            # Check if subjects are similar (accounting for RE:/FW: prefixes)
            subject1_clean = re.sub(r"^(RE:|FW:|Fwd:)\s*", "", subject1, flags=re.IGNORECASE)
            subject2_clean = re.sub(r"^(RE:|FW:|Fwd:)\s*", "", subject2, flags=re.IGNORECASE)
            
            if subject1_clean.lower() == subject2_clean.lower():
                return True
        
        return False