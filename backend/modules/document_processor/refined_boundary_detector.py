"""Refined document boundary detection with better multi-page handling and email chain detection."""

import re
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

import fitz  # PyMuPDF
from loguru import logger

from .improved_ocr_handler import ImprovedOCRHandler
from .construction_patterns import detect_document_type


class RefinedBoundaryDetector:
    """Refined boundary detection with improved accuracy for real-world documents."""
    
    def __init__(self, ocr_handler: Optional[ImprovedOCRHandler] = None):
        self.ocr_handler = ocr_handler
        
        # Strong document start patterns - these indicate definite new documents
        self.strong_start_patterns = [
            # Email patterns that indicate NEW emails (not quoted)
            (r"^From:\s*[^@\n]+@[^>\n]+$", "new_email"),  # Must be at line start, no > prefix
            (r"^Sent:\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", "new_email"),
            (r"^Subject:\s*(?!RE:|FW:|Fwd:)", "new_email"),  # New subject, not reply/forward
            
            # Document headers at start of page
            (r"^\s*SUBMITTAL\s*TRANSMITTAL", "submittal"),
            (r"^\s*APPLICATION\s+AND\s+CERTIFICATE\s+FOR\s+PAYMENT", "payment"),
            (r"^\s*SCHEDULE\s+OF\s+VALUES", "sov"),
            (r"^\s*REQUEST\s+FOR\s+INFORMATION", "rfi"),
            (r"^\s*RFI\s*#?\s*\d+", "rfi"),
            (r"^\s*COST\s+PROPOSAL\s*#?\s*\d+", "cost_proposal"),
            (r"^\s*CHANGE\s+ORDER\s*#?\s*\d+", "change_order"),
            (r"^\s*PACKING\s+SLIP", "shipping"),  # Strong indicator at start
            
            # Strong page 1 indicators
            (r"Page\s+1\s+of\s+\d+", "page_one"),
            (r"Sheet\s+1\s+of\s+\d+", "page_one"),
        ]
        
        # Continuation patterns - these indicate the same document continues
        self.continuation_patterns = [
            r"Page\s+[2-9]\d*\s+of\s+\d+",  # Page 2+
            r"^\s*-\s*\d+\s*-\s*$",  # Page numbers like - 2 -
            r"CONTINUATION\s+SHEET",
            r"Continued from",
            r"See attached",
            r"^\s*\d+\s*$",  # Just a page number
        ]
        
        # Quoted email patterns (indicate this is part of an email chain)
        self.quoted_email_patterns = [
            r"^>\s*From:",  # Quoted email
            r"^>\s*To:",
            r"^>\s*Subject:",
            r"-----Original Message-----",
            r"On .+ wrote:",  # Common email quote format
            r"From:.*Sent:.*To:.*Subject:",  # All on one line (common in forwards)
        ]
        
    def detect_boundaries(self, pdf_doc: fitz.Document) -> List[Tuple[int, int]]:
        """Detect document boundaries with improved multi-page handling."""
        logger.info(f"Detecting boundaries in {pdf_doc.page_count} pages")
        
        # First, analyze all pages to understand document structure
        page_analysis = self._analyze_pages(pdf_doc)
        
        # Then detect boundaries based on analysis
        boundaries = self._detect_boundaries_from_analysis(page_analysis)
        
        # Post-process to handle edge cases
        boundaries = self._post_process_boundaries(boundaries, page_analysis)
        
        logger.info(f"Found {len(boundaries)} documents")
        for i, (start, end) in enumerate(boundaries):
            logger.info(f"  Document {i+1}: pages {start+1}-{end+1}")
        
        return boundaries
    
    def _analyze_pages(self, pdf_doc: fitz.Document) -> List[Dict]:
        """Analyze each page to determine its characteristics."""
        page_analysis = []
        ocr_cache = {}
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text = page.get_text()
            
            # Use OCR for scanned pages
            if len(text.strip()) < 10 and self.ocr_handler:
                if page_num not in ocr_cache:
                    try:
                        logger.debug(f"Performing OCR on page {page_num + 1}")
                        ocr_text, confidence = self.ocr_handler.process_page(page, dpi=200)
                        if confidence > 0.2:
                            text = ocr_text
                            ocr_cache[page_num] = text
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            
            # Analyze page characteristics
            analysis = {
                'page_num': page_num,
                'text': text,
                'text_length': len(text.strip()),
                'doc_types': detect_document_type(text),
                'has_strong_start': False,
                'strong_start_type': None,
                'has_continuation': False,
                'has_quoted_email': False,
                'has_page_number': False,
                'page_number_info': None,
                'is_mostly_blank': len(text.strip()) < 50,
            }
            
            # Check for strong start patterns
            for pattern, doc_type in self.strong_start_patterns:
                if re.search(pattern, text[:1000], re.IGNORECASE | re.MULTILINE):
                    analysis['has_strong_start'] = True
                    analysis['strong_start_type'] = doc_type
                    break
            
            # Check for continuation patterns
            for pattern in self.continuation_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    analysis['has_continuation'] = True
                    break
            
            # Check for quoted emails
            for pattern in self.quoted_email_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    analysis['has_quoted_email'] = True
                    break
            
            # Extract page numbering info
            page_pattern = re.search(r"Page\s+(\d+)\s+of\s+(\d+)", text, re.IGNORECASE)
            if page_pattern:
                analysis['has_page_number'] = True
                analysis['page_number_info'] = {
                    'current': int(page_pattern.group(1)),
                    'total': int(page_pattern.group(2))
                }
            
            page_analysis.append(analysis)
        
        return page_analysis
    
    def _detect_boundaries_from_analysis(self, page_analysis: List[Dict]) -> List[Tuple[int, int]]:
        """Detect boundaries based on page analysis."""
        boundaries = []
        current_start = 0
        current_doc_types = set()
        in_email_chain = False
        email_chain_pages = 0  # Track how many pages in current email chain
        
        for i, analysis in enumerate(page_analysis):
            # Skip mostly blank pages
            if analysis['is_mostly_blank']:
                continue
            
            # Determine if this is a new document
            is_new_document = False
            reason = ""
            
            # Strong start pattern handling
            if analysis['has_strong_start']:
                # Special handling for emails
                if analysis['strong_start_type'] == 'new_email':
                    # Check if this is a genuinely new email thread
                    if i > 0 and self._is_new_email_thread(analysis['text'], i, page_analysis):
                        is_new_document = True
                        reason = "New email thread detected"
                        in_email_chain = True
                        email_chain_pages = 1
                    elif not in_email_chain:
                        # First email
                        is_new_document = True
                        reason = "First email detected"
                        in_email_chain = True
                        email_chain_pages = 1
                    else:
                        # Part of existing chain
                        is_new_document = False
                        reason = "Email chain continuation"
                        email_chain_pages += 1
                else:
                    # Non-email strong start always indicates new document
                    is_new_document = True
                    reason = f"Strong start: {analysis['strong_start_type']}"
                    in_email_chain = False
                    email_chain_pages = 0
            
            # Check for document type transitions
            elif i > 0:
                prev_analysis = page_analysis[i-1]
                
                # If previous page had continuation indicator, this is same document
                if prev_analysis.get('has_continuation'):
                    is_new_document = False
                    reason = "Previous page has continuation"
                
                # Check page numbering
                elif analysis['has_page_number'] and analysis['page_number_info']['current'] == 1:
                    is_new_document = True
                    reason = "Page 1 of new document"
                    in_email_chain = False
                    email_chain_pages = 0
                
                # Check for document type changes
                elif analysis['doc_types'] and prev_analysis['doc_types']:
                    # Get primary doc types (first one detected)
                    curr_primary = analysis['doc_types'][0] if analysis['doc_types'] else None
                    prev_primary = prev_analysis['doc_types'][0] if prev_analysis['doc_types'] else None
                    
                    # Special cases for multi-page documents
                    if curr_primary == prev_primary:
                        # Same document type - check if it's a new instance
                        if curr_primary == 'SHIPPING':
                            # Check if it's a different vendor/order
                            if self._is_different_shipping_doc(analysis['text'], prev_analysis['text']):
                                is_new_document = True
                                reason = "New shipping document (different vendor/order)"
                            else:
                                is_new_document = False
                                reason = "Multi-page shipping document"
                        elif curr_primary in ['PAYMENT', 'DRAWING']:
                            # These often span multiple pages
                            is_new_document = False
                            reason = f"Multi-page {curr_primary}"
                        elif curr_primary == 'EMAIL':
                            # Check if it's a genuinely new email
                            if self._is_new_email_thread(analysis['text'], i, page_analysis):
                                is_new_document = True
                                reason = "New email thread"
                                in_email_chain = True
                                email_chain_pages = 1
                            else:
                                is_new_document = False
                                reason = "Email chain continuation"
                                email_chain_pages += 1
                        else:
                            # Check for specific patterns indicating new document
                            if self._is_new_instance_of_type(analysis['text'], curr_primary):
                                is_new_document = True
                                reason = f"New {curr_primary} instance"
                            else:
                                is_new_document = False
                                reason = f"Continuation of {curr_primary}"
                    else:
                        # Different document types - strong indicator of boundary
                        is_new_document = True
                        reason = f"Type change: {prev_primary} -> {curr_primary}"
                        in_email_chain = curr_primary == 'EMAIL'
                        email_chain_pages = 1 if curr_primary == 'EMAIL' else 0
                
                # Check if previous was mostly blank (common separator)
                elif i > 1 and page_analysis[i-1]['is_mostly_blank']:
                    is_new_document = True
                    reason = "After blank page separator"
                    in_email_chain = False
                    email_chain_pages = 0
                
                # No clear indicators - check text patterns
                else:
                    # Look for patterns that indicate new document without strong start
                    if self._has_weak_start_indicators(analysis['text']):
                        is_new_document = True
                        reason = "Weak start indicators"
                        in_email_chain = False
                        email_chain_pages = 0
            
            # First page is always start of first document
            elif i == 0:
                current_doc_types = set(analysis['doc_types'])
                if 'EMAIL' in current_doc_types:
                    in_email_chain = True
                    email_chain_pages = 1
                continue
            
            # Process boundary decision
            if is_new_document and i > 0:
                # End previous document
                boundaries.append((current_start, i - 1))
                logger.info(f"Boundary at page {i + 1}: {reason}")
                current_start = i
                current_doc_types = set(analysis['doc_types'])
            else:
                # Continue current document
                current_doc_types.update(analysis['doc_types'])
                if analysis['doc_types'] and not in_email_chain:
                    # Update email chain status if we detect email patterns
                    if 'EMAIL' in analysis['doc_types']:
                        in_email_chain = True
                        email_chain_pages = 1
                elif in_email_chain:
                    email_chain_pages += 1
        
        # Add final document
        if current_start < len(page_analysis):
            boundaries.append((current_start, len(page_analysis) - 1))
        
        return boundaries
    
    def _is_new_instance_of_type(self, text: str, doc_type: str) -> bool:
        """Check if this is a new instance of the same document type."""
        # Document-specific checks
        if doc_type == 'SHIPPING':
            # Look for PACKING SLIP header or different order numbers
            if re.search(r"^\s*PACKING\s+SLIP", text[:300], re.IGNORECASE | re.MULTILINE):
                return True
            order_pattern = re.compile(r"Order\s*#?\s*(\d+)", re.IGNORECASE)
            matches = order_pattern.findall(text[:500])
            if matches:
                return True
        
        elif doc_type == 'EMAIL':
            # Check for new email indicators (not quoted)
            if re.search(r"^From:\s*[^>]", text[:500], re.IGNORECASE | re.MULTILINE):
                return True
        
        elif doc_type == 'COST_PROPOSAL':
            # Look for COST PROPOSAL header or proposal numbers
            if re.search(r"COST\s+PROPOSAL", text[:300], re.IGNORECASE):
                return True
            if re.search(r"Proposal\s*#?\s*\d+", text[:300], re.IGNORECASE):
                return True
        
        elif doc_type == 'RFI':
            # Look for RFI headers
            if re.search(r"REQUEST\s+FOR\s+INFORMATION", text[:300], re.IGNORECASE):
                return True
            if re.search(r"RFI\s*#?\s*\d+", text[:300], re.IGNORECASE):
                return True
        
        return False
    
    def _has_weak_start_indicators(self, text: str) -> bool:
        """Check for weak indicators that might suggest document start."""
        weak_indicators = [
            r"^\s*[A-Z\s]{15,}$",  # All caps title line
            r"^(INVOICE|MEMORANDUM|NOTICE|CERTIFICATE)",  # Document type at start
            r"^Dear\s+",  # Letter greeting
            r"^\d{1,2}/\d{1,2}/\d{4}$",  # Date at start of line
        ]
        
        for pattern in weak_indicators:
            if re.search(pattern, text[:200], re.IGNORECASE | re.MULTILINE):
                return True
        
        return False
    
    def _is_new_email_thread(self, text: str, page_idx: int, page_analysis: List[Dict]) -> bool:
        """Check if this email represents a new thread rather than continuation."""
        # Extract current email metadata
        curr_from = self._extract_email_from(text)
        curr_subject = self._extract_email_subject(text)
        curr_date = self._extract_email_date(text)
        
        # Look back at previous emails (up to 4 pages back)
        for j in range(page_idx - 1, max(0, page_idx - 4), -1):
            if 'EMAIL' in page_analysis[j]['doc_types']:
                prev_text = page_analysis[j]['text']
                prev_from = self._extract_email_from(prev_text)
                prev_subject = self._extract_email_subject(prev_text)
                prev_date = self._extract_email_date(prev_text)
                
                # Check if subjects are different (ignoring RE:/FW:)
                if curr_subject and prev_subject:
                    curr_clean = re.sub(r"^(RE:|FW:|Fwd:)\s*", "", curr_subject, flags=re.IGNORECASE).strip()
                    prev_clean = re.sub(r"^(RE:|FW:|Fwd:)\s*", "", prev_subject, flags=re.IGNORECASE).strip()
                    
                    # Different subjects = different threads
                    if curr_clean.lower() != prev_clean.lower():
                        return True
                    
                    # Same subject - check if this is a new conversation
                    # Look for significant time gap or different participants
                    if curr_date != prev_date:
                        # Different dates might indicate new conversation
                        return True
                
                # If we found a related email, assume continuation
                return False
        
        # If no previous email found, this is new
        return True
    
    def _is_different_shipping_doc(self, curr_text: str, prev_text: str) -> bool:
        """Check if two shipping documents are from different vendors/orders."""
        # Extract vendor indicators
        vendors = ['MicroMetl', 'Geary Pacific', 'UPS', 'FedEx']
        
        curr_vendor = None
        prev_vendor = None
        
        for vendor in vendors:
            if vendor.lower() in curr_text[:500].lower():
                curr_vendor = vendor
            if vendor.lower() in prev_text[:500].lower():
                prev_vendor = vendor
        
        # Different vendors = different documents
        if curr_vendor and prev_vendor and curr_vendor != prev_vendor:
            return True
        
        # Check for order numbers
        order_pattern = re.compile(r"Order\s*#?\s*(\d+)", re.IGNORECASE)
        curr_orders = order_pattern.findall(curr_text[:500])
        prev_orders = order_pattern.findall(prev_text[:500])
        
        if curr_orders and prev_orders and curr_orders[0] != prev_orders[0]:
            return True
        
        # Check for explicit "PACKING SLIP" header at start
        if re.search(r"^\s*PACKING\s+SLIP", curr_text[:200], re.IGNORECASE | re.MULTILINE):
            return True
        
        return False
    
    def _post_process_boundaries(self, boundaries: List[Tuple[int, int]], 
                                page_analysis: List[Dict]) -> List[Tuple[int, int]]:
        """Post-process boundaries to handle edge cases."""
        if len(boundaries) <= 1:
            return boundaries
        
        refined = []
        i = 0
        
        while i < len(boundaries):
            start, end = boundaries[i]
            
            # Check for very short documents that might be fragments
            if end - start == 0:  # Single page document
                # Check if it's a continuation sheet or similar
                page_text = page_analysis[start]['text']
                if re.search(r"CONTINUATION\s+SHEET", page_text, re.IGNORECASE):
                    # Try to merge with previous document
                    if refined:
                        prev_start, prev_end = refined[-1]
                        refined[-1] = (prev_start, end)
                        logger.info(f"Merged continuation sheet at page {start + 1}")
                        i += 1
                        continue
            
            # Check for specific cases where merging might be needed
            if i < len(boundaries) - 1:
                next_start, next_end = boundaries[i + 1]
                
                # Check if current ends with "See attached" or similar
                end_text = page_analysis[end]['text']
                if re.search(r"(See attached|Attached please find|Attachment)", end_text[-500:], re.IGNORECASE):
                    # Next document might be the attachment
                    refined.append((start, next_end))
                    logger.info(f"Merged attachment: pages {start+1}-{next_end+1}")
                    i += 2
                    continue
            
            refined.append((start, end))
            i += 1
        
        return refined
    
    def _extract_email_from(self, text: str) -> Optional[str]:
        """Extract From: field from email."""
        from_pattern = re.compile(r"From:\s*([^<\n]+)", re.IGNORECASE)
        match = from_pattern.search(text[:500])
        return match.group(1).strip() if match else None
    
    def _extract_email_date(self, text: str) -> Optional[str]:
        """Extract date from email."""
        # Look for sent date
        sent_pattern = re.compile(r"Sent:\s*([^\n]+)", re.IGNORECASE)
        match = sent_pattern.search(text[:500])
        if match:
            return match.group(1).strip()
        
        # Look for date field
        date_pattern = re.compile(r"Date:\s*([^\n]+)", re.IGNORECASE)
        match = date_pattern.search(text[:500])
        return match.group(1).strip() if match else None
    
    def _extract_email_subject(self, text: str) -> Optional[str]:
        """Extract email subject from text."""
        subject_pattern = re.compile(r"Subject:\s*(.+?)(?:\n|$)", re.IGNORECASE)
        match = subject_pattern.search(text[:1000])
        return match.group(1).strip() if match else None
    
    def _extract_email_dates(self, text: str) -> List[str]:
        """Extract email dates from text."""
        date_pattern = re.compile(
            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},\s+\d{4}",
            re.IGNORECASE
        )
        return date_pattern.findall(text[:1000])