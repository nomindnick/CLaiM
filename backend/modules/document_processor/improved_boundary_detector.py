"""Improved boundary detection with better handling of multi-page documents.

This module improves on the base boundary detector by reducing false positives
in email chains, shipping documents, and technical drawings.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import fitz
from loguru import logger

from .boundary_detector import BoundaryDetector
from .models import DocumentType
from .ocr_handler import OCRHandler


class ImprovedBoundaryDetector(BoundaryDetector):
    """Improved boundary detector that reduces over-detection."""
    
    def __init__(self, ocr_handler: Optional[OCRHandler] = None):
        """Initialize improved boundary detector."""
        super().__init__(ocr_handler)
        
        # Track context for better multi-page detection
        self.email_chain_pages = set()
        self.current_email_subject = None
        self.shipping_doc_pages = set()
        self.current_vendor = None
        self.drawing_pages = set()
        
    def _is_document_start(self, text: str, page: fitz.Page, page_num: int = 0) -> bool:
        """Enhanced document start detection with context awareness.
        
        Args:
            text: Current page text
            page: PyMuPDF page object
            page_num: Current page number
            
        Returns:
            bool: Whether this page starts a new document
        """
        # Get previous page text if available
        prev_text = ""
        if hasattr(self, '_page_texts') and page_num > 0:
            prev_text = self._page_texts.get(page_num - 1, "")
        
        # Store current page text for next iteration
        if not hasattr(self, '_page_texts'):
            self._page_texts = {}
        self._page_texts[page_num] = text
        
        # Check if this is a continuation of an email chain
        if self._is_email_chain_continuation(text, prev_text):
            return False
        
        # Check if this is a continuation of a shipping document
        if self._is_shipping_doc_continuation(text, prev_text):
            return False
        
        # Check if this is a continuation of technical drawings
        if self._is_drawing_continuation(text, prev_text):
            return False
        
        # Now check for new document starts
        # Strong email patterns (but not if it's quoted text)
        if self._is_new_email(text, prev_text):
            return True
        
        # RFI/submittal patterns - check for new vs continuation
        rfi_match = re.search(r'REQUEST\s+FOR\s+INFORMATION.*RFI[:\s#-]*(\d+)', text[:1000], re.IGNORECASE)
        if rfi_match and not self._is_rfi_continuation(text, prev_text):
            return True
        
        # Submittal patterns - check if it's a new submittal
        submittal_match = re.search(r'SUBMITTAL\s*(TRANSMITTAL|#|NUMBER)?[:\s]*(\d+)', text[:500], re.IGNORECASE)
        if submittal_match and not self._is_submittal_continuation(text, prev_text):
            return True
        
        # Check if this is a continuation of a payment application
        if self._is_payment_application_continuation(text, prev_text):
            return False
        
        # Invoice/Payment patterns - but check if it's a new one
        if self._is_new_invoice_or_payment(text, prev_text):
            return True
        
        # Shipping documents - check vendor change
        if self._is_new_shipping_document(text, prev_text):
            return True
        
        # Cost proposals
        cost_match = re.search(r'COST\s+PROPOSAL\s*#?\s*(\d+)', text[:500], re.IGNORECASE)
        if cost_match:
            return True
        
        # Plans and specifications - only if clearly starting a new section
        if self._is_new_drawing_section(text, prev_text):
            return True
        
        # Schedule of values - only if not a continuation
        if re.search(r'SCHEDULE\s+OF\s+VALUES', text[:500], re.IGNORECASE):
            # Check if it's part of a payment application
            if not self._is_payment_application_continuation(text, prev_text):
                return True
        
        # Fall back to parent implementation for other patterns
        parent_result = super()._is_document_start(text, page, page_num)
        
        # But filter out weak boundaries
        if parent_result:
            # Check if it's just a page number or header/footer
            if self._is_likely_header_footer(text):
                return False
        
        return parent_result
    
    def _is_email_chain_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of an email chain."""
        # Look for quoted email patterns (>, >>, etc.)
        quoted_lines = len(re.findall(r'^>+\s*', text, re.MULTILINE))
        total_lines = len(text.split('\n'))
        
        if quoted_lines > total_lines * 0.3:  # More than 30% quoted
            return True
        
        # Check if previous page had email and this has similar subject
        if prev_text and re.search(r'Subject:\s*(.+)', prev_text, re.IGNORECASE):
            prev_subject = re.search(r'Subject:\s*(.+)', prev_text, re.IGNORECASE).group(1).strip()
            curr_subject_match = re.search(r'Subject:\s*(.+)', text, re.IGNORECASE)
            if curr_subject_match:
                curr_subject = curr_subject_match.group(1).strip()
                # Remove RE:/FW: prefixes for comparison
                prev_clean = re.sub(r'^(RE:|FW:|FWD:)\s*', '', prev_subject, flags=re.IGNORECASE).strip()
                curr_clean = re.sub(r'^(RE:|FW:|FWD:)\s*', '', curr_subject, flags=re.IGNORECASE).strip()
                if prev_clean == curr_clean:
                    return True
        
        return False
    
    def _is_new_email(self, text: str, prev_text: str) -> bool:
        """Check if this is a new email (not a continuation or quoted reply)."""
        # Must have From/To/Subject pattern at the beginning
        has_email_header = bool(re.search(
            r'^From:\s*[^\n]+\s+To:\s*[^\n]+\s+(Subject:|Sent:)', 
            text[:1000], 
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        ))
        
        if not has_email_header:
            return False
        
        # Check if it's quoted (indented or has > markers)
        first_from_pos = text.lower().find('from:')
        if first_from_pos > 50:  # If From: is not near the beginning, it's likely quoted
            return False
        
        # Check for signature blocks that might look like emails
        if prev_text and len(text) < 300:  # Short page might be just a signature
            # Check if previous page had email content
            if re.search(r'(From:|To:|Subject:)', prev_text, re.IGNORECASE):
                # This might just be a continuation/signature
                return False
        
        # Check if subject is different from previous
        if prev_text:
            prev_subject_match = re.search(r'Subject:\s*(.+)', prev_text, re.IGNORECASE)
            curr_subject_match = re.search(r'Subject:\s*(.+)', text, re.IGNORECASE)
            
            # If current has no subject but previous did, it's likely a continuation
            if prev_subject_match and not curr_subject_match:
                return False
            
            if prev_subject_match and curr_subject_match:
                prev_subject = prev_subject_match.group(1).strip()
                curr_subject = curr_subject_match.group(1).strip()
                # Remove RE:/FW: prefixes
                prev_clean = re.sub(r'^(RE:|FW:|FWD:)\s*', '', prev_subject, flags=re.IGNORECASE).strip()
                curr_clean = re.sub(r'^(RE:|FW:|FWD:)\s*', '', curr_subject, flags=re.IGNORECASE).strip()
                if prev_clean == curr_clean:
                    return False
        
        # Additional check: if this looks like a forwarded/quoted email within the same chain
        if re.search(r'(Original Message|Forwarded by|wrote:|On\s+\d+/\d+/\d+)', text[:500], re.IGNORECASE):
            return False
        
        return True
    
    def _is_shipping_doc_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of a shipping document."""
        # Check if previous page was a packing slip
        if not prev_text or 'PACKING SLIP' not in prev_text.upper():
            return False
        
        # Check if this page also has packing slip header
        if 'PACKING SLIP' in text[:200].upper():
            # Check if it's the same vendor
            prev_vendor = self._extract_vendor(prev_text)
            curr_vendor = self._extract_vendor(text)
            if prev_vendor and curr_vendor and prev_vendor == curr_vendor:
                # Check if it's the same order
                prev_order = self._extract_order_number(prev_text)
                curr_order = self._extract_order_number(text)
                if prev_order and curr_order and prev_order == curr_order:
                    return True
        
        return False
    
    def _is_new_shipping_document(self, text: str, prev_text: str) -> bool:
        """Check if this is a new shipping document."""
        if 'PACKING SLIP' not in text[:200].upper():
            return False
        
        # If no previous text, it's new
        if not prev_text:
            return True
        
        # Check vendor change
        prev_vendor = self._extract_vendor(prev_text)
        curr_vendor = self._extract_vendor(text)
        if prev_vendor and curr_vendor and prev_vendor != curr_vendor:
            return True
        
        # Check order number change
        prev_order = self._extract_order_number(prev_text)
        curr_order = self._extract_order_number(text)
        if prev_order and curr_order and prev_order != curr_order:
            return True
        
        # If previous wasn't a shipping doc, this is new
        if 'PACKING SLIP' not in prev_text.upper():
            return True
        
        return False
    
    def _extract_vendor(self, text: str) -> Optional[str]:
        """Extract vendor name from shipping document."""
        # Look for common vendors in the construction industry
        if 'MICROMETL' in text.upper():
            return 'MicroMetl'
        elif 'GEARY PACIFIC' in text.upper():
            return 'Geary Pacific'
        
        # Try to extract from header
        vendor_match = re.search(r'(From|Vendor|Supplier):\s*([^\n]+)', text[:500], re.IGNORECASE)
        if vendor_match:
            return vendor_match.group(2).strip()
        
        return None
    
    def _extract_order_number(self, text: str) -> Optional[str]:
        """Extract order number from document."""
        # Look for order/invoice number patterns
        patterns = [
            r'Order\s*#?\s*:?\s*(\d+)',
            r'Invoice\s*#?\s*:?\s*(\d+)',
            r'P\.?O\.?\s*#?\s*:?\s*(\d+)',
            r'Sales\s*Order\s*#?\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _is_drawing_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of technical drawings."""
        # Check if both pages have drawing identifiers
        if not prev_text:
            return False
        
        prev_has_drawing = bool(re.search(r'(SHEET|DWG|DRAWING)\s*\d+', prev_text[:500], re.IGNORECASE))
        curr_has_drawing = bool(re.search(r'(SHEET|DWG|DRAWING)\s*\d+', text[:500], re.IGNORECASE))
        
        if prev_has_drawing and curr_has_drawing:
            # Check if they're sequential sheets
            prev_sheet = re.search(r'SHEET\s*(\d+)', prev_text[:500], re.IGNORECASE)
            curr_sheet = re.search(r'SHEET\s*(\d+)', text[:500], re.IGNORECASE)
            if prev_sheet and curr_sheet:
                prev_num = int(prev_sheet.group(1))
                curr_num = int(curr_sheet.group(1))
                if curr_num == prev_num + 1:
                    return True
        
        return False
    
    def _is_new_drawing_section(self, text: str, prev_text: str) -> bool:
        """Check if this starts a new drawing/specification section."""
        # Must have clear drawing indicators
        has_drawing = bool(re.search(
            r'(STRUCTURAL\s+ENGINEERING|HVAC\s+DUCT|SEISMIC\s+BRACING|SHEET\s+\d+)', 
            text[:500], 
            re.IGNORECASE
        ))
        
        if not has_drawing:
            return False
        
        # If previous page wasn't a drawing, this is new
        if prev_text and not re.search(r'(SHEET|DWG|DRAWING)\s*\d+', prev_text[:500], re.IGNORECASE):
            return True
        
        # If sheet numbers are not sequential, it's a new section
        if prev_text:
            prev_sheet = re.search(r'SHEET\s*(\d+)', prev_text[:500], re.IGNORECASE)
            curr_sheet = re.search(r'SHEET\s*(\d+)', text[:500], re.IGNORECASE)
            if prev_sheet and curr_sheet:
                prev_num = int(prev_sheet.group(1))
                curr_num = int(curr_sheet.group(1))
                if curr_num != prev_num + 1:
                    return True
        
        return False
    
    def _is_rfi_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of an RFI."""
        if not prev_text:
            return False
        
        # Extract RFI numbers
        prev_rfi = re.search(r'RFI[:\s#-]*(\d+)', prev_text[:500], re.IGNORECASE)
        curr_rfi = re.search(r'RFI[:\s#-]*(\d+)', text[:500], re.IGNORECASE)
        
        if prev_rfi and curr_rfi and prev_rfi.group(1) == curr_rfi.group(1):
            return True
        
        return False
    
    def _is_new_invoice_or_payment(self, text: str, prev_text: str) -> bool:
        """Check if this is a new invoice or payment application."""
        # Look for invoice/payment patterns
        has_invoice = bool(re.search(
            r'(INVOICE|APPLICATION.*PAYMENT|PAYMENT\s+APPLICATION)', 
            text[:500], 
            re.IGNORECASE
        ))
        
        if not has_invoice:
            return False
        
        # Check if it has a different invoice number
        if prev_text:
            prev_inv = re.search(r'Invoice\s*#?\s*:?\s*(\d+)', prev_text[:500], re.IGNORECASE)
            curr_inv = re.search(r'Invoice\s*#?\s*:?\s*(\d+)', text[:500], re.IGNORECASE)
            if prev_inv and curr_inv and prev_inv.group(1) != curr_inv.group(1):
                return True
        
        # Check if previous wasn't an invoice
        if prev_text and not re.search(
            r'(INVOICE|APPLICATION.*PAYMENT|PAYMENT\s+APPLICATION)', 
            prev_text[:500], 
            re.IGNORECASE
        ):
            return True
        
        return False
    
    def _is_likely_header_footer(self, text: str) -> bool:
        """Check if the text is likely just a header/footer."""
        # Very short text
        if len(text.strip()) < 100:
            # Just page numbers, dates, or project names
            if re.match(r'^(Page\s+\d+|.*Project.*|\d{1,2}/\d{1,2}/\d{4})$', text.strip(), re.IGNORECASE):
                return True
        
        return False
    
    def _is_submittal_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of a submittal."""
        if not prev_text:
            return False
        
        # Check if previous page had submittal header
        prev_has_submittal = bool(re.search(r'SUBMITTAL', prev_text[:500], re.IGNORECASE))
        curr_has_submittal = bool(re.search(r'SUBMITTAL', text[:500], re.IGNORECASE))
        
        # If previous was submittal and current mentions shop drawing/material submittal
        if prev_has_submittal and re.search(r'SHOP\s+DRAWING.*SUBMITTAL', text[:500], re.IGNORECASE):
            return True
        
        # Check submittal numbers
        if prev_has_submittal and curr_has_submittal:
            prev_num = re.search(r'SUBMITTAL.*?(\d+)', prev_text[:500], re.IGNORECASE)
            curr_num = re.search(r'SUBMITTAL.*?(\d+)', text[:500], re.IGNORECASE)
            if prev_num and curr_num and prev_num.group(1) == curr_num.group(1):
                return True
        
        return False
    
    def _is_payment_application_continuation(self, text: str, prev_text: str) -> bool:
        """Check if this is a continuation of a payment application."""
        if not prev_text:
            return False
        
        # Check for AIA document continuations
        if re.search(r'Document\s+G703.*Continuation\s+Sheet', text[:500], re.IGNORECASE):
            return True
        
        # Check if previous page had payment application
        prev_has_payment = bool(re.search(r'APPLICATION.*PAYMENT|SCHEDULE\s+OF\s+VALUES', prev_text[:500], re.IGNORECASE))
        curr_has_schedule = bool(re.search(r'SCHEDULE\s+OF\s+VALUES', text[:500], re.IGNORECASE))
        
        if prev_has_payment and curr_has_schedule:
            return True
        
        # Check for continuation patterns
        if prev_has_payment and re.search(r'(Application\s+No|Period\s+To|ITEM\s+SCHEDULED)', text[:500], re.IGNORECASE):
            return True
        
        return False