"""Document type specific strategies for boundary detection."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import re

from loguru import logger

from ..llm_client.base_client import LLMClient


class DocumentTypeStrategy(ABC):
    """Base class for document type specific strategies."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
    
    @abstractmethod
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues from previous page.
        
        Returns:
            Tuple of (continues, confidence, reasoning)
        """
        pass
    
    @abstractmethod
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new document of this type.
        
        Returns:
            Tuple of (is_start, confidence)
        """
        pass
    
    @abstractmethod
    def get_prompt_hints(self) -> str:
        """Get specific hints for LLM prompt for this document type."""
        pass


class EmailStrategy(DocumentTypeStrategy):
    """Strategy for email document detection."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues an email chain."""
        
        if not self.llm_client:
            # Simple heuristic fallback
            current_text = current_page['text'][:500].lower()
            has_quote = any(marker in current_text for marker in ['>', '>>', 'from:', 'sent:', '-----'])
            return has_quote, 0.5, "Heuristic: quote markers detected"
        
        # LLM analysis for email continuity
        prompt = f"""Analyze if these two pages are part of the same email chain:
        
Previous page ending:
{previous_page['text'][-500:]}

Current page beginning:
{current_page['text'][:500]}

Consider:
1. Is this a quoted reply (>, >>, etc.)?
2. Does the subject line match?
3. Is this a forwarded section?
4. Are the participants the same?
5. Look for email thread indicators like "On [date], [person] wrote:"

Return JSON: {{"continues": true/false, "confidence": 0.0-1.0, "reason": "..."}}"""
        
        try:
            response = self.llm_client.complete(prompt)
            # Parse response
            import json
            result = json.loads(response.strip())
            return (
                result.get('continues', False),
                result.get('confidence', 0.5),
                result.get('reason', 'LLM analysis')
            )
        except Exception as e:
            logger.warning(f"Email continuity check failed: {e}")
            return False, 0.5, "Analysis failed"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new email."""
        text = page_info['text'][:1000]
        
        # Look for email headers
        email_pattern = r'From:\s*[^\n]+@[^\n]+\nTo:\s*[^\n]+\n(?:Subject:|Date:|Sent:)'
        if re.search(email_pattern, text, re.IGNORECASE):
            # Check if it's quoted
            lines = text.split('\n')[:10]
            quoted_lines = sum(1 for line in lines if line.strip().startswith('>'))
            if quoted_lines > len(lines) * 0.5:
                return False, 0.8  # Likely quoted email
            return True, 0.9
        
        return False, 0.1
    
    def get_prompt_hints(self) -> str:
        return """For emails:
- Look for From/To/Subject headers at the start
- Distinguish between new emails and quoted/forwarded content
- Email chains may span multiple pages
- Check for reply indicators (Re:, Fwd:, >, >>)
- Attachments mentioned in emails may follow on subsequent pages"""


class InvoiceStrategy(DocumentTypeStrategy):
    """Strategy for invoice document detection."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues an invoice."""
        
        # Look for invoice continuation patterns
        current_text = current_page['text'].lower()
        previous_text = previous_page['text'].lower()
        
        # Check for same invoice number
        invoice_pattern = r'invoice\s*#?\s*[:]\s*(\w+)'
        current_invoice = re.search(invoice_pattern, current_text)
        previous_invoice = re.search(invoice_pattern, previous_text)
        
        if current_invoice and previous_invoice:
            if current_invoice.group(1) == previous_invoice.group(1):
                return True, 0.9, "Same invoice number"
        
        # Check for continuation indicators
        if 'continued' in current_text or 'page 2' in current_text:
            return True, 0.8, "Continuation indicator found"
        
        # Check for line items pattern
        if self._has_line_items(current_page) and self._has_line_items(previous_page):
            return True, 0.7, "Continuing line items"
        
        return False, 0.3, "No continuation indicators"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new invoice."""
        text = page_info['text'][:1000].lower()
        
        # Strong indicators
        if 'invoice' in text and any(word in text for word in ['bill to', 'invoice date', 'invoice #']):
            return True, 0.95
        
        # Medium indicators
        if 'invoice' in text and any(word in text for word in ['total', 'amount due', 'payment']):
            return True, 0.7
        
        return False, 0.1
    
    def _has_line_items(self, page_info: Dict) -> bool:
        """Check if page has line item pattern."""
        text = page_info['text']
        # Look for patterns like quantity, description, price
        line_pattern = r'\d+\s+.+\s+\$?\d+\.\d{2}'
        matches = re.findall(line_pattern, text)
        return len(matches) > 3
    
    def get_prompt_hints(self) -> str:
        return """For invoices:
- Look for "Invoice", "Bill To", "Invoice Date", "Invoice #"
- Multi-page invoices have consistent headers/footers
- Line items may continue across pages
- Look for subtotals and page totals
- Payment terms usually appear on last page"""


class RFIStrategy(DocumentTypeStrategy):
    """Strategy for RFI (Request for Information) detection."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues an RFI."""
        
        current_text = current_page['text'].lower()
        previous_text = previous_page['text'].lower()
        
        # Check for same RFI number
        rfi_pattern = r'rfi\s*#?\s*[:]\s*(\d+)'
        current_rfi = re.search(rfi_pattern, current_text)
        previous_rfi = re.search(rfi_pattern, previous_text)
        
        if current_rfi and previous_rfi:
            if current_rfi.group(1) == previous_rfi.group(1):
                return True, 0.95, "Same RFI number"
        
        # Check for response continuation
        if 'response:' in previous_text and not 'question:' in current_text:
            return True, 0.8, "Continuing RFI response"
        
        return False, 0.3, "No RFI continuation indicators"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new RFI."""
        text = page_info['text'][:1000].lower()
        
        # Strong RFI indicators
        if 'request for information' in text or ('rfi' in text and '#' in text):
            return True, 0.9
        
        # Medium indicators
        if 'rfi' in text and any(word in text for word in ['project', 'question', 'response']):
            return True, 0.7
        
        return False, 0.1
    
    def get_prompt_hints(self) -> str:
        return """For RFIs:
- Look for "RFI #", "Request for Information"
- RFIs have question/response format
- May include attachments or drawings
- Response may span multiple pages
- Check for project name/number consistency"""


class DrawingStrategy(DocumentTypeStrategy):
    """Strategy for technical drawing detection."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues a drawing set."""
        
        # Drawings often have minimal text
        if current_page['text_length'] < 100 and previous_page['text_length'] < 100:
            # Both pages are likely drawings
            if current_page['has_images'] and previous_page['has_images']:
                return True, 0.7, "Sequential drawing pages"
        
        # Check for sheet numbers
        current_text = current_page['text'].lower()
        sheet_pattern = r'sheet\s*(\d+)\s*of\s*(\d+)'
        current_sheet = re.search(sheet_pattern, current_text)
        
        if current_sheet:
            current_num = int(current_sheet.group(1))
            previous_text = previous_page['text'].lower()
            previous_sheet = re.search(sheet_pattern, previous_text)
            
            if previous_sheet:
                previous_num = int(previous_sheet.group(1))
                if current_num == previous_num + 1:
                    return True, 0.95, "Sequential sheet numbers"
        
        return False, 0.3, "No drawing set indicators"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new drawing."""
        # Drawings have specific characteristics
        if page_info['has_images'] and page_info['text_length'] < 200:
            # Look for title block patterns
            text = page_info['text'].lower()
            if any(word in text for word in ['drawing', 'sheet', 'scale', 'project']):
                return True, 0.8
            elif page_info['has_many_drawings']:
                return True, 0.7
        
        return False, 0.1
    
    def get_prompt_hints(self) -> str:
        return """For drawings:
- Technical drawings have minimal text
- Look for title blocks with project info
- Sheet numbers (e.g., "Sheet 2 of 5")
- Drawing sets are sequential
- May have revision clouds or notes"""


class SubmittalStrategy(DocumentTypeStrategy):
    """Strategy for submittal document detection."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Check if current page continues a submittal."""
        
        current_text = current_page['text'].lower()
        previous_text = previous_page['text'].lower()
        
        # Check for same submittal number
        submittal_pattern = r'submittal\s*#?\s*[:]\s*(\w+)'
        current_sub = re.search(submittal_pattern, current_text)
        previous_sub = re.search(submittal_pattern, previous_text)
        
        if current_sub and previous_sub:
            if current_sub.group(1) == previous_sub.group(1):
                return True, 0.9, "Same submittal number"
        
        # Submittals often have attachments
        if 'attachment' in previous_text or 'exhibit' in previous_text:
            return True, 0.7, "Likely submittal attachment"
        
        return False, 0.3, "No submittal continuation"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Detect if this page starts a new submittal."""
        text = page_info['text'][:1000].lower()
        
        if 'submittal' in text and any(word in text for word in ['approval', 'review', 'specification']):
            return True, 0.9
        
        if 'submittal' in text:
            return True, 0.7
        
        return False, 0.1
    
    def get_prompt_hints(self) -> str:
        return """For submittals:
- Look for "Submittal #", "For Approval", "For Review"
- Include product data, samples, or shop drawings
- May have multiple attachments
- Check for specification section references
- Often have transmittal cover sheets"""


class DefaultStrategy(DocumentTypeStrategy):
    """Default strategy for unknown document types."""
    
    def validate_continuation(
        self, 
        current_page: Dict, 
        previous_page: Dict,
        window_context: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Generic continuation check."""
        
        # Check for page numbers
        current_text = current_page['text'].lower()
        page_pattern = r'page\s*(\d+)\s*of\s*(\d+)'
        current_page_num = re.search(page_pattern, current_text)
        
        if current_page_num:
            current_num = int(current_page_num.group(1))
            previous_text = previous_page['text'].lower()
            previous_page_num = re.search(page_pattern, previous_text)
            
            if previous_page_num:
                previous_num = int(previous_page_num.group(1))
                if current_num == previous_num + 1:
                    return True, 0.8, "Sequential page numbers"
        
        # Check for continuation words
        if current_text.startswith(('continued', '...', '(continued)')):
            return True, 0.7, "Continuation marker"
        
        return False, 0.3, "No clear continuation"
    
    def detect_document_start(self, page_info: Dict) -> Tuple[bool, float]:
        """Generic document start detection."""
        text = page_info['text'][:500]
        
        # Look for common document headers
        if any(pattern in text.lower() for pattern in ['memorandum', 'letter', 'report', 'notice']):
            return True, 0.6
        
        # Look for date and addressing
        if re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', text):
            if 'Dear' in text or 'To:' in text:
                return True, 0.7
        
        return False, 0.2
    
    def get_prompt_hints(self) -> str:
        return """For general documents:
- Look for letterhead or headers
- Check for date and addressing format
- Page numbers indicate multi-page documents
- Continuation markers like "(continued)"
- Signature blocks indicate document end"""


class DocumentTypeStrategyFactory:
    """Factory for creating document type strategies."""
    
    @staticmethod
    def get_strategy(document_type: str, llm_client: Optional[LLMClient] = None) -> DocumentTypeStrategy:
        """Get appropriate strategy for document type."""
        strategies = {
            'email': EmailStrategy,
            'invoice': InvoiceStrategy,
            'rfi': RFIStrategy,
            'drawing': DrawingStrategy,
            'submittal': SubmittalStrategy,
        }
        
        strategy_class = strategies.get(document_type.lower(), DefaultStrategy)
        return strategy_class(llm_client)
    
    @staticmethod
    def get_all_strategies(llm_client: Optional[LLMClient] = None) -> Dict[str, DocumentTypeStrategy]:
        """Get all available strategies."""
        return {
            'email': EmailStrategy(llm_client),
            'invoice': InvoiceStrategy(llm_client),
            'rfi': RFIStrategy(llm_client),
            'drawing': DrawingStrategy(llm_client),
            'submittal': SubmittalStrategy(llm_client),
            'default': DefaultStrategy(llm_client)
        }