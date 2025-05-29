"""PDF splitter for breaking large PDFs into logical documents."""

import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from loguru import logger

from backend.shared.exceptions import PDFExtractionError
from .models import (
    Document,
    DocumentPage,
    DocumentType,
    ProcessingResult,
    PDFProcessingRequest,
)
from .boundary_detector import BoundaryDetector


class PDFSplitter:
    """Splits large PDF files into individual logical documents."""
    
    def __init__(self):
        self.boundary_detector = BoundaryDetector()
    
    def process_pdf(self, request: PDFProcessingRequest) -> ProcessingResult:
        """Process a PDF file and extract individual documents.
        
        Args:
            request: Processing request with file path and options
            
        Returns:
            ProcessingResult with extracted documents
        """
        start_time = time.time()
        result = ProcessingResult(
            success=False,
            source_pdf_path=request.file_path,
            source_pdf_id=str(uuid.uuid4()),
            total_pages=0,
            documents_found=0,
            processing_time=0,
        )
        
        try:
            # Validate file exists
            if not request.file_path.exists():
                raise PDFExtractionError(f"PDF file not found: {request.file_path}")
            
            # Open PDF
            pdf_doc = fitz.open(str(request.file_path))
            result.total_pages = pdf_doc.page_count
            
            logger.info(f"Processing PDF: {request.file_path.name} ({result.total_pages} pages)")
            
            if request.split_documents:
                # Detect document boundaries
                boundaries = self.boundary_detector.detect_boundaries(pdf_doc)
                logger.info(f"Found {len(boundaries)} document boundaries")
                
                # Extract individual documents
                for i, (start, end) in enumerate(boundaries):
                    if request.max_pages and i >= request.max_pages:
                        result.warnings.append(f"Stopped at {request.max_pages} documents (limit reached)")
                        break
                    
                    document = self._extract_document(
                        pdf_doc,
                        start,
                        end,
                        result.source_pdf_id,
                        request,
                    )
                    result.documents.append(document)
            else:
                # Treat entire PDF as one document
                document = self._extract_document(
                    pdf_doc,
                    0,
                    result.total_pages - 1,
                    result.source_pdf_id,
                    request,
                )
                result.documents.append(document)
            
            pdf_doc.close()
            
            # Update result
            result.success = True
            result.documents_found = len(result.documents)
            result.processing_time = time.time() - start_time
            
            # Calculate statistics
            for doc in result.documents:
                doc_type = doc.type
                result.page_classification[doc_type] = result.page_classification.get(doc_type, 0) + 1
                
                if doc.has_ocr_content:
                    result.ocr_pages += sum(1 for p in doc.pages if p.is_scanned)
            
            if result.documents:
                result.average_confidence = sum(
                    doc.average_ocr_confidence for doc in result.documents
                ) / len(result.documents)
            
            logger.info(
                f"Successfully processed {result.documents_found} documents "
                f"in {result.processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
        
        return result
    
    def _extract_document(
        self,
        pdf_doc: fitz.Document,
        start_page: int,
        end_page: int,
        source_pdf_id: str,
        request: PDFProcessingRequest,
    ) -> Document:
        """Extract a single document from page range.
        
        Args:
            pdf_doc: PyMuPDF document object
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (inclusive)
            source_pdf_id: ID of the source PDF
            request: Processing request options
            
        Returns:
            Extracted Document object
        """
        doc_id = str(uuid.uuid4())
        pages: List[DocumentPage] = []
        full_text = []
        
        # Extract pages
        for page_num in range(start_page, end_page + 1):
            page = pdf_doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Check if page is scanned (no text extracted)
            is_scanned = len(text.strip()) < 10  # Arbitrary threshold
            
            if is_scanned and request.perform_ocr:
                # TODO: Implement OCR using pytesseract
                # For now, just mark as scanned
                text = "[OCR needed]"
            
            # Check for tables and images
            has_tables = self._detect_tables(page)
            has_images = len(page.get_images()) > 0
            
            doc_page = DocumentPage(
                page_number=page_num + 1,  # 1-indexed for display
                text=text,
                is_scanned=is_scanned,
                has_tables=has_tables,
                has_images=has_images,
            )
            
            pages.append(doc_page)
            full_text.append(text)
        
        # Create document
        document = Document(
            id=doc_id,
            source_pdf_id=source_pdf_id,
            source_pdf_path=Path(pdf_doc.name),
            pages=pages,
            page_range=(start_page + 1, end_page + 1),  # 1-indexed for display
            text="\n\n".join(full_text),
        )
        
        # Classify document type if requested
        if request.classify_documents:
            # TODO: Implement classification using DistilBERT
            # For now, use simple heuristics
            document.type = self._classify_document_simple(document.text)
        
        # Extract metadata if requested
        if request.extract_metadata:
            # TODO: Implement metadata extraction
            pass
        
        return document
    
    def _detect_tables(self, page: fitz.Page) -> bool:
        """Detect if a page contains tables.
        
        Simple heuristic based on line detection.
        """
        # Look for horizontal and vertical lines that might indicate tables
        drawings = page.get_drawings()
        
        h_lines = 0
        v_lines = 0
        
        for item in drawings:
            if "l" in item:  # Line items
                for line in item["l"]:
                    if len(line) == 2:  # Simple line with two points
                        p1, p2 = line
                        # Horizontal line
                        if abs(p1.y - p2.y) < 2:
                            h_lines += 1
                        # Vertical line
                        elif abs(p1.x - p2.x) < 2:
                            v_lines += 1
        
        # If we have multiple horizontal and vertical lines, likely a table
        return h_lines > 3 and v_lines > 3
    
    def _classify_document_simple(self, text: str) -> DocumentType:
        """Simple rule-based document classification.
        
        This is a placeholder until DistilBERT integration.
        """
        text_lower = text.lower()
        
        # Email patterns
        if any(marker in text_lower for marker in ["from:", "to:", "subject:", "sent:"]):
            return DocumentType.EMAIL
        
        # RFI patterns
        if "request for information" in text_lower or "rfi" in text_lower:
            return DocumentType.RFI
        
        # Change order patterns
        if "change order" in text_lower or "c.o." in text_lower:
            return DocumentType.CHANGE_ORDER
        
        # Invoice patterns
        if any(term in text_lower for term in ["invoice", "bill", "amount due"]):
            return DocumentType.INVOICE
        
        # Contract patterns
        if any(term in text_lower for term in ["agreement", "contract", "terms and conditions"]):
            return DocumentType.CONTRACT
        
        # Meeting minutes
        if "meeting minutes" in text_lower or "attendees:" in text_lower:
            return DocumentType.MEETING_MINUTES
        
        # Default to unknown
        return DocumentType.UNKNOWN