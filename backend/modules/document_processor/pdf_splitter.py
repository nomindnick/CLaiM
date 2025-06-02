"""PDF splitter for breaking large PDFs into logical documents."""

import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from loguru import logger

from shared.exceptions import PDFExtractionError, OCRError
from .models import (
    Document,
    DocumentPage,
    DocumentType,
    ProcessingResult,
    PDFProcessingRequest,
)
from .boundary_detector import BoundaryDetector
from .ocr_handler import OCRHandler
from .improved_ocr_handler import ImprovedOCRHandler
from .hybrid_boundary_detector import HybridBoundaryDetector, DetectionLevel
from .hybrid_text_extractor import HybridTextExtractor, TextExtractionMethod
# Lazy import to avoid circular dependencies
# from ..ai_classifier.classifier import document_classifier
# from ..ai_classifier.models import ClassificationRequest


class PDFSplitter:
    """Splits large PDF files into individual logical documents."""
    
    def __init__(self, use_visual_detection: bool = True, use_hybrid_text_extraction: bool = True):
        self.ocr_handler = None  # Lazy initialization
        self.boundary_detector = None  # Will be initialized when OCR handler is ready
        self.hybrid_detector = None  # Hybrid boundary detector
        self.text_extractor = None  # Hybrid text extractor
        self.use_visual_detection = use_visual_detection
        self.use_hybrid_text_extraction = use_hybrid_text_extraction
    
    def process_pdf(self, request: PDFProcessingRequest, progress_callback=None) -> ProcessingResult:
        """Process a PDF file and extract individual documents.
        
        Args:
            request: Processing request with file path and options
            progress_callback: Optional callback function(current, total, message)
            
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
            
            if progress_callback:
                progress_callback(0, result.total_pages, "Starting PDF processing")
            
            # Initialize OCR handler if needed (for boundary detection on scanned docs)
            if request.perform_ocr and self.ocr_handler is None:
                # Use improved OCR handler for much better accuracy on CPRA-style documents
                # Share cache directory for all OCR operations
                self.ocr_handler = ImprovedOCRHandler(
                    language=request.ocr_language,
                    min_confidence=0.4,  # Slightly higher threshold due to improved accuracy
                    cache_dir=".ocr_cache"  # Shared cache directory
                )
            
            # Initialize boundary detector
            if self.use_visual_detection:
                # Use hybrid detector for better accuracy
                if self.hybrid_detector is None:
                    self.hybrid_detector = HybridBoundaryDetector(
                        ocr_handler=self.ocr_handler,
                        visual_model="clip-ViT-B-32",
                        cache_dir=".boundary_cache"
                    )
            else:
                # Use pattern-based detector only
                if self.boundary_detector is None:
                    self.boundary_detector = BoundaryDetector(ocr_handler=self.ocr_handler)
            
            if request.split_documents:
                # Detect document boundaries
                if progress_callback:
                    progress_callback(1, result.total_pages, "Detecting document boundaries")
                
                if self.use_visual_detection:
                    # Use hybrid detection with LLM validation for enhanced accuracy
                    max_level = DetectionLevel.LLM  # Default to LLM-enhanced detection
                    
                    # Check if specific detection level is requested
                    if hasattr(request, 'max_detection_level') and request.max_detection_level:
                        # Convert string to enum if needed
                        if isinstance(request.max_detection_level, str):
                            max_level = getattr(DetectionLevel, request.max_detection_level.upper(), DetectionLevel.LLM)
                        else:
                            max_level = request.max_detection_level
                    elif hasattr(request, 'force_visual_only') and request.force_visual_only:
                        max_level = DetectionLevel.VISUAL
                    
                    detection_result = self.hybrid_detector.detect_boundaries(
                        pdf_doc,
                        max_level=max_level,
                        force_visual=request.force_visual_detection if hasattr(request, 'force_visual_detection') else False,
                        force_llm=request.force_llm_validation if hasattr(request, 'force_llm_validation') else False
                    )
                    boundaries = detection_result.boundaries
                    
                    # Store detection metadata for later use
                    result.detection_level = detection_result.detection_level.name.lower()
                    result.boundary_confidence = detection_result.confidence_scores
                    
                    logger.info(
                        f"Found {len(boundaries)} document boundaries using {detection_result.detection_level.name.lower()} detection"
                    )
                else:
                    # Use simple pattern detection
                    boundaries = self.boundary_detector.detect_boundaries(pdf_doc)
                    logger.info(f"Found {len(boundaries)} document boundaries")
                
                # Fill gaps in boundaries to ensure all pages are covered
                boundaries = self._fill_boundary_gaps(boundaries, result.total_pages)
                
                # Extract individual documents
                for i, (start, end) in enumerate(boundaries):
                    if request.max_pages and i >= request.max_pages:
                        result.warnings.append(f"Stopped at {request.max_pages} documents (limit reached)")
                        break
                    
                    if progress_callback:
                        progress_callback(
                            start, 
                            result.total_pages, 
                            f"Extracting document {i+1}/{len(boundaries)} (pages {start+1}-{end+1})"
                        )
                    
                    document = self._extract_document(
                        pdf_doc,
                        start,
                        end,
                        result.source_pdf_id,
                        request,
                        progress_callback,
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
        progress_callback=None,
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
            
            # Extract text using intelligent hybrid approach
            if self.use_hybrid_text_extraction and request.perform_ocr:
                # Initialize hybrid text extractor if needed
                if self.text_extractor is None:
                    self.text_extractor = HybridTextExtractor(
                        language=request.ocr_language,
                        min_confidence=request.min_confidence,
                        ocr_handler=self.ocr_handler  # Share the OCR handler for caching
                    )
                
                try:
                    # Use hybrid extraction
                    text, confidence, extraction_method = self.text_extractor.extract_text(page)
                    
                    # Determine if page is scanned based on extraction method
                    is_scanned = extraction_method != TextExtractionMethod.PYMUPDF
                    
                    logger.info(
                        f"Page {page_num + 1}: {len(text)} chars using {extraction_method.value} "
                        f"(confidence: {confidence:.3f})"
                    )
                    
                except Exception as e:
                    logger.error(f"Hybrid text extraction failed on page {page_num + 1}: {e}")
                    text = "[Text extraction failed]"
                    confidence = 0.0
                    is_scanned = True
            else:
                # Fallback to simple PyMuPDF extraction
                text = page.get_text()
                is_scanned = len(text.strip()) < 10  # Arbitrary threshold
                confidence = 1.0 if not is_scanned else 0.0
                
                # Traditional OCR fallback if needed
                if is_scanned and request.perform_ocr:
                    # Initialize OCR handler if needed
                    if self.ocr_handler is None:
                        self.ocr_handler = OCRHandler(
                            language=request.ocr_language,
                            min_confidence=request.min_confidence
                        )
                    
                    try:
                        # Perform OCR
                        ocr_text, ocr_confidence = self.ocr_handler.process_page(page)
                        
                        if ocr_text and ocr_confidence >= request.min_confidence:
                            text = ocr_text
                            confidence = ocr_confidence
                            logger.info(
                                f"OCR successful for page {page_num + 1} "
                                f"(confidence: {confidence:.2f})"
                            )
                        else:
                            text = "[OCR failed - low confidence]"
                            confidence = ocr_confidence
                            logger.warning(
                                f"OCR failed for page {page_num + 1} - "
                                f"confidence {ocr_confidence:.2f} below threshold"
                            )
                            
                    except OCRError as e:
                        logger.error(f"OCR error on page {page_num + 1}: {e}")
                        text = "[OCR failed - error]"
                        confidence = 0.0
                    except Exception as e:
                        logger.error(f"Unexpected OCR error on page {page_num + 1}: {e}")
                        text = "[OCR failed - unexpected error]"
                        confidence = 0.0
            
            # Check for tables and images
            has_tables = self._detect_tables(page)
            has_images = len(page.get_images()) > 0
            
            doc_page = DocumentPage(
                page_number=page_num + 1,  # 1-indexed for display
                text=text,
                is_scanned=is_scanned,
                has_tables=has_tables,
                has_images=has_images,
                confidence=confidence,
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
            try:
                # Lazy import to avoid circular dependencies
                from ..ai_classifier.classifier import document_classifier
                from ..ai_classifier.models import ClassificationRequest
                
                # Use AI classifier for document type prediction
                classification_request = ClassificationRequest(
                    text=document.text,
                    title=f"Document from pages {start_page + 1}-{end_page + 1}",
                    min_confidence=0.3,
                    require_reasoning=False
                )
                
                classification_result = document_classifier.classify(classification_request)
                document.type = classification_result.document_type
                document.classification_confidence = classification_result.confidence
                
                logger.info(
                    f"Classified document (pages {start_page + 1}-{end_page + 1}) as "
                    f"{document.type.value} with confidence {document.classification_confidence:.3f}"
                )
                
            except Exception as e:
                logger.error(f"AI classification failed, using fallback: {e}")
                # Fallback to simple rule-based classification
                document.type = self._classify_document_simple(document.text)
                document.classification_confidence = 0.5  # Default confidence for fallback
        
        # Extract metadata if requested
        if request.extract_metadata:
            # TODO: Implement metadata extraction
            pass
        
        return document
    
    def _fill_boundary_gaps(self, boundaries: List[Tuple[int, int]], total_pages: int) -> List[Tuple[int, int]]:
        """Fill gaps between detected boundaries to ensure all pages are covered.
        
        Args:
            boundaries: List of (start, end) page ranges (0-indexed)
            total_pages: Total number of pages in the PDF
            
        Returns:
            Complete list of boundaries covering all pages
        """
        if not boundaries:
            # No boundaries found, treat entire PDF as one document
            return [(0, total_pages - 1)]
        
        # Sort boundaries by start page
        sorted_boundaries = sorted(boundaries)
        filled_boundaries = []
        
        # Check if there's a gap before the first boundary
        first_start = sorted_boundaries[0][0]
        if first_start > 0:
            filled_boundaries.append((0, first_start - 1))
            logger.info(f"Added gap-filling boundary: pages 1-{first_start}")
        
        # Add the first boundary
        filled_boundaries.append(sorted_boundaries[0])
        
        # Check for gaps between boundaries and fill them
        for i in range(1, len(sorted_boundaries)):
            prev_end = filled_boundaries[-1][1]
            current_start = sorted_boundaries[i][0]
            
            # If there's a gap, fill it
            if current_start > prev_end + 1:
                gap_start = prev_end + 1
                gap_end = current_start - 1
                filled_boundaries.append((gap_start, gap_end))
                logger.info(f"Added gap-filling boundary: pages {gap_start + 1}-{gap_end + 1}")
            
            # Add the current boundary
            filled_boundaries.append(sorted_boundaries[i])
        
        # Check if there's a gap after the last boundary
        last_end = filled_boundaries[-1][1]
        if last_end < total_pages - 1:
            gap_start = last_end + 1
            gap_end = total_pages - 1
            filled_boundaries.append((gap_start, gap_end))
            logger.info(f"Added gap-filling boundary: pages {gap_start + 1}-{gap_end + 1}")
        
        logger.info(f"Boundary gap filling: {len(boundaries)} original -> {len(filled_boundaries)} total boundaries")
        return filled_boundaries
    
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