"""Document processor module for CLaiM.

This module handles:
- PDF splitting into logical documents
- Document boundary detection
- OCR processing for scanned pages
- Basic metadata extraction
"""

from .pdf_splitter import PDFSplitter
from .models import (
    Document,
    DocumentPage,
    DocumentMetadata,
    DocumentType,
    ProcessingResult,
)

__all__ = [
    "PDFSplitter",
    "Document",
    "DocumentPage",
    "DocumentMetadata",
    "DocumentType",
    "ProcessingResult",
]