"""Main storage manager that coordinates SQLite, file system, and future vector storage."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

import fitz  # PyMuPDF
from loguru import logger

from api.config import get_settings
from shared.exceptions import StorageError
from modules.document_processor.models import Document, DocumentPage, DocumentType
from .models import (
    StoredDocument,
    StoredPage,
    DocumentMetadata,
    SearchFilter,
    SearchResult,
    StorageStatus,
    StorageStats,
    BulkImportRequest,
    BulkImportResult,
)
from .sqlite_handler import SQLiteHandler


class StorageManager:
    """Manages document storage across SQLite, file system, and vector store."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize storage manager.
        
        Args:
            storage_dir: Base directory for storage (uses settings if not provided)
        """
        settings = get_settings()
        self.storage_dir = storage_dir or settings.storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-directories
        self.db_dir = self.storage_dir / "database"
        self.pdf_dir = self.storage_dir / "pdfs"
        self.extracted_dir = self.storage_dir / "extracted"
        
        for dir_path in [self.db_dir, self.pdf_dir, self.extracted_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite handler
        self.db = SQLiteHandler(self.db_dir / "documents.db")
        
        # TODO: Initialize Qdrant vector store
        # TODO: Initialize DuckDB for graph storage
        
        logger.info(f"Storage manager initialized at {self.storage_dir}")
    
    def store_document(self, document: Document, source_pdf_path: Path) -> str:
        """Store a processed document.
        
        Args:
            document: Processed document from document processor
            source_pdf_path: Path to source PDF file
            
        Returns:
            Stored document ID
        """
        try:
            # Convert to storage model
            stored_doc = self._convert_to_stored_document(document, source_pdf_path)
            
            # Copy source PDF if not already in storage
            stored_pdf_path = self._store_source_pdf(source_pdf_path, stored_doc.source_pdf_id)
            stored_doc.source_pdf_path = stored_pdf_path
            
            # Extract document pages to separate PDF
            if document.pages:
                extracted_path = self._extract_document_pages(
                    stored_pdf_path,
                    stored_doc.page_range[0] - 1,  # Convert to 0-indexed
                    stored_doc.page_range[1] - 1,
                    stored_doc.id
                )
                stored_doc.storage_path = extracted_path
            
            # Convert pages
            stored_pages = []
            if document.pages:
                for page in document.pages:
                    stored_page = StoredPage(
                        id=str(uuid4()),
                        document_id=stored_doc.id,
                        page_number=page.page_number,
                        text=page.text,
                        is_scanned=page.is_scanned,
                        has_tables=page.has_tables,
                        has_images=page.has_images,
                        ocr_confidence=page.confidence,
                    )
                    stored_pages.append(stored_page)
            
            # Save to database
            self.db.save_document(stored_doc, stored_pages)
            
            # Update status
            stored_doc.status = StorageStatus.STORED
            stored_doc.updated_at = datetime.utcnow()
            self.db.update_document_status(stored_doc.id, StorageStatus.STORED)
            
            # TODO: Generate and store embeddings
            # TODO: Index in vector store
            
            logger.info(f"Successfully stored document {stored_doc.id}")
            return stored_doc.id
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise StorageError(f"Failed to store document: {e}")
    
    def _convert_to_stored_document(self, document: Document, source_pdf_path: Path) -> StoredDocument:
        """Convert processed document to storage model."""
        # Extract metadata using the metadata extractor
        from modules.metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor()
        extracted_metadata = extractor.extract_metadata(document)
        
        # Convert to storage metadata format
        metadata = DocumentMetadata(
            dates=extracted_metadata.date and [extracted_metadata.date] or [],
            parties=[p.name for p in extracted_metadata.parties],
            reference_numbers=self._group_reference_numbers(extracted_metadata.reference_numbers),
            amounts=extracted_metadata.amounts,
            custom_fields={
                "keywords": extracted_metadata.keywords,
                "project_number": extracted_metadata.project_number,
                "project_name": extracted_metadata.project_name,
            }
        )
        
        # Use extracted title/subject
        title = extracted_metadata.title or extracted_metadata.subject or f"{document.type.value} Document"
        
        return StoredDocument(
            id=document.id,
            source_pdf_id=document.source_pdf_id,
            source_pdf_path=source_pdf_path,
            type=document.type,
            page_range=document.page_range,
            page_count=len(document.pages) if document.pages else 0,
            title=title,
            text=document.text,
            summary="",  # Summary will be generated later
            metadata=metadata,
            classification_confidence=document.classification_confidence,
            status=StorageStatus.PENDING,
        )
    
    def _group_reference_numbers(self, ref_numbers: List[str]) -> Dict[str, List[str]]:
        """Group reference numbers by type."""
        grouped = {}
        for ref in ref_numbers:
            # Parse the reference format (e.g., "RFI 123" -> type="RFI", num="123")
            parts = ref.split(' ', 1)
            if len(parts) == 2:
                ref_type = parts[0].replace('#', '')
                ref_num = parts[1].replace('#', '')
                if ref_type not in grouped:
                    grouped[ref_type] = []
                grouped[ref_type].append(ref_num)
        return grouped
    
    def _store_source_pdf(self, source_path: Path, pdf_id: str) -> Path:
        """Store source PDF in storage directory."""
        dest_path = self.pdf_dir / f"{pdf_id}_{source_path.name}"
        
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.info(f"Stored source PDF: {dest_path.name}")
        
        return dest_path
    
    def _extract_document_pages(self, pdf_path: Path, start_page: int, end_page: int, doc_id: str) -> Path:
        """Extract specific pages from PDF to new file."""
        output_path = self.extracted_dir / f"{doc_id}.pdf"
        
        try:
            # Open source PDF
            src_pdf = fitz.open(str(pdf_path))
            
            # Create new PDF with selected pages
            dest_pdf = fitz.open()
            
            for page_num in range(start_page, end_page + 1):
                if page_num < src_pdf.page_count:
                    dest_pdf.insert_pdf(src_pdf, from_page=page_num, to_page=page_num)
            
            # Save extracted pages
            dest_pdf.save(str(output_path))
            dest_pdf.close()
            src_pdf.close()
            
            logger.info(f"Extracted pages {start_page+1}-{end_page+1} to {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to extract pages: {e}")
            raise StorageError(f"Failed to extract pages: {e}")
    
    def get_document(self, document_id: str, include_pages: bool = False) -> Optional[StoredDocument]:
        """Retrieve document by ID.
        
        Args:
            document_id: Document ID
            include_pages: Whether to include page data
            
        Returns:
            Stored document if found
        """
        return self.db.get_document(document_id, include_pages)
    
    def search_documents(self, filter: SearchFilter) -> SearchResult:
        """Search documents with filters.
        
        Args:
            filter: Search criteria
            
        Returns:
            Search results
        """
        # TODO: Implement hybrid search with vector store
        # For now, just use SQLite full-text search
        return self.db.search_documents(filter)
    
    def delete_document(self, document_id: str, delete_files: bool = True) -> bool:
        """Delete document from storage.
        
        Args:
            document_id: Document ID to delete
            delete_files: Whether to delete associated files
            
        Returns:
            True if deleted successfully
        """
        # Get document to find file paths
        doc = self.db.get_document(document_id)
        if not doc:
            return False
        
        # Delete from database
        if not self.db.delete_document(document_id):
            return False
        
        # Delete files if requested
        if delete_files and doc.storage_path and doc.storage_path.exists():
            try:
                doc.storage_path.unlink()
                logger.info(f"Deleted extracted PDF: {doc.storage_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete file {doc.storage_path}: {e}")
        
        # TODO: Remove from vector store
        
        return True
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = self.db.get_stats()
        
        # Calculate file storage sizes
        if self.pdf_dir.exists():
            pdf_size = sum(f.stat().st_size for f in self.pdf_dir.glob("*.pdf"))
            stats.file_storage_size_mb += pdf_size / (1024 * 1024)
        
        if self.extracted_dir.exists():
            extracted_size = sum(f.stat().st_size for f in self.extracted_dir.glob("*.pdf"))
            stats.file_storage_size_mb += extracted_size / (1024 * 1024)
        
        stats.total_size_mb = stats.database_size_mb + stats.file_storage_size_mb
        
        return stats
    
    def bulk_import(self, request: BulkImportRequest) -> BulkImportResult:
        """Bulk import multiple PDF files.
        
        Args:
            request: Bulk import request
            
        Returns:
            Import results
        """
        start_time = datetime.utcnow()
        result = BulkImportResult(
            total_files=len(request.source_pdf_paths),
            successful_imports=0,
            failed_imports=0,
            documents_created=0,
        )
        
        # Process each PDF
        for pdf_path in request.source_pdf_paths:
            try:
                # TODO: Process PDF through document processor
                # For now, just count as failed
                result.failed_imports += 1
                result.errors.append({
                    "file": str(pdf_path),
                    "error": "Document processor not yet integrated"
                })
                
            except Exception as e:
                logger.error(f"Failed to import {pdf_path}: {e}")
                result.failed_imports += 1
                result.errors.append({
                    "file": str(pdf_path),
                    "error": str(e)
                })
                
                if not request.continue_on_error:
                    break
        
        result.import_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        return result
    
    def optimize_storage(self) -> None:
        """Optimize storage by cleaning up and compacting."""
        logger.info("Starting storage optimization...")
        
        # Vacuum SQLite database
        self.db.vacuum()
        
        # Clean up orphaned files
        self._cleanup_orphaned_files()
        
        # TODO: Optimize vector store
        
        logger.info("Storage optimization complete")
    
    def _cleanup_orphaned_files(self) -> None:
        """Remove files not referenced in database."""
        # Get all document IDs and source PDF IDs from database
        with self.db._get_connection() as conn:
            doc_ids = set(row[0] for row in conn.execute("SELECT id FROM documents"))
            pdf_ids = set(row[0] for row in conn.execute("SELECT DISTINCT source_pdf_id FROM documents"))
        
        # Check extracted files
        for file_path in self.extracted_dir.glob("*.pdf"):
            doc_id = file_path.stem
            if doc_id not in doc_ids:
                logger.info(f"Removing orphaned extracted file: {file_path.name}")
                file_path.unlink()
        
        # Check source PDFs
        for file_path in self.pdf_dir.glob("*.pdf"):
            # Extract PDF ID from filename (format: {pdf_id}_{original_name}.pdf)
            pdf_id = file_path.name.split("_", 1)[0]
            if pdf_id not in pdf_ids:
                logger.info(f"Removing orphaned source PDF: {file_path.name}")
                file_path.unlink()