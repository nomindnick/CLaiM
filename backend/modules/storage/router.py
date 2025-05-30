"""FastAPI router for storage operations."""

from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import FileResponse
from loguru import logger
import os

from api.config import get_settings
from shared.exceptions import StorageError
from .models import (
    StoredDocument,
    SearchFilter,
    SearchResult,
    StorageStats,
    BulkImportRequest,
    BulkImportResult,
)
from .storage_manager import StorageManager

router = APIRouter(tags=["storage"])

# Dependency to get storage manager
_storage_manager = None


def get_storage_manager() -> StorageManager:
    """Get storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


@router.get("/documents/{document_id}", response_model=StoredDocument)
async def get_document(
    document_id: str,
    include_pages: bool = Query(False, description="Include page data"),
    storage: StorageManager = Depends(get_storage_manager),
):
    """Get document by ID."""
    document = storage.get_document(document_id, include_pages)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/documents/search", response_model=SearchResult)
async def search_documents(
    filter: SearchFilter = Body(...),
    storage: StorageManager = Depends(get_storage_manager),
):
    """Search documents with filters."""
    try:
        return storage.search_documents(filter)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    delete_files: bool = Query(True, description="Delete associated files"),
    storage: StorageManager = Depends(get_storage_manager),
):
    """Delete document from storage."""
    if not storage.delete_document(document_id, delete_files):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}


@router.get("/stats", response_model=StorageStats)
async def get_storage_stats(
    storage: StorageManager = Depends(get_storage_manager),
):
    """Get storage statistics."""
    return storage.get_stats()


@router.post("/import/bulk", response_model=BulkImportResult)
async def bulk_import_documents(
    request: BulkImportRequest = Body(...),
    storage: StorageManager = Depends(get_storage_manager),
):
    """Bulk import PDF documents."""
    try:
        return storage.bulk_import(request)
    except Exception as e:
        logger.error(f"Bulk import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk import failed: {str(e)}")


@router.post("/optimize")
async def optimize_storage(
    storage: StorageManager = Depends(get_storage_manager),
):
    """Optimize storage by cleaning up and compacting."""
    try:
        storage.optimize_storage()
        return {"message": "Storage optimization complete"}
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/documents/by-source/{source_pdf_id}", response_model=List[StoredDocument])
async def get_documents_by_source(
    source_pdf_id: str,
    storage: StorageManager = Depends(get_storage_manager),
):
    """Get all documents from a source PDF."""
    filter = SearchFilter(query=None)
    # Use SQL directly for now
    with storage.db._get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM documents
            WHERE source_pdf_id = ?
            ORDER BY page_start ASC
        """, (source_pdf_id,)).fetchall()
        
        documents = [storage.db._row_to_document(row) for row in rows]
        return documents


@router.get("/documents/recent", response_model=List[StoredDocument])
async def get_recent_documents(
    limit: int = Query(10, ge=1, le=100),
    storage: StorageManager = Depends(get_storage_manager),
):
    """Get recently added documents."""
    filter = SearchFilter(
        limit=limit,
        sort_by="created_at",
        sort_descending=True,
    )
    result = storage.search_documents(filter)
    return result.documents


@router.get("/documents/{document_id}/pdf")
async def get_document_pdf(
    document_id: str,
    storage: StorageManager = Depends(get_storage_manager),
):
    """Get PDF file for document."""
    document = storage.get_document(document_id, include_pages=False)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if extracted PDF exists
    extracted_path = os.path.join("storage", "extracted", f"{document_id}.pdf")
    if os.path.exists(extracted_path):
        return FileResponse(
            path=extracted_path,
            media_type="application/pdf",
            filename=f"{document.title}.pdf"
        )
    
    # Fall back to source PDF if available
    if document.source_pdf_path and os.path.exists(document.source_pdf_path):
        return FileResponse(
            path=document.source_pdf_path,
            media_type="application/pdf",
            filename=f"{document.title}.pdf"
        )
    
    raise HTTPException(status_code=404, detail="PDF file not found")