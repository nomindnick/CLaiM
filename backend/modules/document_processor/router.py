"""FastAPI router for document processing endpoints."""

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger

from api.config import settings
from shared.exceptions import DocumentProcessingError
from .models import ProcessingResult, PDFProcessingRequest
from .pdf_splitter import PDFSplitter
from modules.storage.storage_manager import StorageManager
from modules.metadata_extractor.extractor import MetadataExtractor


router = APIRouter(
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

# Initialize services
pdf_splitter = PDFSplitter()
storage_manager = StorageManager()
metadata_extractor = MetadataExtractor()


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    """Upload a PDF document for processing.
    
    Args:
        file: PDF file to upload
        background_tasks: FastAPI background tasks
        
    Returns:
        Upload confirmation with job ID
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check file size (if available)
    # Note: file.size might be None for streaming uploads
    if file.size is not None and file.size > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size / 1024 / 1024}MB"
        )
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = settings.upload_dir / f"{file_id}{file_ext}"
        
        # Save uploaded file
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded file: {file.filename} -> {save_path}")
        
        # Queue for processing
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_pdf_background,
            job_id=job_id,
            file_path=save_path,
        )
        
        return {
            "message": "File uploaded successfully",
            "document_id": file_id,  # Frontend expects document_id
            "file_id": file_id,
            "job_id": job_id,
            "filename": file.filename,
            "size": file.size,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )
    finally:
        file.file.close()


@router.post("/process/{file_id}", response_model=ProcessingResult)
async def process_document(file_id: str) -> ProcessingResult:
    """Process an uploaded PDF document.
    
    Args:
        file_id: ID of uploaded file
        
    Returns:
        Processing result with extracted documents
    """
    # Find uploaded file
    upload_files = list(settings.upload_dir.glob(f"{file_id}.*"))
    if not upload_files:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_id}"
        )
    
    file_path = upload_files[0]
    
    try:
        # Create processing request
        request = PDFProcessingRequest(
            file_path=file_path,
            split_documents=True,
            perform_ocr=True,
            extract_metadata=True,
            classify_documents=True,
        )
        
        # Process PDF
        result = pdf_splitter.process_pdf(request)
        
        if not result.success:
            raise DocumentProcessingError(
                "Failed to process PDF",
                details={"errors": result.errors}
            )
        
        return result
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Get status of a processing job.
    
    Args:
        job_id: Job ID from upload
        
    Returns:
        Job status information
    """
    # TODO: Implement job tracking with Redis or database
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job tracking not yet implemented"
    }


@router.get("/types")
async def get_document_types() -> List[str]:
    """Get list of supported document types.
    
    Returns:
        List of document type values
    """
    from .models import DocumentType
    return [dt.value for dt in DocumentType]


@router.get("/list")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    document_type: Optional[str] = None,
) -> dict:
    """List stored documents with pagination.
    
    Args:
        limit: Maximum documents to return
        offset: Number of documents to skip
        document_type: Filter by document type
        
    Returns:
        List of documents with metadata
    """
    from modules.storage.models import SearchFilter, DocumentType as StorageDocType
    
    try:
        # Build search filter
        filter = SearchFilter(
            limit=limit,
            offset=offset
        )
        
        if document_type:
            try:
                filter.document_type = StorageDocType(document_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid document type: {document_type}"
                )
        
        # Search documents
        results = storage_manager.search_documents(filter)
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "type": doc.type.value,
                    "page_count": doc.page_count,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "metadata": {
                        "dates": [d.isoformat() for d in doc.metadata.dates] if doc.metadata else [],
                        "parties": doc.metadata.parties if doc.metadata else [],
                        "amounts": doc.metadata.amounts if doc.metadata else [],
                    } if doc.metadata else None
                }
                for doc in results.documents
            ],
            "total": results.total_results,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{document_id}")
async def get_document(document_id: str, include_text: bool = False) -> dict:
    """Get a specific document by ID.
    
    Args:
        document_id: Document ID
        include_text: Whether to include full text content
        
    Returns:
        Document details
    """
    try:
        doc = storage_manager.get_document(document_id, include_pages=include_text)
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        response = {
            "id": doc.id,
            "title": doc.title,
            "type": doc.type.value,
            "page_count": doc.page_count,
            "page_range": doc.page_range,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "storage_path": str(doc.storage_path) if doc.storage_path else None,
            "metadata": {
                "dates": [d.isoformat() for d in doc.metadata.dates] if doc.metadata else [],
                "parties": doc.metadata.parties if doc.metadata else [],
                "reference_numbers": doc.metadata.reference_numbers if doc.metadata else {},
                "amounts": doc.metadata.amounts if doc.metadata else [],
                "custom_fields": doc.metadata.custom_fields if doc.metadata else {}
            } if doc.metadata else None
        }
        
        if include_text:
            response["text"] = doc.text
            response["summary"] = doc.summary
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document: {str(e)}"
        )


async def process_pdf_background(job_id: str, file_path: Path):
    """Background task to process PDF.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to PDF file
    """
    logger.info(f"Starting background processing for job {job_id}")
    
    try:
        request = PDFProcessingRequest(
            file_path=file_path,
            split_documents=True,
            perform_ocr=True,
            extract_metadata=True,
            classify_documents=True,
        )
        
        result = pdf_splitter.process_pdf(request)
        
        if result.success and result.documents:
            # Save each extracted document to storage
            stored_doc_ids = []
            for document in result.documents:
                try:
                    # Extract metadata is already done in storage_manager
                    doc_id = storage_manager.store_document(document, file_path)
                    stored_doc_ids.append(doc_id)
                    logger.info(f"Stored document {doc_id} from job {job_id}")
                except Exception as e:
                    logger.error(f"Failed to store document from job {job_id}: {e}")
            
            logger.info(f"Completed processing job {job_id}: {len(stored_doc_ids)}/{result.documents_found} documents stored")
        else:
            logger.error(f"Processing failed for job {job_id}: {result.errors}")
        
    except Exception as e:
        logger.error(f"Error in background processing job {job_id}: {e}")
        # TODO: Update job status with error in database