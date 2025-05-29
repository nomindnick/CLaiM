"""FastAPI router for document processing endpoints."""

import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger

from backend.api.config import settings
from backend.shared.exceptions import DocumentProcessingError
from .models import ProcessingResult, PDFProcessingRequest
from .pdf_splitter import PDFSplitter


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

# Initialize PDF splitter
pdf_splitter = PDFSplitter()


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
    
    # Check file size
    if file.size and file.size > settings.max_upload_size:
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
            "file_id": file_id,
            "job_id": job_id,
            "filename": file.filename,
            "size": file.size,
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
        
        # TODO: Save result to database
        # TODO: Update job status
        
        logger.info(f"Completed processing job {job_id}: {result.documents_found} documents")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        # TODO: Update job status with error