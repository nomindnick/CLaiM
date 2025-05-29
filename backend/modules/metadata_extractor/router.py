"""FastAPI router for metadata extraction endpoints."""

from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from loguru import logger

from modules.document_processor.models import Document, DocumentPage, DocumentType
from modules.document_processor.pdf_splitter import PDFSplitter
from .extractor import MetadataExtractor

router = APIRouter(tags=["metadata"])


@router.post("/extract")
async def extract_metadata(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """Extract metadata from an uploaded PDF document.
    
    Args:
        file: PDF file to extract metadata from
        
    Returns:
        Extracted metadata
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{uuid4()}_{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)
        
        # Process PDF
        pdf_splitter = PDFSplitter()
        result = pdf_splitter.process_pdf(PDFProcessingRequest(file_path=temp_path))
        
        if not result.success or not result.documents:
            raise HTTPException(status_code=400, detail="Failed to process PDF")
        
        # Extract metadata from each document
        extractor = MetadataExtractor()
        extracted_metadata = []
        
        for doc in result.documents:
            metadata = extractor.extract_metadata(doc)
            extracted_metadata.append({
                "document_id": doc.id,
                "type": doc.type.value,
                "page_range": doc.page_range,
                "metadata": {
                    "title": metadata.title,
                    "date": metadata.date.isoformat() if metadata.date else None,
                    "parties": [
                        {
                            "name": p.name,
                            "role": p.role,
                            "email": p.email,
                            "phone": p.phone,
                            "company": p.company
                        } for p in metadata.parties
                    ],
                    "reference_numbers": metadata.reference_numbers,
                    "amounts": metadata.amounts,
                    "keywords": metadata.keywords,
                    "project_number": metadata.project_number,
                    "project_name": metadata.project_name,
                    "subject": metadata.subject
                }
            })
        
        # Clean up
        if background_tasks:
            background_tasks.add_task(temp_path.unlink, missing_ok=True)
        else:
            temp_path.unlink(missing_ok=True)
        
        return {
            "success": True,
            "source_file": file.filename,
            "documents_found": len(extracted_metadata),
            "extracted_metadata": extracted_metadata
        }
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/text")
async def extract_metadata_from_text(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract metadata from provided text.
    
    Args:
        request: Dictionary with 'text' and optional 'document_type'
        
    Returns:
        Extracted metadata
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Create a mock document
    doc = Document(
        id=str(uuid4()),
        source_pdf_id="text-input",
        source_pdf_path=Path("/tmp/text-input.txt"),
        type=DocumentType(request.get("document_type", "unknown")),
        pages=[DocumentPage(page_number=1, text=text)],
        page_range=(1, 1),
        text=text
    )
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(doc)
    
    return {
        "success": True,
        "metadata": {
            "title": metadata.title,
            "date": metadata.date.isoformat() if metadata.date else None,
            "parties": [
                {
                    "name": p.name,
                    "role": p.role,
                    "email": p.email,
                    "phone": p.phone,
                    "company": p.company
                } for p in metadata.parties
            ],
            "reference_numbers": metadata.reference_numbers,
            "amounts": metadata.amounts,
            "keywords": metadata.keywords,
            "project_number": metadata.project_number,
            "project_name": metadata.project_name,
            "subject": metadata.subject
        }
    }


@router.post("/normalize/parties")
async def normalize_parties(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize and deduplicate party names.
    
    Args:
        request: Dictionary with 'parties' list
        
    Returns:
        Normalized and grouped parties
    """
    parties = request.get("parties", [])
    if not parties:
        raise HTTPException(status_code=400, detail="No parties provided")
    
    from .normalizer import EntityNormalizer
    normalizer = EntityNormalizer()
    
    # Normalize each party
    normalized = []
    for party in parties:
        if isinstance(party, str):
            normalized.append({
                "original": party,
                "normalized": normalizer.normalize_party_name(party)
            })
        elif isinstance(party, dict) and "name" in party:
            normalized.append({
                "original": party["name"],
                "normalized": normalizer.normalize_party_name(party["name"]),
                "metadata": {k: v for k, v in party.items() if k != "name"}
            })
    
    # Find similar parties
    all_names = [n["original"] for n in normalized]
    similar_groups = normalizer.find_similar_parties(all_names)
    
    return {
        "success": True,
        "normalized_parties": normalized,
        "similar_groups": [
            {
                "canonical": canonical,
                "variations": variations
            }
            for canonical, variations in similar_groups.items()
        ]
    }


# Import required model
from modules.document_processor.models import PDFProcessingRequest