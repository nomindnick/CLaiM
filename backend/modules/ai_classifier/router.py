"""FastAPI router for document classification endpoints."""

from typing import Dict, Any
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from .models import ClassificationRequest, ClassificationResult, ModelStatus
from .classifier import document_classifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classifier", tags=["classification"])


@router.post("/classify", response_model=ClassificationResult)
async def classify_document(request: ClassificationRequest) -> ClassificationResult:
    """Classify document text and return the predicted document type.
    
    Args:
        request: Classification request with text and options
        
    Returns:
        Classification result with type, confidence, and features
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Document text cannot be empty")
        
        result = document_classifier.classify(request)
        
        logger.info(
            f"Document classified as {result.document_type.value} "
            f"with confidence {result.confidence:.3f}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify-file", response_model=ClassificationResult)
async def classify_file(
    file: UploadFile = File(...),
    require_reasoning: bool = False,
    min_confidence: float = 0.3
) -> ClassificationResult:
    """Classify an uploaded text file.
    
    Args:
        file: Text file to classify
        require_reasoning: Include explanation in result
        min_confidence: Minimum confidence for classification
        
    Returns:
        Classification result
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('text/'):
            raise HTTPException(
                status_code=400, 
                detail="Only text files are supported for classification"
            )
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File content cannot be empty")
        
        # Create classification request
        request = ClassificationRequest(
            text=text,
            title=file.filename,
            require_reasoning=require_reasoning,
            min_confidence=min_confidence
        )
        
        result = document_classifier.classify(request)
        return result
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        logger.error(f"File classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/status", response_model=ModelStatus)
async def get_model_status() -> ModelStatus:
    """Get current classification model status.
    
    Returns:
        Model status information
    """
    try:
        status = document_classifier.get_model_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/preload")
async def preload_model() -> Dict[str, Any]:
    """Preload the classification model for faster inference.
    
    Returns:
        Success status and model information
    """
    try:
        success = document_classifier.preload_model()
        
        if success:
            status = document_classifier.get_model_status()
            return {
                "success": True,
                "message": "Model preloaded successfully",
                "model_status": status.dict()
            }
        else:
            return {
                "success": False,
                "message": "Failed to preload model",
                "model_status": None
            }
    except Exception as e:
        logger.error(f"Model preload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model preload failed: {str(e)}")


@router.post("/unload")
async def unload_model() -> Dict[str, str]:
    """Unload the classification model to free memory.
    
    Returns:
        Success message
    """
    try:
        document_classifier.model_manager.unload_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        logger.error(f"Model unload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model unload failed: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for classification service.
    
    Returns:
        Service health status
    """
    try:
        is_loaded = document_classifier.model_manager.is_loaded()
        
        return {
            "status": "healthy",
            "model_loaded": is_loaded,
            "timestamp": "2025-05-30T12:00:00Z",  # Would use actual timestamp
            "service": "document_classifier"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False,
            "timestamp": "2025-05-30T12:00:00Z",
            "service": "document_classifier"
        }