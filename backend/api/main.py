"""Main FastAPI application entry point."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .config import settings
from .privacy_manager import privacy_manager


# Configure logging
logger.add(
    "logs/claim_{time}.log",
    rotation="1 day",
    retention="7 days",
    level=settings.log_level,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Privacy mode: {privacy_manager.get_mode().value}")
    
    # Initialize resources here
    # TODO: Initialize model manager, database connections, etc.
    
    yield
    
    # Cleanup resources here
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Construction Litigation AI Manager - Transform construction documents into actionable intelligence",
    version=settings.app_version,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.debug(f"{request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.debug(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    
    return response


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "privacy_mode": privacy_manager.get_privacy_info(),
    }


# Root API endpoint
@app.get(settings.api_prefix)
async def api_root() -> Dict[str, Any]:
    """API root endpoint with basic information."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
        "health": "/health",
    }


# Privacy endpoint
@app.get(f"{settings.api_prefix}/privacy")
async def get_privacy_status() -> Dict[str, Any]:
    """Get current privacy mode and settings."""
    return privacy_manager.get_privacy_info()


@app.put(f"{settings.api_prefix}/privacy/mode")
async def update_privacy_mode(mode: str) -> Dict[str, Any]:
    """Update privacy mode."""
    try:
        from .config import PrivacyMode
        privacy_mode = PrivacyMode(mode)
        privacy_manager.set_mode(privacy_mode)
        logger.info(f"Privacy mode updated to: {mode}")
        return {"status": "success", "mode": mode}
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid privacy mode: {mode}"},
        )


# TODO: Import and include routers from modules
# from backend.modules.document_processor.router import router as document_router
# app.include_router(document_router, prefix=f"{settings.api_prefix}/documents")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )