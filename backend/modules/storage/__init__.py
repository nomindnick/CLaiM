"""Storage module for CLaiM - handles document persistence and retrieval.

This module provides:
- SQLite for document metadata and full-text search
- Qdrant for vector embeddings and semantic search
- DuckDB for graph relationships (future)
- File system storage for PDFs and extracted pages
"""

from .models import (
    StoredDocument,
    StoredPage,
    DocumentMetadata,
    SearchResult,
    StorageStats,
)
from .sqlite_handler import SQLiteHandler

# Note: StorageManager import is conditional to avoid circular dependencies
# Import it directly when needed: from backend.modules.storage.storage_manager import StorageManager

__all__ = [
    "SQLiteHandler",
    "StoredDocument",
    "StoredPage",
    "DocumentMetadata",
    "SearchResult",
    "StorageStats",
]