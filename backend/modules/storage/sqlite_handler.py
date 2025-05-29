"""SQLite handler for document storage with FTS5 full-text search."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from uuid import uuid4

from loguru import logger

from shared.exceptions import StorageError
from modules.document_processor.models import DocumentType
from .models import (
    StoredDocument,
    StoredPage,
    DocumentMetadata,
    SearchFilter,
    SearchResult,
    StorageStatus,
    StorageStats,
)


class SQLiteHandler:
    """Handles SQLite database operations for document storage."""
    
    def __init__(self, db_path: Path):
        """Initialize SQLite handler.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_pdf_id TEXT NOT NULL,
                    source_pdf_path TEXT NOT NULL,
                    type TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    title TEXT,
                    text TEXT NOT NULL,
                    summary TEXT,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT,
                    key_facts_json TEXT,
                    classification_confidence REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    storage_path TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    indexed_at TIMESTAMP,
                    UNIQUE(source_pdf_id, page_start, page_end)
                )
            """)
            
            # Pages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    is_scanned BOOLEAN DEFAULT FALSE,
                    has_tables BOOLEAN DEFAULT FALSE,
                    has_images BOOLEAN DEFAULT FALSE,
                    ocr_confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, page_number)
                )
            """)
            
            # Document relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(source_id, target_id, relationship_type)
                )
            """)
            
            # Full-text search virtual table
            # Note: We don't use content=documents because parties and reference_numbers
            # are extracted from JSON, not direct columns
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    id UNINDEXED,
                    title,
                    text,
                    parties,
                    reference_numbers,
                    tokenize='porter unicode61'
                )
            """)
            
            # Drop old problematic triggers if they exist
            conn.execute("DROP TRIGGER IF EXISTS documents_fts_insert")
            conn.execute("DROP TRIGGER IF EXISTS documents_fts_update")
            
            # Note: FTS5 triggers with json_extract have issues in some SQLite versions
            # We'll handle FTS population manually in the save_document method instead
            
            # Only create delete trigger which is simple and works reliably
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_delete
                AFTER DELETE ON documents BEGIN
                    DELETE FROM documents_fts WHERE id = old.id;
                END
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_pdf ON documents(source_pdf_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON document_relationships(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON document_relationships(target_id)")
            
            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")
    
    def save_document(self, document: StoredDocument, pages: Optional[List[StoredPage]] = None) -> str:
        """Save document to database.
        
        Args:
            document: Document to save
            pages: Optional list of pages to save
            
        Returns:
            Document ID
        """
        try:
            with self._get_connection() as conn:
                # Ensure document has an ID
                if not document.id:
                    document.id = str(uuid4())
                
                # Convert complex fields to JSON
                metadata_json = document.metadata.model_dump_json()
                embedding_json = json.dumps(document.embedding) if document.embedding else None
                key_facts_json = json.dumps(document.key_facts) if document.key_facts else None
                
                # Insert or replace document
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        id, source_pdf_id, source_pdf_path, type,
                        page_start, page_end, page_count,
                        title, text, summary, metadata_json,
                        embedding_json, key_facts_json, classification_confidence,
                        status, storage_path, created_at, updated_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id,
                    document.source_pdf_id,
                    str(document.source_pdf_path),
                    document.type.value,
                    document.page_range[0],
                    document.page_range[1],
                    document.page_count,
                    document.title,
                    document.text,
                    document.summary,
                    metadata_json,
                    embedding_json,
                    key_facts_json,
                    document.classification_confidence,
                    document.status.value,
                    str(document.storage_path) if document.storage_path else None,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                    document.indexed_at.isoformat() if document.indexed_at else None,
                ))
                
                # Save pages if provided
                if pages:
                    for page in pages:
                        if not page.id:
                            page.id = str(uuid4())
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO pages (
                                id, document_id, page_number, text,
                                is_scanned, has_tables, has_images,
                                ocr_confidence, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            page.id,
                            document.id,
                            page.page_number,
                            page.text,
                            page.is_scanned,
                            page.has_tables,
                            page.has_images,
                            page.ocr_confidence,
                            page.created_at.isoformat(),
                        ))
                
                # Save relationships
                for related_id in document.responds_to:
                    self._save_relationship(conn, document.id, related_id, "responds_to")
                
                for related_id in document.references:
                    self._save_relationship(conn, document.id, related_id, "references")
                
                for related_id in document.related_documents:
                    self._save_relationship(conn, document.id, related_id, "related")
                
                # Manually populate FTS table since triggers with json_extract are problematic
                self._populate_fts(conn, document, metadata_json)
                
                conn.commit()
                logger.info(f"Saved document {document.id} ({document.type.value})")
                return document.id
                
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise StorageError(f"Failed to save document: {e}")
    
    def _save_relationship(self, conn: sqlite3.Connection, source_id: str, target_id: str, rel_type: str) -> None:
        """Save document relationship."""
        rel_id = str(uuid4())
        try:
            conn.execute("""
                INSERT OR IGNORE INTO document_relationships (
                    id, source_id, target_id, relationship_type, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (rel_id, source_id, target_id, rel_type, datetime.utcnow().isoformat()))
        except sqlite3.IntegrityError:
            # Relationship already exists
            pass
    
    def get_document(self, document_id: str, include_pages: bool = False) -> Optional[StoredDocument]:
        """Get document by ID.
        
        Args:
            document_id: Document ID
            include_pages: Whether to include pages in result
            
        Returns:
            Document if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM documents WHERE id = ?
            """, (document_id,)).fetchone()
            
            if not row:
                return None
            
            document = self._row_to_document(row)
            
            # Load relationships
            relationships = conn.execute("""
                SELECT target_id, relationship_type
                FROM document_relationships
                WHERE source_id = ?
            """, (document_id,)).fetchall()
            
            for rel in relationships:
                if rel["relationship_type"] == "responds_to":
                    document.responds_to.append(rel["target_id"])
                elif rel["relationship_type"] == "references":
                    document.references.append(rel["target_id"])
                elif rel["relationship_type"] == "related":
                    document.related_documents.append(rel["target_id"])
            
            return document
    
    def search_documents(self, filter: SearchFilter) -> SearchResult:
        """Search documents with filters.
        
        Args:
            filter: Search filter criteria
            
        Returns:
            Search results
        """
        start_time = datetime.utcnow()
        
        with self._get_connection() as conn:
            # Build query
            query_parts = []
            params = []
            
            # Full-text search
            if filter.query:
                query_parts.append("""
                    id IN (
                        SELECT id FROM documents_fts
                        WHERE documents_fts MATCH ?
                    )
                """)
                params.append(filter.query)
            
            # Type filter
            if filter.document_types:
                placeholders = ",".join(["?" for _ in filter.document_types])
                query_parts.append(f"type IN ({placeholders})")
                params.extend([dt.value for dt in filter.document_types])
            
            # Date range filter
            if filter.date_from or filter.date_to:
                date_conditions = []
                if filter.date_from:
                    date_conditions.append("created_at >= ?")
                    params.append(filter.date_from.isoformat())
                if filter.date_to:
                    date_conditions.append("created_at <= ?")
                    params.append(filter.date_to.isoformat())
                query_parts.append(f"({' AND '.join(date_conditions)})")
            
            # Build final query
            where_clause = ""
            if query_parts:
                where_clause = f"WHERE {' AND '.join(query_parts)}"
            
            # Count total results
            count_query = f"SELECT COUNT(*) FROM documents {where_clause}"
            total_count = conn.execute(count_query, params).fetchone()[0]
            
            # Get paginated results
            order_clause = f"ORDER BY {filter.sort_by} {'DESC' if filter.sort_descending else 'ASC'}"
            limit_clause = f"LIMIT {filter.limit} OFFSET {filter.offset}"
            
            results_query = f"""
                SELECT * FROM documents
                {where_clause}
                {order_clause}
                {limit_clause}
            """
            
            rows = conn.execute(results_query, params).fetchall()
            documents = [self._row_to_document(row) for row in rows]
            
            # Calculate search time
            search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return SearchResult(
                documents=documents,
                total_count=total_count,
                offset=filter.offset,
                limit=filter.limit,
                search_time_ms=search_time_ms,
                query_used=filter.query,
                filters_applied=filter.model_dump(exclude_none=True),
            )
    
    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        """Convert database row to StoredDocument."""
        # Parse JSON fields
        metadata = DocumentMetadata.model_validate_json(row["metadata_json"])
        embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
        key_facts = json.loads(row["key_facts_json"]) if row["key_facts_json"] else []
        
        return StoredDocument(
            id=row["id"],
            source_pdf_id=row["source_pdf_id"],
            source_pdf_path=Path(row["source_pdf_path"]),
            type=DocumentType(row["type"]),
            page_range=(row["page_start"], row["page_end"]),
            page_count=row["page_count"],
            title=row["title"],
            text=row["text"],
            summary=row["summary"],
            metadata=metadata,
            embedding=embedding,
            key_facts=key_facts,
            classification_confidence=row["classification_confidence"],
            status=StorageStatus(row["status"]),
            storage_path=Path(row["storage_path"]) if row["storage_path"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
        )
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        with self._get_connection() as conn:
            stats = StorageStats()
            
            # Document counts
            stats.total_documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            stats.total_pages = conn.execute("SELECT SUM(page_count) FROM documents").fetchone()[0] or 0
            stats.total_pdfs = conn.execute("SELECT COUNT(DISTINCT source_pdf_id) FROM documents").fetchone()[0]
            
            # Documents by type
            type_counts = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM documents
                GROUP BY type
            """).fetchall()
            
            for row in type_counts:
                stats.documents_by_type[DocumentType(row["type"])] = row["count"]
            
            # OCR stats
            stats.documents_with_ocr = conn.execute("""
                SELECT COUNT(DISTINCT document_id)
                FROM pages
                WHERE is_scanned = TRUE
            """).fetchone()[0]
            
            # Embedding stats
            stats.documents_with_embeddings = conn.execute("""
                SELECT COUNT(*)
                FROM documents
                WHERE embedding_json IS NOT NULL
            """).fetchone()[0]
            
            # Average pages
            if stats.total_documents > 0:
                stats.average_pages_per_document = stats.total_pages / stats.total_documents
            
            # Date ranges
            date_range = conn.execute("""
                SELECT MIN(created_at) as earliest, MAX(created_at) as latest
                FROM documents
            """).fetchone()
            
            if date_range["earliest"]:
                stats.earliest_document = datetime.fromisoformat(date_range["earliest"])
                stats.latest_document = datetime.fromisoformat(date_range["latest"])
            
            # Last indexed
            last_indexed = conn.execute("""
                SELECT MAX(indexed_at)
                FROM documents
                WHERE indexed_at IS NOT NULL
            """).fetchone()[0]
            
            if last_indexed:
                stats.last_indexed = datetime.fromisoformat(last_indexed)
            
            # Storage sizes (approximate)
            stats.database_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            
            return stats
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from database.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def update_document_status(self, document_id: str, status: StorageStatus) -> bool:
        """Update document status.
        
        Args:
            document_id: Document ID
            status: New status
            
        Returns:
            True if updated, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE documents
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status.value, datetime.utcnow().isoformat(), document_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def _populate_fts(self, conn: sqlite3.Connection, document: StoredDocument, metadata_json: str) -> None:
        """Manually populate FTS table.
        
        This is done manually instead of via triggers because SQLite FTS5 has issues
        with json_extract in triggers in some versions.
        """
        try:
            # Extract metadata for FTS
            metadata_dict = json.loads(metadata_json)
            parties_list = metadata_dict.get('parties', [])
            ref_numbers_dict = metadata_dict.get('reference_numbers', {})
            
            # Convert lists/dicts to searchable text
            parties_text = ' '.join(parties_list) if parties_list else ''
            ref_numbers_text = ' '.join(
                f"{k} {' '.join(str(v) for v in vals)}" 
                for k, vals in ref_numbers_dict.items()
            ) if ref_numbers_dict else ''
            
            # Get the rowid of the document we just inserted
            rowid = conn.execute(
                "SELECT rowid FROM documents WHERE id = ?", 
                (document.id,)
            ).fetchone()[0]
            
            # Check if document already exists in FTS
            existing = conn.execute(
                "SELECT rowid FROM documents_fts WHERE id = ?",
                (document.id,)
            ).fetchone()
            
            if existing:
                # Update existing FTS entry
                conn.execute("""
                    UPDATE documents_fts
                    SET title = ?,
                        text = ?,
                        parties = ?,
                        reference_numbers = ?
                    WHERE id = ?
                """, (
                    document.title,
                    document.text,
                    parties_text,
                    ref_numbers_text,
                    document.id
                ))
            else:
                # Insert new FTS entry
                conn.execute("""
                    INSERT INTO documents_fts(rowid, id, title, text, parties, reference_numbers)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rowid,
                    document.id,
                    document.title,
                    document.text,
                    parties_text,
                    ref_numbers_text
                ))
                
        except Exception as e:
            logger.warning(f"Failed to populate FTS for document {document.id}: {e}")
            # Don't fail the entire save operation just because FTS failed
    
    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")