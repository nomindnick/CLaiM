"""Global configuration for CLaiM application."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class PrivacyMode(str, Enum):
    """Privacy mode settings for AI operations."""
    
    FULL_LOCAL = "full_local"
    HYBRID_SAFE = "hybrid_safe"
    FULL_FEATURED = "full_featured"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "CLaiM"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    database_url: str = "sqlite:///./claim.db"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    
    # Model paths
    models_dir: Path = Path("../models")
    distilbert_model: str = "distilbert-legal.gguf"
    phi_model: str = "phi-3.5-mini-Q4_K_M.gguf"
    embeddings_model: str = "all-MiniLM-L6-v2.gguf"
    
    # Privacy
    default_privacy_mode: PrivacyMode = PrivacyMode.FULL_LOCAL
    
    # API Keys (optional)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # File storage
    upload_dir: Path = Path("./uploads")
    max_upload_size: int = 104857600  # 100MB
    allowed_extensions: List[str] = ["pdf", "png", "jpg", "jpeg"]
    
    # Performance
    max_workers: int = 4
    batch_size: int = 32
    cache_size: int = 1000
    
    # Security
    secret_key: str = "change-this-in-production"
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)


# Global settings instance
settings = Settings()