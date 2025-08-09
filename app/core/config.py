"""
Configuration management using Pydantic Settings
"""
import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/llm_query_db"
    postgres_user: str = "user"
    postgres_password: str = "password"
    postgres_db: str = "llm_query_db"
    
    # API Keys
    pinecone_api_key: str
    pinecone_environment: str = "us-west1-gcp-free"
    pinecone_index_name: str = "document-embeddings"
    pinecone_host: Optional[str] = None
    gemini_api_key: str
    
    # Authentication
    bearer_token: str = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"
    jwt_secret_key: str = "your_jwt_secret_key_here"
    
    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"
    max_document_size_mb: int = 50
    embedding_dimension: int = 768
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    pdf_batch_size: int = 10
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    max_concurrent_tasks: int = 10
    max_thread_pool_workers: int = 4
    max_memory_usage_mb: int = 1024
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 3600
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_context_chunks: int = 5
    min_confidence_threshold: float = 0.5
    
    # Vector Search Settings
    similarity_threshold: float = 0.3  # Very low threshold for maximum recall
    max_search_results: int = 10
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        if not (v.startswith("postgresql://") or v.startswith("postgresql+asyncpg://")):
            raise ValueError("Database URL must be a PostgreSQL connection string")
        return v
    
    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v


# Global settings instance
settings = Settings()