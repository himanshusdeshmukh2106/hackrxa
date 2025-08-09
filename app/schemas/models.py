"""
Core data models for internal use
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
import hashlib


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    EMAIL = "email"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """Document model for internal processing"""
    
    id: str = Field(default="")  # Will be set based on URL
    url: str
    content_type: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    status: DocumentStatus = DocumentStatus.PENDING
    text_chunks: List["TextChunk"] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @model_validator(mode="after")
    def determine_document_type_and_id(self):
        """Determine document type from URL or content type and generate deterministic ID"""
        # Generate deterministic ID based on URL
        if not self.id and self.url:
            # Create a deterministic ID based on URL hash
            url_hash = hashlib.sha256(self.url.encode()).hexdigest()[:32]
            # Format as UUID-like string for compatibility
            self.id = f"{url_hash[:8]}-{url_hash[8:12]}-{url_hash[12:16]}-{url_hash[16:20]}-{url_hash[20:32]}"
        
        # Determine document type
        if self.document_type != DocumentType.UNKNOWN:
            return self
        
        url = self.url.lower() if self.url else ""
        content_type = self.content_type.lower() if self.content_type else ""
        
        if url.endswith(".pdf") or "pdf" in content_type:
            self.document_type = DocumentType.PDF
        elif url.endswith((".docx", ".doc")) or "word" in content_type:
            self.document_type = DocumentType.DOCX
        elif url.endswith(".txt") or "text/plain" in content_type:
            self.document_type = DocumentType.TEXT
        elif "email" in content_type or url.endswith(".eml"):
            self.document_type = DocumentType.EMAIL
        
        return self


class TextChunk(BaseModel):
    """Text chunk model for document processing"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate chunk content"""
        if not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()


class Entity(BaseModel):
    """Named entity model"""
    
    text: str
    label: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class QueryIntent(BaseModel):
    original_query: str
    intent_type: str
    entities: List[Entity]
    confidence: float
    processed_query: str
    keywords: Optional[List[str]] = None


class SearchResult(BaseModel):
    """Search result model"""
    
    chunk_id: str
    content: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_explanation: Optional[str] = None
    source_location: Optional[str] = None


class Answer(BaseModel):
    """Answer model with explainability"""
    
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunks: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingMetrics(BaseModel):
    """Processing metrics model"""
    
    document_id: str
    processing_time_ms: int
    chunk_count: int
    embedding_time_ms: Optional[int] = None
    query_time_ms: Optional[int] = None
    llm_time_ms: Optional[int] = None
    total_tokens_used: Optional[int] = None


# Update forward references
Document.model_rebuild()
QueryIntent.model_rebuild()