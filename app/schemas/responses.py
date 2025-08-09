"""
Response schemas for API endpoints
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Response model for the main query endpoint"""
    
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions",
        example=[
            "A grace period of thirty days is provided for premium payment after the due date.",
            "There is a waiting period of thirty-six (36) months of continuous coverage."
        ]
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
                ]
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request format",
                "details": {"field": "documents", "issue": "Invalid URL format"},
                "timestamp": "2025-01-08T12:00:00Z"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="Overall service status (healthy/degraded/unhealthy/error)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    database: bool = Field(..., description="Database connection status")
    vector_store: Optional[bool] = Field(default=None, description="Vector store connection status")
    llm_service: Optional[bool] = Field(default=None, description="LLM service status")
    embedding_service: Optional[bool] = Field(default=None, description="Embedding service status")
    circuit_breakers: Optional[Dict[str, Any]] = Field(default=None, description="Circuit breaker status")
    error: Optional[str] = Field(default=None, description="Error message if status is error")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-08T12:00:00Z",
                "version": "1.0.0",
                "database": True,
                "vector_store": True,
                "llm_service": True,
                "embedding_service": True,
                "circuit_breakers": {
                    "pinecone": {"state": "closed", "failure_count": 0},
                    "gemini": {"state": "closed", "failure_count": 0}
                }
            }
        }
    }