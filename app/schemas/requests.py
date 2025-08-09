"""
Request schemas for API endpoints
"""
from typing import List
from pydantic import BaseModel, HttpUrl, field_validator, Field


class QueryRequest(BaseModel):
    """Request model for the main query endpoint"""
    
    documents: str = Field(
        ...,
        description="Blob URL of the document to process",
        example="https://example.blob.core.windows.net/assets/policy.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of questions to ask about the document"
    )
    
    @field_validator("documents")
    @classmethod
    def validate_document_url(cls, v):
        """Validate that the document URL is properly formatted"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Document URL must be a valid HTTP/HTTPS URL")
        return v
    
    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v):
        """Validate questions list"""
        if not v:
            raise ValueError("At least one question is required")
        
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
            if len(question) > 500:
                raise ValueError("Questions must be less than 500 characters")
        
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?"
                ]
            }
        }
    }