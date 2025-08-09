"""
Unit tests for Pydantic schemas
"""
import pytest
from pydantic import ValidationError
from datetime import datetime

from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse, ErrorResponse, HealthResponse
from app.schemas.models import Document, TextChunk, DocumentType, DocumentStatus


class TestQueryRequest:
    """Test QueryRequest validation"""
    
    def test_valid_request(self):
        """Test valid request creation"""
        request = QueryRequest(
            documents="https://example.com/document.pdf",
            questions=["What is the policy coverage?", "What are the terms?"]
        )
        assert request.documents == "https://example.com/document.pdf"
        assert len(request.questions) == 2
    
    def test_invalid_url(self):
        """Test invalid URL validation"""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="invalid-url",
                questions=["What is the policy coverage?"]
            )
        assert "Document URL must be a valid HTTP/HTTPS URL" in str(exc_info.value)
    
    def test_empty_questions(self):
        """Test empty questions validation"""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=[]
            )
        assert "List should have at least 1 item" in str(exc_info.value)
    
    def test_empty_question_string(self):
        """Test empty question string validation"""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=["", "Valid question"]
            )
        assert "Questions cannot be empty" in str(exc_info.value)
    
    def test_too_long_question(self):
        """Test question length validation"""
        long_question = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                documents="https://example.com/document.pdf",
                questions=[long_question]
            )
        assert "Questions must be less than 500 characters" in str(exc_info.value)


class TestQueryResponse:
    """Test QueryResponse model"""
    
    def test_valid_response(self):
        """Test valid response creation"""
        response = QueryResponse(
            answers=["Answer 1", "Answer 2"]
        )
        assert len(response.answers) == 2
        assert response.answers[0] == "Answer 1"


class TestErrorResponse:
    """Test ErrorResponse model"""
    
    def test_error_response_creation(self):
        """Test error response creation"""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            details={"field": "documents"}
        )
        assert error.error == "ValidationError"
        assert error.message == "Invalid input"
        assert error.details["field"] == "documents"
        assert isinstance(error.timestamp, datetime)


class TestDocument:
    """Test Document model"""
    
    def test_document_creation(self):
        """Test document creation with defaults"""
        doc = Document(url="https://example.com/test.pdf")
        assert doc.url == "https://example.com/test.pdf"
        assert doc.status == DocumentStatus.PENDING
        assert doc.document_type == DocumentType.PDF
        assert isinstance(doc.created_at, datetime)
        assert len(doc.text_chunks) == 0
    
    def test_document_type_detection_pdf(self):
        """Test PDF document type detection"""
        doc = Document(url="https://example.com/document.pdf")
        assert doc.document_type == DocumentType.PDF
    
    def test_document_type_detection_docx(self):
        """Test DOCX document type detection"""
        doc = Document(url="https://example.com/document.docx")
        assert doc.document_type == DocumentType.DOCX
    
    def test_document_type_detection_email(self):
        """Test email document type detection"""
        doc = Document(url="https://example.com/message.eml")
        assert doc.document_type == DocumentType.EMAIL


class TestTextChunk:
    """Test TextChunk model"""
    
    def test_text_chunk_creation(self):
        """Test text chunk creation"""
        chunk = TextChunk(
            document_id="doc-123",
            content="This is a test chunk",
            chunk_index=0
        )
        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is a test chunk"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
    
    def test_empty_content_validation(self):
        """Test empty content validation"""
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                document_id="doc-123",
                content="   ",
                chunk_index=0
            )
        assert "Chunk content cannot be empty" in str(exc_info.value)
    
    def test_content_stripping(self):
        """Test content whitespace stripping"""
        chunk = TextChunk(
            document_id="doc-123",
            content="  This is a test chunk  ",
            chunk_index=0
        )
        assert chunk.content == "This is a test chunk"