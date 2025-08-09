"""
Custom exceptions for the application
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException


class BaseAppException(Exception):
    """Base exception class for application-specific errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(BaseAppException):
    """Raised when document processing fails"""
    pass


class EmbeddingGenerationError(BaseAppException):
    """Raised when embedding generation fails"""
    pass


class VectorStoreError(BaseAppException):
    """Raised when vector store operations fail"""
    pass


class LLMServiceError(BaseAppException):
    """Raised when LLM service operations fail"""
    pass


class ResponseGenerationError(BaseAppException):
    """Raised when response generation fails"""
    pass


class AuthenticationError(BaseAppException):
    """Raised when authentication fails"""
    pass


class ValidationError(BaseAppException):
    """Raised when data validation fails"""
    pass


class ConfigurationError(BaseAppException):
    """Raised when configuration is invalid"""
    pass


# HTTP Exception mappings
def create_http_exception(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create an HTTPException with structured error response"""
    
    error_detail = {
        "error": message,
        "details": details or {},
        "timestamp": "2025-01-08T00:00:00Z"  # Will be updated in middleware
    }
    
    return HTTPException(status_code=status_code, detail=error_detail)


# Common HTTP exceptions
class HTTPExceptions:
    """Common HTTP exceptions used throughout the application"""
    
    @staticmethod
    def unauthorized(message: str = "Authentication failed") -> HTTPException:
        return create_http_exception(401, message)
    
    @staticmethod
    def forbidden(message: str = "Access forbidden") -> HTTPException:
        return create_http_exception(403, message)
    
    @staticmethod
    def not_found(message: str = "Resource not found") -> HTTPException:
        return create_http_exception(404, message)
    
    @staticmethod
    def unprocessable_entity(
        message: str = "Validation error",
        details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        return create_http_exception(422, message, details)
    
    @staticmethod
    def internal_server_error(
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        return create_http_exception(500, message, details)
    
    @staticmethod
    def gateway_timeout(message: str = "Request timeout") -> HTTPException:
        return create_http_exception(504, message)