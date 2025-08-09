"""
Centralized error handling utilities
"""
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import LoggerMixin
from app.core.exceptions import BaseAppException
from app.core.monitoring import system_monitor


class ErrorHandler(LoggerMixin):
    """Centralized error handling"""
    
    def __init__(self):
        self.error_counts = {}
    
    async def handle_application_error(
        self, 
        request: Request, 
        error: Exception
    ) -> JSONResponse:
        """Handle application errors with proper logging and response formatting"""
        
        error_id = self._generate_error_id()
        error_type = type(error).__name__
        
        # Log the error with context
        self.logger.error(
            f"Application error [{error_id}]: {error_type} - {str(error)}",
            extra={
                "error_id": error_id,
                "error_type": error_type,
                "request_path": request.url.path,
                "request_method": request.method,
                "client_ip": request.client.host if request.client else "unknown",
                "traceback": traceback.format_exc()
            }
        )
        
        # Record error metrics
        system_monitor.metrics_collector.increment_counter(
            "application_errors_total",
            1.0,
            {"error_type": error_type, "endpoint": request.url.path}
        )
        
        # Determine response based on error type
        if isinstance(error, BaseAppException):
            return self._create_app_error_response(error, error_id)
        elif isinstance(error, (HTTPException, StarletteHTTPException)):
            return self._create_http_error_response(error, error_id)
        else:
            return self._create_generic_error_response(error, error_id)
    
    def _create_app_error_response(
        self, 
        error: BaseAppException, 
        error_id: str
    ) -> JSONResponse:
        """Create response for application-specific errors"""
        
        status_code_map = {
            "DocumentProcessingError": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "EmbeddingGenerationError": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "VectorStoreError": status.HTTP_503_SERVICE_UNAVAILABLE,
            "LLMServiceError": status.HTTP_503_SERVICE_UNAVAILABLE,
            "AuthenticationError": status.HTTP_401_UNAUTHORIZED,
            "ValidationError": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ConfigurationError": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        
        error_type = type(error).__name__
        status_code = status_code_map.get(error_type, status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_type,
                "message": error.message,
                "details": error.details,
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _create_http_error_response(
        self, 
        error: Union[HTTPException, StarletteHTTPException], 
        error_id: str
    ) -> JSONResponse:
        """Create response for HTTP errors"""
        
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": "HTTPError",
                "message": str(error.detail),
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers=getattr(error, 'headers', None)
        )
    
    def _create_generic_error_response(
        self, 
        error: Exception, 
        error_id: str
    ) -> JSONResponse:
        """Create response for generic errors"""
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        import uuid
        return str(uuid.uuid4())[:8]


class RetryHandler(LoggerMixin):
    """Retry logic for transient failures"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
    
    @asynccontextmanager
    async def retry_on_failure(self, operation_name: str):
        """Context manager for retry logic"""
        attempt = 0
        last_error = None
        
        while attempt < self.max_attempts:
            attempt += 1
            
            try:
                yield attempt
                return  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                
                if attempt >= self.max_attempts:
                    self.logger.error(
                        f"Operation {operation_name} failed after {self.max_attempts} attempts: {str(e)}"
                    )
                    raise e
                
                delay = self.base_delay * (2 ** (attempt - 1))  # Exponential backoff
                self.logger.warning(
                    f"Operation {operation_name} failed on attempt {attempt}, retrying in {delay}s: {str(e)}"
                )
                
                import asyncio
                await asyncio.sleep(delay)


class TimeoutHandler(LoggerMixin):
    """Timeout handling for long-running operations"""
    
    @asynccontextmanager
    async def timeout_after(self, seconds: float, operation_name: str):
        """Context manager for operation timeout"""
        import asyncio
        
        try:
            async with asyncio.timeout(seconds):
                yield
                
        except asyncio.TimeoutError:
            self.logger.error(f"Operation {operation_name} timed out after {seconds} seconds")
            
            # Record timeout metric
            system_monitor.metrics_collector.increment_counter(
                "operation_timeouts_total",
                1.0,
                {"operation": operation_name}
            )
            
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Operation {operation_name} timed out"
            )


# Global error handler instances
error_handler = ErrorHandler()
retry_handler = RetryHandler()
timeout_handler = TimeoutHandler()