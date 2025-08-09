"""
Error handling middleware
"""
import time
from typing import Any
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import LoggerMixin
from app.core.error_handlers import error_handler
from app.core.monitoring import system_monitor


class ErrorHandlingMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware for centralized error handling"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle all unhandled exceptions"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record successful request metrics
            processing_time = (time.time() - start_time) * 1000
            system_monitor.record_request_metrics(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                duration_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            # Handle the error using centralized error handler
            processing_time = (time.time() - start_time) * 1000
            
            # Record error metrics
            system_monitor.record_request_metrics(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,  # Default to 500 for unhandled errors
                duration_ms=processing_time
            )
            
            # Create error response
            error_response = await error_handler.handle_application_error(request, e)
            
            return error_response


class ResponseTimeMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware to track and enforce response time limits"""
    
    def __init__(self, app, max_response_time_seconds: float = 30.0):
        super().__init__(app)
        self.max_response_time_seconds = max_response_time_seconds
    
    async def dispatch(self, request: Request, call_next):
        """Track response times and warn on slow responses"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            processing_time = time.time() - start_time
            
            # Add processing time header
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
            
            # Log slow responses
            if processing_time > self.max_response_time_seconds:
                self.logger.warning(
                    f"Slow response detected: {processing_time:.3f}s for {request.method} {request.url.path}"
                )
                
                # Record slow response metric
                system_monitor.metrics_collector.increment_counter(
                    "slow_responses_total",
                    1.0,
                    {
                        "endpoint": request.url.path,
                        "method": request.method
                    }
                )
            
            # Record response time histogram
            system_monitor.metrics_collector.record_histogram(
                "response_time_seconds",
                processing_time,
                {
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status_code": str(response.status_code)
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log error with processing time
            self.logger.error(
                f"Request failed after {processing_time:.3f}s: {str(e)} "
                f"for {request.method} {request.url.path}"
            )
            
            raise


class HealthCheckMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware to perform basic health checks on requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Perform health checks before processing requests"""
        
        # Check if system is healthy enough to process requests
        system_status = system_monitor.get_system_status()
        health_status = system_status.get("health", {}).get("status", "unknown")
        
        # If system is unhealthy, return service unavailable for non-health endpoints
        if health_status == "unhealthy" and request.url.path not in ["/health", "/metrics"]:
            self.logger.warning(f"Rejecting request to {request.url.path} due to unhealthy system status")
            
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ServiceUnavailable",
                    "message": "System is currently unhealthy and cannot process requests",
                    "timestamp": datetime.utcnow().isoformat(),
                    "health_status": health_status
                }
            )
        
        # Process request normally
        return await call_next(request)