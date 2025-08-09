"""
Authentication middleware for FastAPI
"""
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.services.database import db_manager


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the identifier"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        request_times = self.requests[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if under limit
        if len(request_times) >= self.max_requests:
            return False
        
        # Add current request
        request_times.append(now)
        return True
    
    def get_reset_time(self, identifier: str) -> int:
        """Get time when rate limit resets for identifier"""
        request_times = self.requests[identifier]
        if not request_times:
            return int(time.time())
        
        return int(request_times[0] + self.window_seconds)


class BearerTokenAuth(HTTPBearer):
    """Bearer token authentication"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.valid_token = settings.bearer_token
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        
        if credentials:
            if not self.verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        return credentials
    
    def verify_token(self, token: str) -> bool:
        """Verify the bearer token"""
        return token == self.valid_token


class AuthenticationMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Authentication and rate limiting middleware"""
    
    def __init__(self, app, rate_limit_requests: int = 100, rate_limit_window: int = 3600):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.bearer_auth = BearerTokenAuth(auto_error=False)
        
        # Paths that don't require authentication
        self.public_paths = {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/",
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication and rate limiting"""
        start_time = time.time()
        
        try:
            # Skip authentication for public paths
            if self._is_public_path(request.url.path):
                response = await call_next(request)
                return self._add_response_headers(response, start_time)
            
            # Get client identifier for rate limiting
            client_id = self._get_client_identifier(request)
            
            # Check rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                return self._create_rate_limit_response(client_id)
            
            # Authenticate request
            auth_result = await self._authenticate_request(request)
            if not auth_result["authenticated"]:
                return self._create_auth_error_response(auth_result["error"])
            
            # Add authentication info to request state
            request.state.authenticated = True
            request.state.client_id = client_id
            request.state.auth_method = auth_result["method"]
            
            # Log successful authentication
            await self._log_request(request, "authenticated", start_time)
            
            # Process request
            response = await call_next(request)
            
            # Add response headers
            response = self._add_response_headers(response, start_time)
            
            # Log response
            await self._log_response(request, response, start_time)
            
            return response
            
        except HTTPException as e:
            # Log authentication failure
            await self._log_request(request, f"auth_failed: {e.detail}", start_time)
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "AuthenticationError",
                    "message": e.detail,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers=e.headers or {}
            )
        
        except Exception as e:
            # Log unexpected error
            self.logger.error(f"Authentication middleware error: {str(e)}")
            await self._log_request(request, f"middleware_error: {str(e)}", start_time)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "InternalServerError",
                    "message": "Authentication middleware error",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require authentication)"""
        return path in self.public_paths or path.startswith("/docs") or path.startswith("/redoc")
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get from X-Forwarded-For header first (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Fall back to direct client IP
            client_ip = request.client.host if request.client else "unknown"
        
        # Include user agent for more specific identification
        user_agent = request.headers.get("User-Agent", "unknown")
        
        return f"{client_ip}:{hash(user_agent) % 10000}"
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate the request"""
        try:
            # Check for Bearer token
            authorization = request.headers.get("Authorization")
            
            if not authorization:
                return {
                    "authenticated": False,
                    "error": "Missing Authorization header",
                    "method": None
                }
            
            if not authorization.startswith("Bearer "):
                return {
                    "authenticated": False,
                    "error": "Invalid Authorization header format. Expected 'Bearer <token>'",
                    "method": None
                }
            
            token = authorization[7:]  # Remove "Bearer " prefix
            
            if not token:
                return {
                    "authenticated": False,
                    "error": "Empty bearer token",
                    "method": None
                }
            
            # Verify token
            if token != settings.bearer_token:
                return {
                    "authenticated": False,
                    "error": "Invalid bearer token",
                    "method": None
                }
            
            return {
                "authenticated": True,
                "error": None,
                "method": "bearer_token"
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return {
                "authenticated": False,
                "error": f"Authentication processing error: {str(e)}",
                "method": None
            }
    
    def _create_rate_limit_response(self, client_id: str) -> JSONResponse:
        """Create rate limit exceeded response"""
        reset_time = self.rate_limiter.get_reset_time(client_id)
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "RateLimitExceeded",
                "message": f"Rate limit exceeded. Maximum {self.rate_limiter.max_requests} requests per {self.rate_limiter.window_seconds} seconds.",
                "retry_after": reset_time - int(time.time()),
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={
                "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                "X-RateLimit-Window": str(self.rate_limiter.window_seconds),
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(reset_time - int(time.time()))
            }
        )
    
    def _create_auth_error_response(self, error_message: str) -> JSONResponse:
        """Create authentication error response"""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "AuthenticationError",
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _add_response_headers(self, response, start_time: float):
        """Add standard response headers"""
        processing_time = time.time() - start_time
        
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Timestamp"] = datetime.utcnow().isoformat()
        
        return response
    
    async def _log_request(self, request: Request, event: str, start_time: float):
        """Log request details"""
        try:
            client_id = getattr(request.state, "client_id", self._get_client_identifier(request))
            
            log_data = {
                "event": event,
                "method": request.method,
                "path": request.url.path,
                "client_id": client_id,
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            
            # Log to database if available
            if hasattr(db_manager, 'log_query'):
                await db_manager.log_query({
                    "query": f"{request.method} {request.url.path}",
                    "response": event,
                    "processing_time_ms": log_data["processing_time_ms"]
                })
            
            self.logger.info(f"Request {event}: {log_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log request: {str(e)}")
    
    async def _log_response(self, request: Request, response, start_time: float):
        """Log response details"""
        try:
            processing_time = int((time.time() - start_time) * 1000)
            
            log_data = {
                "event": "response",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if response.status_code >= 400:
                self.logger.warning(f"Error response: {log_data}")
            else:
                self.logger.info(f"Successful response: {log_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log response: {str(e)}")


class RequestLoggingMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware for detailed request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response details"""
        start_time = time.time()
        
        # Log incoming request
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Response: {response.status_code} "
                f"({processing_time:.3f}s) "
                f"for {request.method} {request.url.path}"
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(
                f"Request failed: {str(e)} "
                f"({processing_time:.3f}s) "
                f"for {request.method} {request.url.path}"
            )
            
            raise


# Dependency for route-level authentication
bearer_auth = BearerTokenAuth()


async def get_current_user(request: Request) -> Dict[str, Any]:
    """Dependency to get current authenticated user info"""
    # Check if request was authenticated by middleware
    if hasattr(request.state, 'authenticated') and request.state.authenticated:
        return {
            "authenticated": True,
            "client_id": getattr(request.state, 'client_id', 'unknown'),
            "auth_method": getattr(request.state, 'auth_method', 'bearer_token')
        }
    
    # Fallback authentication check
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    if token != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "authenticated": True,
        "auth_method": "bearer_token"
    }