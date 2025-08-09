"""
Unit tests for authentication middleware
"""
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.responses import JSONResponse

from app.middleware.auth import (
    RateLimiter, BearerTokenAuth, AuthenticationMiddleware, 
    RequestLoggingMiddleware, get_current_user
)


class TestRateLimiter:
    """Test RateLimiter functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter instance"""
        return RateLimiter(max_requests=5, window_seconds=60)
    
    def test_rate_limiter_allows_requests_under_limit(self, rate_limiter):
        """Test that requests under limit are allowed"""
        identifier = "test_client"
        
        # Should allow requests under the limit
        for i in range(5):
            assert rate_limiter.is_allowed(identifier) == True
    
    def test_rate_limiter_blocks_requests_over_limit(self, rate_limiter):
        """Test that requests over limit are blocked"""
        identifier = "test_client"
        
        # Fill up the limit
        for i in range(5):
            rate_limiter.is_allowed(identifier)
        
        # Next request should be blocked
        assert rate_limiter.is_allowed(identifier) == False
    
    def test_rate_limiter_different_identifiers(self, rate_limiter):
        """Test that different identifiers have separate limits"""
        client1 = "client1"
        client2 = "client2"
        
        # Fill up limit for client1
        for i in range(5):
            rate_limiter.is_allowed(client1)
        
        # client1 should be blocked
        assert rate_limiter.is_allowed(client1) == False
        
        # client2 should still be allowed
        assert rate_limiter.is_allowed(client2) == True
    
    def test_rate_limiter_window_cleanup(self, rate_limiter):
        """Test that old requests are cleaned up"""
        identifier = "test_client"
        
        # Mock time to simulate old requests
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Fill up the limit
            for i in range(5):
                rate_limiter.is_allowed(identifier)
            
            # Should be blocked
            assert rate_limiter.is_allowed(identifier) == False
            
            # Move time forward beyond window
            mock_time.return_value = 70  # 70 seconds later
            
            # Should be allowed again
            assert rate_limiter.is_allowed(identifier) == True
    
    def test_get_reset_time(self, rate_limiter):
        """Test reset time calculation"""
        identifier = "test_client"
        
        with patch('time.time') as mock_time:
            mock_time.return_value = 100
            
            # Make a request
            rate_limiter.is_allowed(identifier)
            
            # Reset time should be current time + window
            reset_time = rate_limiter.get_reset_time(identifier)
            assert reset_time == 160  # 100 + 60


class TestBearerTokenAuth:
    """Test BearerTokenAuth functionality"""
    
    @pytest.fixture
    def bearer_auth(self):
        """Create BearerTokenAuth instance"""
        with patch('app.middleware.auth.settings') as mock_settings:
            mock_settings.bearer_token = "valid_token_123"
            return BearerTokenAuth()
    
    def test_verify_token_valid(self, bearer_auth):
        """Test token verification with valid token"""
        assert bearer_auth.verify_token("valid_token_123") == True
    
    def test_verify_token_invalid(self, bearer_auth):
        """Test token verification with invalid token"""
        assert bearer_auth.verify_token("invalid_token") == False
        assert bearer_auth.verify_token("") == False
        assert bearer_auth.verify_token("wrong_token_456") == False
    
    @pytest.mark.asyncio
    async def test_call_with_valid_credentials(self, bearer_auth):
        """Test __call__ method with valid credentials"""
        # Mock request with valid authorization header
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer valid_token_123"}
        
        with patch.object(bearer_auth, 'verify_token', return_value=True):
            with patch('fastapi.security.HTTPBearer.__call__') as mock_parent_call:
                mock_credentials = HTTPAuthorizationCredentials(
                    scheme="Bearer", 
                    credentials="valid_token_123"
                )
                mock_parent_call.return_value = mock_credentials
                
                result = await bearer_auth(mock_request)
                
                assert result == mock_credentials
    
    @pytest.mark.asyncio
    async def test_call_with_invalid_credentials(self, bearer_auth):
        """Test __call__ method with invalid credentials"""
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer invalid_token"}
        
        with patch('fastapi.security.HTTPBearer.__call__') as mock_parent_call:
            mock_credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", 
                credentials="invalid_token"
            )
            mock_parent_call.return_value = mock_credentials
            
            with pytest.raises(HTTPException) as exc_info:
                await bearer_auth(mock_request)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid authentication token" in exc_info.value.detail


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware functionality"""
    
    @pytest.fixture
    def auth_middleware(self):
        """Create AuthenticationMiddleware instance"""
        mock_app = MagicMock()
        return AuthenticationMiddleware(mock_app, rate_limit_requests=10, rate_limit_window=60)
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.url.path = "/hackrx/run"
        request.method = "POST"
        request.client.host = "127.0.0.1"
        request.headers = {
            "Authorization": "Bearer valid_token_123",
            "User-Agent": "TestClient/1.0"
        }
        request.state = MagicMock()
        return request
    
    def test_is_public_path(self, auth_middleware):
        """Test public path detection"""
        assert auth_middleware._is_public_path("/docs") == True
        assert auth_middleware._is_public_path("/redoc") == True
        assert auth_middleware._is_public_path("/health") == True
        assert auth_middleware._is_public_path("/openapi.json") == True
        assert auth_middleware._is_public_path("/") == True
        assert auth_middleware._is_public_path("/hackrx/run") == False
        assert auth_middleware._is_public_path("/api/v1/query") == False
    
    def test_get_client_identifier(self, auth_middleware, mock_request):
        """Test client identifier generation"""
        identifier = auth_middleware._get_client_identifier(mock_request)
        
        assert isinstance(identifier, str)
        assert "127.0.0.1" in identifier
        assert ":" in identifier  # Should include user agent hash
    
    def test_get_client_identifier_with_forwarded_for(self, auth_middleware, mock_request):
        """Test client identifier with X-Forwarded-For header"""
        mock_request.headers["X-Forwarded-For"] = "192.168.1.1, 10.0.0.1"
        
        identifier = auth_middleware._get_client_identifier(mock_request)
        
        assert "192.168.1.1" in identifier  # Should use first IP
    
    @pytest.mark.asyncio
    async def test_authenticate_request_valid_token(self, auth_middleware):
        """Test request authentication with valid token"""
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer valid_token_123"}
        
        with patch('app.middleware.auth.settings') as mock_settings:
            mock_settings.bearer_token = "valid_token_123"
            
            result = await auth_middleware._authenticate_request(mock_request)
            
            assert result["authenticated"] == True
            assert result["error"] is None
            assert result["method"] == "bearer_token"
    
    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_token(self, auth_middleware):
        """Test request authentication with invalid token"""
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('app.middleware.auth.settings') as mock_settings:
            mock_settings.bearer_token = "valid_token_123"
            
            result = await auth_middleware._authenticate_request(mock_request)
            
            assert result["authenticated"] == False
            assert "Invalid bearer token" in result["error"]
            assert result["method"] is None
    
    @pytest.mark.asyncio
    async def test_authenticate_request_missing_header(self, auth_middleware):
        """Test request authentication with missing Authorization header"""
        mock_request = MagicMock()
        mock_request.headers = {}
        
        result = await auth_middleware._authenticate_request(mock_request)
        
        assert result["authenticated"] == False
        assert "Missing Authorization header" in result["error"]
    
    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_format(self, auth_middleware):
        """Test request authentication with invalid header format"""
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        
        result = await auth_middleware._authenticate_request(mock_request)
        
        assert result["authenticated"] == False
        assert "Invalid Authorization header format" in result["error"]
    
    def test_create_rate_limit_response(self, auth_middleware):
        """Test rate limit response creation"""
        client_id = "test_client"
        
        with patch.object(auth_middleware.rate_limiter, 'get_reset_time', return_value=1234567890):
            with patch('time.time', return_value=1234567800):  # 90 seconds before reset
                response = auth_middleware._create_rate_limit_response(client_id)
                
                assert isinstance(response, JSONResponse)
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                assert "RateLimitExceeded" in str(response.body)
                assert "X-RateLimit-Limit" in response.headers
                assert "Retry-After" in response.headers
    
    def test_create_auth_error_response(self, auth_middleware):
        """Test authentication error response creation"""
        error_message = "Invalid token"
        
        response = auth_middleware._create_auth_error_response(error_message)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "AuthenticationError" in str(response.body)
        assert "WWW-Authenticate" in response.headers
    
    def test_add_response_headers(self, auth_middleware):
        """Test response header addition"""
        mock_response = MagicMock()
        mock_response.headers = {}
        start_time = time.time() - 0.5  # 500ms ago
        
        result = auth_middleware._add_response_headers(mock_response, start_time)
        
        assert "X-Processing-Time" in result.headers
        assert "X-API-Version" in result.headers
        assert "X-Timestamp" in result.headers
        assert result.headers["X-API-Version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_dispatch_public_path(self, auth_middleware):
        """Test middleware dispatch for public paths"""
        mock_request = MagicMock()
        mock_request.url.path = "/docs"
        
        mock_call_next = AsyncMock()
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        result = await auth_middleware.dispatch(mock_request, mock_call_next)
        
        # Should call next without authentication
        mock_call_next.assert_called_once_with(mock_request)
        assert result == mock_response
    
    @pytest.mark.asyncio
    async def test_dispatch_authenticated_request(self, auth_middleware, mock_request):
        """Test middleware dispatch for authenticated request"""
        with patch('app.middleware.auth.settings') as mock_settings:
            mock_settings.bearer_token = "valid_token_123"
            
            mock_call_next = AsyncMock()
            mock_response = MagicMock()
            mock_response.headers = {}
            mock_call_next.return_value = mock_response
            
            with patch.object(auth_middleware, '_log_request', new_callable=AsyncMock):
                with patch.object(auth_middleware, '_log_response', new_callable=AsyncMock):
                    result = await auth_middleware.dispatch(mock_request, mock_call_next)
                    
                    # Should set authentication state
                    assert mock_request.state.authenticated == True
                    assert mock_request.state.auth_method == "bearer_token"
                    
                    # Should call next
                    mock_call_next.assert_called_once_with(mock_request)
    
    @pytest.mark.asyncio
    async def test_dispatch_rate_limited(self, auth_middleware, mock_request):
        """Test middleware dispatch when rate limited"""
        # Fill up rate limit
        client_id = auth_middleware._get_client_identifier(mock_request)
        for _ in range(10):  # Fill the limit
            auth_middleware.rate_limiter.is_allowed(client_id)
        
        mock_call_next = AsyncMock()
        
        result = await auth_middleware.dispatch(mock_request, mock_call_next)
        
        # Should return rate limit response without calling next
        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        mock_call_next.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_dispatch_authentication_failure(self, auth_middleware, mock_request):
        """Test middleware dispatch with authentication failure"""
        mock_request.headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('app.middleware.auth.settings') as mock_settings:
            mock_settings.bearer_token = "valid_token_123"
            
            mock_call_next = AsyncMock()
            
            with patch.object(auth_middleware, '_log_request', new_callable=AsyncMock):
                result = await auth_middleware.dispatch(mock_request, mock_call_next)
                
                # Should return auth error response
                assert isinstance(result, JSONResponse)
                assert result.status_code == status.HTTP_401_UNAUTHORIZED
                mock_call_next.assert_not_called()


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware functionality"""
    
    @pytest.fixture
    def logging_middleware(self):
        """Create RequestLoggingMiddleware instance"""
        mock_app = MagicMock()
        return RequestLoggingMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_dispatch_successful_request(self, logging_middleware):
        """Test middleware dispatch for successful request"""
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/hackrx/run"
        mock_request.client.host = "127.0.0.1"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_call_next = AsyncMock(return_value=mock_response)
        
        with patch.object(logging_middleware, 'logger') as mock_logger:
            result = await logging_middleware.dispatch(mock_request, mock_call_next)
            
            assert result == mock_response
            assert mock_logger.info.call_count >= 2  # Request and response logs
            mock_call_next.assert_called_once_with(mock_request)
    
    @pytest.mark.asyncio
    async def test_dispatch_failed_request(self, logging_middleware):
        """Test middleware dispatch for failed request"""
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/hackrx/run"
        mock_request.client.host = "127.0.0.1"
        
        mock_call_next = AsyncMock(side_effect=Exception("Request failed"))
        
        with patch.object(logging_middleware, 'logger') as mock_logger:
            with pytest.raises(Exception) as exc_info:
                await logging_middleware.dispatch(mock_request, mock_call_next)
            
            assert "Request failed" in str(exc_info.value)
            mock_logger.error.assert_called_once()


class TestGetCurrentUser:
    """Test get_current_user dependency"""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self):
        """Test get_current_user function"""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test_token_123"
        )
        
        user_info = await get_current_user(credentials)
        
        assert user_info["authenticated"] == True
        assert user_info["token"] == "test_token_123"
        assert user_info["auth_method"] == "bearer_token"