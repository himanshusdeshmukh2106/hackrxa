"""
Tests for error handling and monitoring
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.core.error_handlers import ErrorHandler, RetryHandler, TimeoutHandler
from app.core.exceptions import DocumentProcessingError, VectorStoreError
from app.middleware.error_handling import ErrorHandlingMiddleware, ResponseTimeMiddleware
from app.core.monitoring import CircuitBreaker, MetricsCollector, HealthChecker


class TestErrorHandler:
    """Test error handling functionality"""
    
    def setup_method(self):
        self.error_handler = ErrorHandler()
    
    @pytest.mark.asyncio
    async def test_handle_application_error_with_app_exception(self):
        """Test handling of application-specific exceptions"""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "POST"
        request.client.host = "127.0.0.1"
        
        error = DocumentProcessingError("Failed to process document", {"file": "test.pdf"})
        
        response = await self.error_handler.handle_application_error(request, error)
        
        assert response.status_code == 422
        response_data = response.body.decode()
        assert "DocumentProcessingError" in response_data
        assert "Failed to process document" in response_data
    
    @pytest.mark.asyncio
    async def test_handle_application_error_with_http_exception(self):
        """Test handling of HTTP exceptions"""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        error = HTTPException(status_code=404, detail="Not found")
        
        response = await self.error_handler.handle_application_error(request, error)
        
        assert response.status_code == 404
        response_data = response.body.decode()
        assert "HTTPError" in response_data
        assert "Not found" in response_data
    
    @pytest.mark.asyncio
    async def test_handle_application_error_with_generic_exception(self):
        """Test handling of generic exceptions"""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        error = ValueError("Invalid value")
        
        response = await self.error_handler.handle_application_error(request, error)
        
        assert response.status_code == 500
        response_data = response.body.decode()
        assert "InternalServerError" in response_data
        assert "unexpected error occurred" in response_data


class TestRetryHandler:
    """Test retry handling functionality"""
    
    def setup_method(self):
        self.retry_handler = RetryHandler(max_attempts=3, base_delay=0.1)
    
    @pytest.mark.asyncio
    async def test_retry_success_on_first_attempt(self):
        """Test successful operation on first attempt"""
        attempt_count = 0
        
        async with self.retry_handler.retry_on_failure("test_operation") as attempt:
            attempt_count = attempt
            # Simulate successful operation
            pass
        
        assert attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful operation after initial failures"""
        attempt_count = 0
        
        async with self.retry_handler.retry_on_failure("test_operation") as attempt:
            attempt_count = attempt
            if attempt < 3:
                raise ValueError("Temporary failure")
            # Success on third attempt
        
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion"""
        with pytest.raises(ValueError, match="Permanent failure"):
            async with self.retry_handler.retry_on_failure("test_operation") as attempt:
                raise ValueError("Permanent failure")


class TestTimeoutHandler:
    """Test timeout handling functionality"""
    
    def setup_method(self):
        self.timeout_handler = TimeoutHandler()
    
    @pytest.mark.asyncio
    async def test_operation_within_timeout(self):
        """Test operation that completes within timeout"""
        async with self.timeout_handler.timeout_after(1.0, "test_operation"):
            await asyncio.sleep(0.1)  # Short operation
        
        # Should complete without exception
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self):
        """Test operation that times out"""
        with pytest.raises(HTTPException) as exc_info:
            async with self.timeout_handler.timeout_after(0.1, "test_operation"):
                await asyncio.sleep(1.0)  # Long operation
        
        assert exc_info.value.status_code == 504
        assert "timed out" in str(exc_info.value.detail)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def setup_method(self):
        self.circuit_breaker = CircuitBreaker(
            name="test_service",
            failure_threshold=2,
            recovery_timeout=1
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        async def successful_operation():
            return "success"
        
        result = await self.circuit_breaker.call(successful_operation)
        assert result == "success"
        assert self.circuit_breaker.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        async def failing_operation():
            raise Exception("Service failure")
        
        # First failure
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_operation)
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state.value == "open"
        
        # Third call should fail immediately due to open circuit
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            await self.circuit_breaker.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state"""
        async def failing_operation():
            raise Exception("Service failure")
        
        async def successful_operation():
            return "success"
        
        # Trigger circuit opening
        for _ in range(2):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state.value == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should move to half-open and succeed
        result = await self.circuit_breaker.call(successful_operation)
        assert result == "success"
        assert self.circuit_breaker.state.value == "closed"


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def setup_method(self):
        self.metrics_collector = MetricsCollector(retention_hours=1)
    
    def test_increment_counter(self):
        """Test counter increment"""
        self.metrics_collector.increment_counter("test_counter", 1.0)
        self.metrics_collector.increment_counter("test_counter", 2.0)
        
        assert self.metrics_collector.counters["test_counter"] == 3.0
    
    def test_set_gauge(self):
        """Test gauge setting"""
        self.metrics_collector.set_gauge("test_gauge", 42.0)
        
        assert self.metrics_collector.gauges["test_gauge"] == 42.0
    
    def test_record_histogram(self):
        """Test histogram recording"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for value in values:
            self.metrics_collector.record_histogram("test_histogram", value)
        
        assert len(self.metrics_collector.histograms["test_histogram"]) == 5
        assert self.metrics_collector.histograms["test_histogram"] == values
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation"""
        self.metrics_collector.increment_counter("test_counter", 5.0)
        self.metrics_collector.set_gauge("test_gauge", 10.0)
        self.metrics_collector.record_histogram("test_histogram", 1.0)
        self.metrics_collector.record_histogram("test_histogram", 2.0)
        self.metrics_collector.record_histogram("test_histogram", 3.0)
        
        summary = self.metrics_collector.get_metrics_summary()
        
        assert summary["counters"]["test_counter"] == 5.0
        assert summary["gauges"]["test_gauge"] == 10.0
        assert summary["histograms"]["test_histogram"]["count"] == 3
        assert summary["histograms"]["test_histogram"]["avg"] == 2.0


class TestHealthChecker:
    """Test health checking functionality"""
    
    def setup_method(self):
        self.health_checker = HealthChecker()
    
    @pytest.mark.asyncio
    async def test_check_service_health_success(self):
        """Test successful health check"""
        async def healthy_service():
            return True
        
        status = await self.health_checker.check_service_health("test_service", healthy_service)
        
        assert status.service == "test_service"
        assert status.status == "healthy"
        assert status.response_time_ms is not None
        assert status.response_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_check_service_health_failure(self):
        """Test failed health check"""
        async def unhealthy_service():
            raise Exception("Service unavailable")
        
        status = await self.health_checker.check_service_health("test_service", unhealthy_service)
        
        assert status.service == "test_service"
        assert status.status == "unhealthy"
        assert "Service unavailable" in status.details["error"]
    
    def test_get_overall_health_all_healthy(self):
        """Test overall health when all services are healthy"""
        from app.core.monitoring import HealthStatus
        from datetime import datetime
        
        self.health_checker.health_status = {
            "service1": HealthStatus("service1", "healthy"),
            "service2": HealthStatus("service2", "healthy")
        }
        
        overall_health = self.health_checker.get_overall_health()
        
        assert overall_health["status"] == "healthy"
        assert overall_health["healthy_services"] == 2
        assert overall_health["total_services"] == 2
    
    def test_get_overall_health_degraded(self):
        """Test overall health when some services are unhealthy"""
        from app.core.monitoring import HealthStatus
        
        self.health_checker.health_status = {
            "service1": HealthStatus("service1", "healthy"),
            "service2": HealthStatus("service2", "unhealthy")
        }
        
        overall_health = self.health_checker.get_overall_health()
        
        assert overall_health["status"] == "degraded"
        assert overall_health["healthy_services"] == 1
        assert overall_health["total_services"] == 2


@pytest.mark.asyncio
async def test_error_handling_middleware():
    """Test error handling middleware"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)
    
    @app.get("/test-error")
    async def test_error():
        raise ValueError("Test error")
    
    @app.get("/test-success")
    async def test_success():
        return {"message": "success"}
    
    client = TestClient(app)
    
    # Test error handling
    response = client.get("/test-error")
    assert response.status_code == 500
    assert "InternalServerError" in response.json()["error"]
    
    # Test successful request
    response = client.get("/test-success")
    assert response.status_code == 200
    assert response.json()["message"] == "success"


@pytest.mark.asyncio
async def test_response_time_middleware():
    """Test response time middleware"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    app.add_middleware(ResponseTimeMiddleware, max_response_time_seconds=0.1)
    
    @app.get("/test-fast")
    async def test_fast():
        return {"message": "fast"}
    
    @app.get("/test-slow")
    async def test_slow():
        await asyncio.sleep(0.2)  # Slower than threshold
        return {"message": "slow"}
    
    client = TestClient(app)
    
    # Test fast response
    response = client.get("/test-fast")
    assert response.status_code == 200
    assert "X-Processing-Time" in response.headers
    
    # Test slow response (should still work but be logged)
    response = client.get("/test-slow")
    assert response.status_code == 200
    assert "X-Processing-Time" in response.headers