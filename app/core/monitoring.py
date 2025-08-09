"""
Monitoring and metrics collection
"""
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import LoggerMixin
from app.core.config import settings


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class HealthStatus:
    """Health status data structure"""
    service: str
    status: str  # healthy, unhealthy, degraded
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(LoggerMixin):
    """Circuit breaker implementation for external service calls"""
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info(f"Circuit breaker {self.name} reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class MetricsCollector(LoggerMixin):
    """Metrics collection and aggregation"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Start cleanup task only if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._cleanup_old_metrics())
        except RuntimeError:
            # No event loop running, will start cleanup later
            pass
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self.counters[key],
            labels=labels or {}
        )
        self.metrics[key].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        )
        self.metrics[key].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {}
        )
        self.metrics[key].append(metric)
    
    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000  # Convert to ms
                    self.record_histogram(f"{name}_duration_ms", duration, labels)
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000  # Convert to ms
                    self.record_histogram(f"{name}_duration_ms", duration, labels)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        # Calculate histogram statistics
        for key, values in self.histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
        
        return summary
    
    def get_recent_metrics(self, name: str, minutes: int = 60) -> List[Metric]:
        """Get recent metrics for a specific name"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = []
        
        for key, metric_deque in self.metrics.items():
            if name in key:
                for metric in metric_deque:
                    if metric.timestamp >= cutoff_time:
                        recent_metrics.append(metric)
        
        return sorted(recent_metrics, key=lambda m: m.timestamp)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    async def _cleanup_old_metrics(self):
        """Periodically clean up old metrics"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
                
                for key, metric_deque in self.metrics.items():
                    # Remove old metrics
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                
                self.logger.debug("Cleaned up old metrics")
                
            except Exception as e:
                self.logger.error(f"Error cleaning up metrics: {str(e)}")
            
            # Sleep for 1 hour
            await asyncio.sleep(3600)


class HealthChecker(LoggerMixin):
    """Health checking for system components"""
    
    def __init__(self):
        self.health_status: Dict[str, HealthStatus] = {}
        self.check_interval = 60  # seconds
        self.running = False
    
    async def start_health_checks(self):
        """Start periodic health checks"""
        self.running = True
        asyncio.create_task(self._periodic_health_checks())
    
    async def stop_health_checks(self):
        """Stop periodic health checks"""
        self.running = False
    
    async def check_service_health(self, service_name: str, check_func) -> HealthStatus:
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus(
                service=service_name,
                status="healthy" if result else "unhealthy",
                response_time_ms=response_time,
                details={"check_result": result}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus(
                service=service_name,
                status="unhealthy",
                response_time_ms=response_time,
                details={"error": str(e)}
            )
            
            self.logger.error(f"Health check failed for {service_name}: {str(e)}")
        
        self.health_status[service_name] = status
        return status
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.health_status:
            return {
                "status": "unknown",
                "services": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        healthy_services = sum(1 for status in self.health_status.values() if status.status == "healthy")
        total_services = len(self.health_status)
        
        if healthy_services == total_services:
            overall_status = "healthy"
        elif healthy_services == 0:
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "services": {name: {
                "status": status.status,
                "response_time_ms": status.response_time_ms,
                "timestamp": status.timestamp.isoformat(),
                "details": status.details
            } for name, status in self.health_status.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _periodic_health_checks(self):
        """Run periodic health checks"""
        while self.running:
            try:
                # Import here to avoid circular imports
                from app.services.database import db_manager
                from app.services.vector_store import vector_store
                from app.services.llm_service import llm_service
                from app.services.embedding_service import embedding_service
                
                # Check all services
                await asyncio.gather(
                    self.check_service_health("database", db_manager.health_check),
                    self.check_service_health("vector_store", vector_store.health_check),
                    self.check_service_health("llm_service", llm_service.health_check),
                    return_exceptions=True
                )
                
                self.logger.debug("Completed periodic health checks")
                
            except Exception as e:
                self.logger.error(f"Error in periodic health checks: {str(e)}")
            
            await asyncio.sleep(self.check_interval)


class SystemMonitor(LoggerMixin):
    """Main system monitoring coordinator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Create circuit breakers for external services
        self.circuit_breakers["pinecone"] = CircuitBreaker(
            name="pinecone",
            failure_threshold=3,
            recovery_timeout=30
        )
        
        self.circuit_breakers["gemini"] = CircuitBreaker(
            name="gemini",
            failure_threshold=5,
            recovery_timeout=60
        )
        
        self.circuit_breakers["database"] = CircuitBreaker(
            name="database",
            failure_threshold=3,
            recovery_timeout=30
        )
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        await self.health_checker.start_health_checks()
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        await self.health_checker.stop_health_checks()
        self.logger.info("System monitoring stopped")
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def record_request_metrics(self, endpoint: str, method: str, status_code: int, duration_ms: float):
        """Record HTTP request metrics"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        self.metrics_collector.increment_counter("http_requests_total", 1.0, labels)
        self.metrics_collector.record_histogram("http_request_duration_ms", duration_ms, labels)
        
        # Record error rate
        if status_code >= 400:
            self.metrics_collector.increment_counter("http_errors_total", 1.0, labels)
    
    def record_processing_metrics(self, operation: str, duration_ms: float, success: bool):
        """Record processing operation metrics"""
        labels = {
            "operation": operation,
            "status": "success" if success else "error"
        }
        
        self.metrics_collector.increment_counter("processing_operations_total", 1.0, labels)
        self.metrics_collector.record_histogram("processing_duration_ms", duration_ms, labels)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "health": self.health_checker.get_overall_health(),
            "metrics": self.metrics_collector.get_metrics_summary(),
            "circuit_breakers": {
                name: breaker.get_status() 
                for name, breaker in self.circuit_breakers.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Global system monitor instance
system_monitor = SystemMonitor()