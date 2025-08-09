"""
Logging configuration for the application
"""
import logging
import sys
from typing import Dict, Any
from datetime import datetime
import json

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging() -> None:
    """Configure application logging"""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on environment
    if settings.environment == "production":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)
    
    def log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log a message with additional context"""
        logger = self.logger
        extra_fields = kwargs
        
        # Create a log record with extra fields
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields
        
        logger.handle(record)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_request_metrics(self, 
                          method: str, 
                          path: str, 
                          status_code: int, 
                          response_time_ms: float,
                          **kwargs):
        """Log request performance metrics"""
        metrics = {
            "event_type": "request_metrics",
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        metrics.update(kwargs)
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Request metrics",
            args=(),
            exc_info=None
        )
        record.extra_fields = metrics
        self.logger.handle(record)
    
    def log_service_metrics(self, 
                          service_name: str, 
                          operation: str, 
                          duration_ms: float,
                          success: bool,
                          **kwargs):
        """Log service performance metrics"""
        metrics = {
            "event_type": "service_metrics",
            "service_name": service_name,
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        metrics.update(kwargs)
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Service metrics",
            args=(),
            exc_info=None
        )
        record.extra_fields = metrics
        self.logger.handle(record)


class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_auth_event(self, 
                      event_type: str, 
                      client_ip: str, 
                      user_agent: str,
                      success: bool,
                      **kwargs):
        """Log authentication events"""
        event = {
            "event_type": "auth_event",
            "auth_event_type": event_type,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        event.update(kwargs)
        
        level = logging.INFO if success else logging.WARNING
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=f"Authentication event: {event_type}",
            args=(),
            exc_info=None
        )
        record.extra_fields = event
        self.logger.handle(record)
    
    def log_rate_limit_event(self, 
                           client_ip: str, 
                           path: str,
                           limit_exceeded: bool,
                           **kwargs):
        """Log rate limiting events"""
        event = {
            "event_type": "rate_limit_event",
            "client_ip": client_ip,
            "path": path,
            "limit_exceeded": limit_exceeded,
            "timestamp": datetime.utcnow().isoformat()
        }
        event.update(kwargs)
        
        level = logging.WARNING if limit_exceeded else logging.INFO
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg="Rate limit event",
            args=(),
            exc_info=None
        )
        record.extra_fields = event
        self.logger.handle(record)


# Global logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()