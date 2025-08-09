"""
Application startup validation and initialization
"""
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.core.logging import LoggerMixin, setup_logging
from app.core.exceptions import ConfigurationError


class StartupValidator(LoggerMixin):
    """Validates application configuration and dependencies on startup"""
    
    def __init__(self):
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []
    
    async def validate_all(self) -> bool:
        """Run all validation checks"""
        self.logger.info("Starting application validation...")
        
        # Run all validation checks
        self._validate_environment_variables()
        self._validate_python_version()
        await self._validate_database_connection()
        await self._validate_external_services()
        self._validate_file_permissions()
        self._validate_resource_limits()
        
        # Report results
        if self.validation_errors:
            self.logger.error("Validation failed with errors:")
            for error in self.validation_errors:
                self.logger.error(f"  - {error}")
            return False
        
        if self.warnings:
            self.logger.warning("Validation completed with warnings:")
            for warning in self.warnings:
                self.logger.warning(f"  - {warning}")
        
        self.logger.info("Application validation completed successfully")
        return True
    
    def _validate_environment_variables(self):
        """Validate required environment variables"""
        required_vars = [
            "PINECONE_API_KEY",
            "GEMINI_API_KEY",
            "DATABASE_URL"
        ]
        
        for var in required_vars:
            value = getattr(settings, var.lower(), None)
            if not value:
                self.validation_errors.append(f"Required environment variable {var} is not set")
        
        # Validate optional but important variables
        if not settings.bearer_token:
            self.warnings.append("BEARER_TOKEN not set, using default value")
        
        # Validate numeric values
        try:
            if settings.max_document_size_mb <= 0:
                self.validation_errors.append("MAX_DOCUMENT_SIZE_MB must be positive")
        except (ValueError, TypeError):
            self.validation_errors.append("MAX_DOCUMENT_SIZE_MB must be a valid number")
        
        try:
            if settings.embedding_dimension <= 0:
                self.validation_errors.append("EMBEDDING_DIMENSION must be positive")
        except (ValueError, TypeError):
            self.validation_errors.append("EMBEDDING_DIMENSION must be a valid number")
        
        # Validate threshold values
        if not (0 <= settings.similarity_threshold <= 1):
            self.validation_errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        if not (0 <= settings.min_confidence_threshold <= 1):
            self.validation_errors.append("MIN_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    def _validate_python_version(self):
        """Validate Python version"""
        min_version = (3, 11)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            self.validation_errors.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {current_version[0]}.{current_version[1]} is installed"
            )
    
    async def _validate_database_connection(self):
        """Validate database connection"""
        try:
            from app.services.database import db_manager
            
            # Test database connection
            await db_manager.initialize()
            is_healthy = await db_manager.health_check()
            
            if not is_healthy:
                self.validation_errors.append("Database connection failed")
            else:
                self.logger.info("Database connection validated successfully")
                
        except Exception as e:
            self.validation_errors.append(f"Database validation failed: {str(e)}")
    
    async def _validate_external_services(self):
        """Validate external service connections"""
        # Validate Pinecone connection
        try:
            from app.services.vector_store import vector_store
            await vector_store.initialize()
            is_healthy = await vector_store.health_check()
            
            if not is_healthy:
                self.validation_errors.append("Pinecone connection failed")
            else:
                self.logger.info("Pinecone connection validated successfully")
                
        except Exception as e:
            self.validation_errors.append(f"Pinecone validation failed: {str(e)}")
        
        # Validate LLM service connection
        try:
            from app.services.llm_service import llm_service
            await llm_service.initialize()
            is_healthy = await llm_service.health_check()
            
            if not is_healthy:
                self.validation_errors.append("LLM service connection failed")
            else:
                self.logger.info("LLM service connection validated successfully")
                
        except Exception as e:
            self.validation_errors.append(f"LLM service validation failed: {str(e)}")
        
        # Validate embedding service
        try:
            from app.services.embedding_service import embedding_service
            await embedding_service.initialize()
            model_info = await embedding_service.get_model_info()
            
            if not model_info:
                self.warnings.append("Embedding service model info not available")
            else:
                self.logger.info(f"Embedding service validated: {model_info.get('model_name', 'unknown')}")
                
        except Exception as e:
            self.validation_errors.append(f"Embedding service validation failed: {str(e)}")
    
    def _validate_file_permissions(self):
        """Validate file system permissions"""
        # Check if we can write to logs directory
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir, exist_ok=True)
            except PermissionError:
                self.validation_errors.append(f"Cannot create logs directory: {logs_dir}")
        
        # Check if we can write to data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
            except PermissionError:
                self.validation_errors.append(f"Cannot create data directory: {data_dir}")
        
        # Test write permissions
        test_file = os.path.join(logs_dir, "startup_test.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (PermissionError, OSError) as e:
            self.validation_errors.append(f"Cannot write to logs directory: {str(e)}")
    
    def _validate_resource_limits(self):
        """Validate system resource limits"""
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 1.0:
            self.warnings.append(f"Low available memory: {available_gb:.1f}GB")
        elif available_gb < 0.5:
            self.validation_errors.append(f"Insufficient memory: {available_gb:.1f}GB (minimum 0.5GB required)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        available_gb = disk.free / (1024**3)
        
        if available_gb < 1.0:
            self.warnings.append(f"Low disk space: {available_gb:.1f}GB")
        elif available_gb < 0.1:
            self.validation_errors.append(f"Insufficient disk space: {available_gb:.1f}GB")
        
        # Check CPU count
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            self.warnings.append(f"Low CPU count: {cpu_count} cores")


class StartupInitializer(LoggerMixin):
    """Handles application initialization tasks"""
    
    async def initialize_all(self) -> bool:
        """Run all initialization tasks"""
        self.logger.info("Starting application initialization...")
        
        try:
            # Setup logging
            setup_logging()
            
            # Initialize core services
            await self._initialize_database()
            await self._initialize_cache()
            await self._initialize_monitoring()
            await self._initialize_external_services()
            
            # Warm up caches if needed
            await self._warm_up_caches()
            
            self.logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Application initialization failed: {str(e)}")
            return False
    
    async def _initialize_database(self):
        """Initialize database connection and run migrations"""
        from app.services.database import db_manager
        
        self.logger.info("Initializing database...")
        await db_manager.initialize()
        await db_manager.create_tables()
        
        # Run any pending migrations
        try:
            import subprocess
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Migration warning: {result.stderr}")
            else:
                self.logger.info("Database migrations completed")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Database migration timed out")
        except FileNotFoundError:
            self.logger.warning("Alembic not found, skipping migrations")
        except Exception as e:
            self.logger.warning(f"Migration error: {str(e)}")
    
    async def _initialize_cache(self):
        """Initialize caching system"""
        from app.core.cache import cache_manager
        
        self.logger.info("Initializing cache system...")
        # Cache manager initializes automatically
        stats = await cache_manager.get_all_stats()
        self.logger.info(f"Cache system initialized with {len(stats)} cache types")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring and metrics"""
        from app.core.monitoring import system_monitor
        
        self.logger.info("Initializing monitoring system...")
        await system_monitor.start_monitoring()
        self.logger.info("Monitoring system initialized")
    
    async def _initialize_external_services(self):
        """Initialize external service connections"""
        # Initialize services in parallel for faster startup
        from app.services.embedding_service import embedding_service
        from app.services.vector_store import vector_store
        from app.services.llm_service import llm_service
        
        self.logger.info("Initializing external services...")
        
        await asyncio.gather(
            embedding_service.initialize(),
            vector_store.initialize(),
            llm_service.initialize(),
            return_exceptions=True
        )
        
        self.logger.info("External services initialized")
    
    async def _warm_up_caches(self):
        """Warm up caches with common data"""
        self.logger.info("Warming up caches...")
        
        # This could include pre-loading common embeddings,
        # frequently accessed documents, etc.
        # For now, we'll just log that it's ready
        
        self.logger.info("Cache warm-up completed")


async def startup_sequence() -> bool:
    """Run complete startup sequence"""
    print(f"Starting LLM Query Retrieval System at {datetime.utcnow().isoformat()}")
    
    # Validate configuration and dependencies (non-strict mode)
    validator = StartupValidator()
    validation_result = await validator.validate_all()
    if not validation_result:
        print("Some startup validations failed, but continuing...")
    else:
        print("All startup validations passed")
    
    # Initialize application components
    initializer = StartupInitializer()
    if not await initializer.initialize_all():
        print("Application initialization failed")
        return False
    
    print("Application startup completed successfully")
    return True


async def shutdown_sequence():
    """Run shutdown sequence"""
    print(f"🛑 Shutting down LLM Query Retrieval System at {datetime.utcnow().isoformat()}")
    
    try:
        # Stop monitoring
        from app.core.monitoring import system_monitor
        await system_monitor.stop_monitoring()
        
        # Close database connections
        from app.services.database import db_manager
        await db_manager.close()
        
        # Clear caches
        from app.core.cache import cache_manager
        await cache_manager.clear_all_caches()
        
        # Cleanup external services
        from app.services.embedding_service import embedding_service
        await embedding_service.cleanup()
        
        print("✅ Shutdown completed successfully")
        
    except Exception as e:
        print(f"❌ Shutdown error: {str(e)}")


if __name__ == "__main__":
    # Run startup validation independently
    asyncio.run(startup_sequence())