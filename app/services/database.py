"""
Database connection and management service
"""
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.connection_pool import connection_pool_manager
from app.core.exceptions import ConfigurationError
from app.models.database import Base, Document, QueryLog, SystemConfig


class DatabaseManager(LoggerMixin):
    """Database connection and operations manager"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = None
        self.async_session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection and session factory"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.log_level == "DEBUG",
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._initialized = True
            self.logger.info("Database connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise ConfigurationError(f"Database initialization failed: {str(e)}")
    
    async def create_tables(self) -> None:
        """Create database tables"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check database connection health"""
        if not self._initialized:
            return False
        
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager"""
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def store_document_metadata(self, metadata: Dict[str, Any]) -> str:
        """Store document metadata and return document ID"""
        async with self.get_session() as session:
            document = Document(
                url=metadata["url"],
                content_type=metadata.get("content_type"),
                status=metadata.get("status", "processing"),
                metadata=metadata.get("metadata", {})
            )
            session.add(document)
            await session.flush()
            
            self.logger.info(f"Stored document metadata: {document.id}")
            return str(document.id)
    
    async def update_document_status(self, document_id: str, status: str) -> bool:
        """Update document processing status"""
        async with self.get_session() as session:
            result = await session.get(Document, document_id)
            if result:
                result.status = status
                self.logger.info(f"Updated document {document_id} status to {status}")
                return True
            return False
    
    async def log_query(self, query_log: Dict[str, Any]) -> bool:
        """Log query and response"""
        async with self.get_session() as session:
            log_entry = QueryLog(
                document_id=query_log.get("document_id"),
                query=query_log["query"],
                response=query_log.get("response"),
                processing_time_ms=query_log.get("processing_time_ms")
            )
            session.add(log_entry)
            
            self.logger.info(f"Logged query: {log_entry.id}")
            return True
    
    async def get_configuration(self, key: str) -> Optional[str]:
        """Get configuration value by key"""
        async with self.get_session() as session:
            result = await session.get(SystemConfig, key)
            return result.value if result else None
    
    async def set_configuration(self, key: str, value: str, description: str = "") -> bool:
        """Set configuration value"""
        async with self.get_session() as session:
            config = await session.get(SystemConfig, key)
            if config:
                config.value = value
                config.description = description
            else:
                config = SystemConfig(key=key, value=value, description=description)
                session.add(config)
            
            self.logger.info(f"Set configuration: {key}")
            return True
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()