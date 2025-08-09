"""
SQLAlchemy database models
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, nullable=False)
    content_type = Column(String(50))
    status = Column(String(20), default="processing")
    doc_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to query logs
    query_logs = relationship("QueryLog", back_populates="document")
    
    def __repr__(self):
        return f"<Document(id={self.id}, url={self.url[:50]}...)>"


class QueryLog(Base):
    """Query logs table"""
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    query = Column(Text, nullable=False)
    response = Column(Text)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("Document", back_populates="query_logs")
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, query={self.query[:50]}...)>"


class SystemConfig(Base):
    """System configuration table"""
    __tablename__ = "system_config"
    
    key = Column(String(100), primary_key=True)
    value = Column(Text)
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemConfig(key={self.key}, value={self.value[:50]}...)>"