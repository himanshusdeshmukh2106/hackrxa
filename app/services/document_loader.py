"""
Document loader service for downloading and processing documents from URLs
"""
import asyncio
import aiohttp
import hashlib
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import mimetypes
from pathlib import Path

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import DocumentProcessingError
from app.schemas.models import Document, DocumentType, DocumentStatus


class DocumentLoader(LoggerMixin):
    """Service for loading documents from URLs"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_file_size = settings.max_document_size_mb * 1024 * 1024  # Convert to bytes
        self.cache_dir = Path("data/document_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds),
            headers={"User-Agent": "LLM-Query-Retrieval-System/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def load_document(self, url: str) -> Document:
        """
        Load document from URL with retry logic and error handling
        
        Args:
            url: Document URL to download
            
        Returns:
            Document object with metadata
            
        Raises:
            DocumentProcessingError: If document cannot be loaded
        """
        if not self.session:
            raise DocumentProcessingError("DocumentLoader not initialized as context manager")
        
        # Create a unique filename for the cached file
        file_hash = hashlib.sha256(url.encode()).hexdigest()
        cached_file_path = self.cache_dir / file_hash
        
        # Check if the file is already cached
        if cached_file_path.exists():
            self.logger.info(f"Loading document from cache: {cached_file_path}")
            content = cached_file_path.read_bytes()
            content_type = mimetypes.guess_type(url)[0] or 'application/octet-stream'
        else:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise DocumentProcessingError(f"Invalid URL format: {url}")
            
            # Attempt download with retries
            content, content_type = await self._download_document(url)
            
            # Save the downloaded file to the cache
            cached_file_path.write_bytes(content)
            self.logger.info(f"Saved document to cache: {cached_file_path}")
            
        # Create document object
        document = Document(
            url=url,
            content_type=content_type,
            status=DocumentStatus.PROCESSING,
            metadata={
                "file_size": len(content),
                "content_type": content_type,
                "download_timestamp": "2025-01-08T00:00:00Z"
            }
        )
        
        # Store raw content temporarily
        document.metadata["raw_content"] = content
        
        self.logger.info(f"Successfully loaded document: {url} ({len(content)} bytes)")
        return document

    async def _download_document(self, url: str) -> Tuple[bytes, str]:
        """Download document content and return it with its content type."""
        for attempt in range(settings.retry_attempts):
            try:
                self.logger.info(f"Attempting to download document (attempt {attempt + 1}): {url}")
                
                async with self.session.get(url) as response:
                    # Check response status
                    if response.status != 200:
                        raise DocumentProcessingError(
                            f"Failed to download document: HTTP {response.status}",
                            details={"url": url, "status": response.status}
                        )
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        raise DocumentProcessingError(
                            f"Document too large: {content_length} bytes (max: {self.max_file_size})",
                            details={"url": url, "size": content_length}
                        )
                    
                    # Get content type
                    content_type = response.headers.get('content-type', '')
                    
                    # Read content
                    content = await response.read()
                    
                    # Validate content size
                    if len(content) > self.max_file_size:
                        raise DocumentProcessingError(
                            f"Document too large: {len(content)} bytes (max: {self.max_file_size})",
                            details={"url": url, "size": len(content)}
                        )
                    
                    return content, content_type
                    
            except aiohttp.ClientError as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == settings.retry_attempts - 1:
                    raise DocumentProcessingError(
                        f"Failed to download document after {settings.retry_attempts} attempts: {str(e)}",
                        details={"url": url, "error": str(e)}
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                self.logger.error(f"Unexpected error downloading document: {str(e)}")
                raise DocumentProcessingError(
                    f"Unexpected error downloading document: {str(e)}",
                    details={"url": url, "error": str(e)}
                )
        
        # This should never be reached due to the retry logic above
        raise DocumentProcessingError("Failed to download document")
    
    def _determine_document_type(self, url: str, content_type: str) -> DocumentType:
        """
        Determine document type from URL and content type
        
        Args:
            url: Document URL
            content_type: HTTP content type header
            
        Returns:
            DocumentType enum value
        """
        # Check URL extension
        url_lower = url.lower()
        if url_lower.endswith('.pdf'):
            return DocumentType.PDF
        elif url_lower.endswith(('.docx', '.doc')):
            return DocumentType.DOCX
        elif url_lower.endswith('.eml'):
            return DocumentType.EMAIL
        
        # Check content type
        content_type_lower = content_type.lower()
        if 'pdf' in content_type_lower:
            return DocumentType.PDF
        elif 'word' in content_type_lower or 'officedocument' in content_type_lower:
            return DocumentType.DOCX
        elif 'email' in content_type_lower or 'message' in content_type_lower:
            return DocumentType.EMAIL
        
        return DocumentType.UNKNOWN
    
    def _validate_document_format(self, content: bytes, document_type: DocumentType) -> bool:
        """
        Validate document format by checking file signatures
        
        Args:
            content: Raw document content
            document_type: Expected document type
            
        Returns:
            True if format is valid, False otherwise
        """
        if not content:
            return False
        
        # PDF signature
        if document_type == DocumentType.PDF:
            return content.startswith(b'%PDF-')
        
        # DOCX signature (ZIP file with specific structure)
        elif document_type == DocumentType.DOCX:
            return content.startswith(b'PK\x03\x04')
        
        # Email format is more flexible, just check for basic structure
        elif document_type == DocumentType.EMAIL:
            try:
                content_str = content.decode('utf-8', errors='ignore')
                return any(header in content_str for header in ['From:', 'To:', 'Subject:', 'Date:'])
            except:
                return False
        
        return True
    
    async def validate_url_accessibility(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if URL is accessible without downloading full content
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_accessible, error_message)
        """
        if not self.session:
            return False, "DocumentLoader not initialized"
        
        try:
            async with self.session.head(url) as response:
                if response.status == 200:
                    return True, None
                else:
                    return False, f"HTTP {response.status}"
        except Exception as e:
            return False, str(e)