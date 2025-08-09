"""
Unit tests for document loader service
"""
import pytest
import pytest_asyncio
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientResponse

from app.services.document_loader import DocumentLoader
from app.core.exceptions import DocumentProcessingError
from app.schemas.models import DocumentType, DocumentStatus


class TestDocumentLoader:
    """Test DocumentLoader functionality"""
    
    @pytest_asyncio.fixture
    async def loader(self):
        """Create DocumentLoader instance"""
        async with DocumentLoader() as loader:
            yield loader
    
    @pytest.mark.asyncio
    async def test_load_document_success(self, loader):
        """Test successful document loading"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'application/pdf',
            'content-length': '1000'
        }
        mock_response.read = AsyncMock(return_value=b'%PDF-1.4 test content')
        
        # Mock session
        loader.session.get = AsyncMock()
        loader.session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        loader.session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Test
        document = await loader.load_document("https://example.com/test.pdf")
        
        assert document.url == "https://example.com/test.pdf"
        assert document.content_type == "application/pdf"
        assert document.status == DocumentStatus.PROCESSING
        assert document.document_type == DocumentType.PDF
        assert document.metadata["file_size"] == 19
    
    @pytest.mark.asyncio
    async def test_load_document_invalid_url(self, loader):
        """Test loading with invalid URL"""
        with pytest.raises(DocumentProcessingError) as exc_info:
            await loader.load_document("invalid-url")
        
        assert "Invalid URL format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_document_http_error(self, loader):
        """Test loading with HTTP error"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Mock session
        loader.session.get = AsyncMock()
        loader.session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        loader.session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            await loader.load_document("https://example.com/notfound.pdf")
        
        assert "HTTP 404" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_document_too_large(self, loader):
        """Test loading document that's too large"""
        # Mock response with large content-length
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'application/pdf',
            'content-length': str(loader.max_file_size + 1)
        }
        
        # Mock session
        loader.session.get = AsyncMock()
        loader.session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        loader.session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            await loader.load_document("https://example.com/large.pdf")
        
        assert "Document too large" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_document_retry_logic(self, loader):
        """Test retry logic on failure"""
        # Mock first two attempts to fail, third to succeed
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.headers = {'content-type': 'application/pdf'}
        mock_response_success.read = AsyncMock(return_value=b'%PDF-1.4 test')
        
        # Mock session to fail twice, then succeed
        loader.session.get = AsyncMock()
        loader.session.get.return_value.__aenter__ = AsyncMock(
            side_effect=[mock_response_fail, mock_response_fail, mock_response_success]
        )
        loader.session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            document = await loader.load_document("https://example.com/test.pdf")
        
        assert document.url == "https://example.com/test.pdf"
        assert loader.session.get.call_count == 3
    
    def test_determine_document_type_pdf(self):
        """Test PDF document type determination"""
        loader = DocumentLoader()
        
        # Test by URL extension
        doc_type = loader._determine_document_type("https://example.com/test.pdf", "")
        assert doc_type == DocumentType.PDF
        
        # Test by content type
        doc_type = loader._determine_document_type("https://example.com/test", "application/pdf")
        assert doc_type == DocumentType.PDF
    
    def test_determine_document_type_docx(self):
        """Test DOCX document type determination"""
        loader = DocumentLoader()
        
        # Test by URL extension
        doc_type = loader._determine_document_type("https://example.com/test.docx", "")
        assert doc_type == DocumentType.DOCX
        
        # Test by content type
        doc_type = loader._determine_document_type("https://example.com/test", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert doc_type == DocumentType.DOCX
    
    def test_determine_document_type_email(self):
        """Test email document type determination"""
        loader = DocumentLoader()
        
        # Test by URL extension
        doc_type = loader._determine_document_type("https://example.com/test.eml", "")
        assert doc_type == DocumentType.EMAIL
        
        # Test by content type
        doc_type = loader._determine_document_type("https://example.com/test", "message/rfc822")
        assert doc_type == DocumentType.EMAIL
    
    def test_determine_document_type_unknown(self):
        """Test unknown document type determination"""
        loader = DocumentLoader()
        
        doc_type = loader._determine_document_type("https://example.com/test.txt", "text/plain")
        assert doc_type == DocumentType.UNKNOWN
    
    def test_validate_document_format_pdf(self):
        """Test PDF format validation"""
        loader = DocumentLoader()
        
        # Valid PDF
        assert loader._validate_document_format(b'%PDF-1.4 content', DocumentType.PDF) == True
        
        # Invalid PDF
        assert loader._validate_document_format(b'not a pdf', DocumentType.PDF) == False
    
    def test_validate_document_format_docx(self):
        """Test DOCX format validation"""
        loader = DocumentLoader()
        
        # Valid DOCX (ZIP signature)
        assert loader._validate_document_format(b'PK\x03\x04content', DocumentType.DOCX) == True
        
        # Invalid DOCX
        assert loader._validate_document_format(b'not a docx', DocumentType.DOCX) == False
    
    def test_validate_document_format_email(self):
        """Test email format validation"""
        loader = DocumentLoader()
        
        # Valid email
        email_content = b'From: test@example.com\nTo: user@example.com\nSubject: Test\n\nBody'
        assert loader._validate_document_format(email_content, DocumentType.EMAIL) == True
        
        # Invalid email
        assert loader._validate_document_format(b'not an email', DocumentType.EMAIL) == False
    
    @pytest.mark.asyncio
    async def test_validate_url_accessibility_success(self, loader):
        """Test URL accessibility validation success"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        
        # Mock session
        loader.session.head = AsyncMock()
        loader.session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        loader.session.head.return_value.__aexit__ = AsyncMock(return_value=None)
        
        is_accessible, error = await loader.validate_url_accessibility("https://example.com/test.pdf")
        
        assert is_accessible == True
        assert error is None
    
    @pytest.mark.asyncio
    async def test_validate_url_accessibility_failure(self, loader):
        """Test URL accessibility validation failure"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Mock session
        loader.session.head = AsyncMock()
        loader.session.head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        loader.session.head.return_value.__aexit__ = AsyncMock(return_value=None)
        
        is_accessible, error = await loader.validate_url_accessibility("https://example.com/notfound.pdf")
        
        assert is_accessible == False
        assert "HTTP 404" in error