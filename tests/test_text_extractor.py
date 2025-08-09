"""
Unit tests for text extraction service
"""
import pytest
import io
from unittest.mock import patch, MagicMock

from app.services.text_extractor import TextExtractor
from app.core.exceptions import DocumentProcessingError
from app.schemas.models import Document, DocumentType, TextChunk


class TestTextExtractor:
    """Test TextExtractor functionality"""
    
    @pytest.fixture
    def extractor(self):
        """Create TextExtractor instance"""
        return TextExtractor()
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document with raw content"""
        doc = Document(
            url="https://example.com/test.pdf",
            document_type=DocumentType.PDF
        )
        doc.metadata["raw_content"] = b"%PDF-1.4 sample content"
        return doc
    
    @pytest.mark.asyncio
    async def test_extract_text_no_raw_content(self, extractor):
        """Test extraction with no raw content"""
        doc = Document(url="https://example.com/test.pdf")
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            await extractor.extract_text(doc)
        
        assert "No raw content found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_extract_text_unsupported_type(self, extractor):
        """Test extraction with unsupported document type"""
        doc = Document(
            url="https://example.com/test.txt",
            document_type=DocumentType.UNKNOWN
        )
        doc.metadata["raw_content"] = b"test content"
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            await extractor.extract_text(doc)
        
        assert "Unsupported document type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.services.text_extractor.pdfplumber')
    async def test_extract_pdf_text_success(self, mock_pdfplumber, extractor, sample_document):
        """Test successful PDF text extraction"""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_page.extract_tables.return_value = []
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        chunks = await extractor.extract_text(sample_document)
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert "Sample PDF text content" in chunks[0].content
        assert chunks[0].document_id == sample_document.id
    
    @pytest.mark.asyncio
    @patch('app.services.text_extractor.pdfplumber')
    @patch('app.services.text_extractor.PyPDF2')
    async def test_extract_pdf_fallback_to_pypdf2(self, mock_pypdf2, mock_pdfplumber, extractor, sample_document):
        """Test PDF extraction fallback to PyPDF2"""
        # Mock pdfplumber to fail
        mock_pdfplumber.open.side_effect = Exception("pdfplumber failed")
        
        # Mock PyPDF2
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF2 extracted text"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        chunks = await extractor.extract_text(sample_document)
        
        assert len(chunks) == 1
        assert "PyPDF2 extracted text" in chunks[0].content
    
    @pytest.mark.asyncio
    @patch('app.services.text_extractor.DocxDocument')
    async def test_extract_docx_text_success(self, mock_docx, extractor):
        """Test successful DOCX text extraction"""
        # Create DOCX document
        doc = Document(
            url="https://example.com/test.docx",
            document_type=DocumentType.DOCX
        )
        doc.metadata["raw_content"] = b"PK\x03\x04 docx content"
        
        # Mock docx
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Sample DOCX paragraph"
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []
        mock_doc.sections = []
        mock_docx.return_value = mock_doc
        
        chunks = await extractor.extract_text(doc)
        
        assert len(chunks) == 1
        assert "Sample DOCX paragraph" in chunks[0].content
    
    @pytest.mark.asyncio
    async def test_extract_email_text_success(self, extractor):
        """Test successful email text extraction"""
        # Create email document
        doc = Document(
            url="https://example.com/test.eml",
            document_type=DocumentType.EMAIL
        )
        
        email_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Wed, 8 Jan 2025 12:00:00 +0000

This is the email body content.
"""
        doc.metadata["raw_content"] = email_content.encode('utf-8')
        
        chunks = await extractor.extract_text(doc)
        
        assert len(chunks) == 1
        assert "sender@example.com" in chunks[0].content
        assert "Test Email" in chunks[0].content
        assert "email body content" in chunks[0].content
    
    def test_create_text_chunks_single(self, extractor):
        """Test creating single chunk for short text"""
        text = "This is a short text that fits in one chunk."
        chunks = extractor._create_text_chunks("doc-123", text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc-123"
    
    def test_create_text_chunks_multiple(self, extractor):
        """Test creating multiple chunks for long text"""
        # Create text longer than max_chunk_size
        long_text = "This is a sentence. " * 100  # Much longer than default chunk size
        chunks = extractor._create_text_chunks("doc-123", long_text)
        
        assert len(chunks) > 1
        assert all(chunk.document_id == "doc-123" for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
        
        # Check that chunks have reasonable sizes
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk.content) <= extractor.max_chunk_size
    
    def test_format_table_text(self, extractor):
        """Test table formatting"""
        table = [
            ["Header 1", "Header 2", "Header 3"],
            ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
            ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
        ]
        
        formatted = extractor._format_table_text(table, 1, 1)
        
        assert "[Page 1 - Table 1]" in formatted
        assert "Header 1 | Header 2 | Header 3" in formatted
        assert "Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3" in formatted
    
    def test_html_to_text(self, extractor):
        """Test HTML to text conversion"""
        html = "<p>This is <b>bold</b> text with <a href='#'>link</a>.</p>"
        text = extractor._html_to_text(html)
        
        assert text == "This is bold text with link."
        assert "<" not in text
        assert ">" not in text
    
    def test_html_to_text_entities(self, extractor):
        """Test HTML entity conversion"""
        html = "Text with&nbsp;entities &lt;tag&gt; &amp; more"
        text = extractor._html_to_text(html)
        
        assert text == "Text with entities <tag> & more"
    
    @pytest.mark.asyncio
    async def test_extract_text_empty_content(self, extractor):
        """Test extraction with empty text content"""
        doc = Document(
            url="https://example.com/empty.pdf",
            document_type=DocumentType.PDF
        )
        doc.metadata["raw_content"] = b"%PDF-1.4"
        
        with patch('app.services.text_extractor.pdfplumber') as mock_pdfplumber:
            # Mock to return empty text
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = ""
            mock_page.extract_tables.return_value = []
            mock_pdf.pages = [mock_page]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            with patch('app.services.text_extractor.PyPDF2') as mock_pypdf2:
                mock_reader = MagicMock()
                mock_page = MagicMock()
                mock_page.extract_text.return_value = ""
                mock_reader.pages = [mock_page]
                mock_pypdf2.PdfReader.return_value = mock_reader
                
                with pytest.raises(DocumentProcessingError) as exc_info:
                    await extractor.extract_text(doc)
                
                assert "No text content extracted" in str(exc_info.value)