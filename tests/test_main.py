"""
Unit tests for main FastAPI application
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from app.main import app, QueryProcessor, query_processor
from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse
from app.schemas.models import Document, TextChunk, SearchResult, Answer


class TestQueryProcessor:
    """Test QueryProcessor functionality"""
    
    @pytest.fixture
    def query_processor(self):
        """Create QueryProcessor instance"""
        return QueryProcessor()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample query request"""
        return QueryRequest(
            documents="https://example.com/test.pdf",
            questions=[
                "What is the coverage?",
                "What are the exclusions?"
            ]
        )
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document"""
        doc = Document(
            id="doc123",
            url="https://example.com/test.pdf",
            content_type="application/pdf"
        )
        doc.text_chunks = [
            TextChunk(
                document_id="doc123",
                content="This policy covers medical expenses up to $10,000.",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            ),
            TextChunk(
                document_id="doc123",
                content="Pre-existing conditions are excluded from coverage.",
                chunk_index=1,
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        return doc
    
    @pytest.mark.asyncio
    @patch('app.main.DocumentLoader')
    @patch('app.main.db_manager')
    async def test_load_and_process_document(self, mock_db_manager, mock_loader_class, query_processor, sample_document):
        """Test document loading and processing"""
        # Mock document loader
        mock_loader = AsyncMock()
        mock_loader.load_document.return_value = sample_document
        mock_loader_class.return_value.__aenter__.return_value = mock_loader
        mock_loader_class.return_value.__aexit__.return_value = None
        
        # Mock database
        mock_db_manager.store_document_metadata.return_value = "doc123"
        
        # Mock text extractor
        with patch.object(query_processor.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = sample_document.text_chunks
            
            result = await query_processor._load_and_process_document("https://example.com/test.pdf")
            
            assert result.id == "doc123"
            assert len(result.text_chunks) == 2
            mock_db_manager.store_document_metadata.assert_called_once()
            mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.main.embedding_service')
    @patch('app.main.vector_store')
    @patch('app.main.db_manager')
    async def test_process_document_embeddings(self, mock_db_manager, mock_vector_store, mock_embedding_service, query_processor, sample_document):
        """Test document embedding processing"""
        mock_embedding_service.embed_text_chunks.return_value = sample_document.text_chunks
        mock_vector_store.store_text_chunks.return_value = True
        mock_db_manager.update_document_status.return_value = True
        
        await query_processor._process_document_embeddings(sample_document)
        
        mock_embedding_service.embed_text_chunks.assert_called_once()
        mock_vector_store.store_text_chunks.assert_called_once()
        mock_db_manager.update_document_status.assert_called_with("doc123", "completed")
    
    @pytest.mark.asyncio
    @patch('app.main.query_analyzer')
    @patch('app.main.search_engine')
    @patch('app.main.response_generator')
    async def test_process_questions(self, mock_response_generator, mock_search_engine, mock_query_analyzer, query_processor):
        """Test question processing"""
        questions = ["What is covered?", "What are exclusions?"]
        document_id = "doc123"
        
        # Mock query analyzer
        mock_intent = MagicMock()
        mock_intent.intent_type = "coverage"
        mock_query_analyzer.analyze_query.return_value = mock_intent
        
        # Mock search engine
        mock_search_results = [
            SearchResult(
                chunk_id="chunk1",
                content="Coverage information",
                similarity_score=0.8,
                document_metadata={}
            )
        ]
        mock_search_engine.hybrid_search.return_value = mock_search_results
        
        # Mock response generator
        mock_answer = Answer(
            question="What is covered?",
            answer="Medical expenses are covered up to $10,000.",
            confidence=0.8,
            source_chunks=["chunk1"],
            reasoning="Based on policy document"
        )
        mock_response_generator.generate_answer.return_value = mock_answer
        
        answers = await query_processor._process_questions(questions, document_id)
        
        assert len(answers) == 2
        assert answers[0] == "Medical expenses are covered up to $10,000."
        mock_query_analyzer.analyze_query.assert_called()
        mock_search_engine.hybrid_search.assert_called()
    
    @pytest.mark.asyncio
    @patch('app.main.query_analyzer')
    @patch('app.main.search_engine')
    @patch('app.main.response_generator')
    async def test_process_questions_batch_processing(self, mock_response_generator, mock_search_engine, mock_query_analyzer, query_processor):
        """Test question processing with batch processing"""
        questions = ["Question 1?", "Question 2?"]
        document_id = "doc123"
        
        # Mock shared search results (high overlap)
        shared_results = [
            SearchResult(chunk_id="chunk1", content="Shared content", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk2", content="More content", similarity_score=0.8, document_metadata={})
        ]
        
        mock_query_analyzer.analyze_query.return_value = MagicMock()
        mock_search_engine.hybrid_search.return_value = shared_results
        
        # Mock batch answer generation
        mock_answers = [
            Answer(question="Question 1?", answer="Answer 1", confidence=0.8, source_chunks=[], reasoning=""),
            Answer(question="Question 2?", answer="Answer 2", confidence=0.7, source_chunks=[], reasoning="")
        ]
        mock_response_generator.generate_batch_answers.return_value = mock_answers
        
        answers = await query_processor._process_questions(questions, document_id)
        
        assert len(answers) == 2
        assert answers[0] == "Answer 1"
        assert answers[1] == "Answer 2"
        mock_response_generator.generate_batch_answers.assert_called_once()
    
    def test_can_use_shared_context_high_overlap(self, query_processor):
        """Test shared context detection with high overlap"""
        search_results_1 = [
            SearchResult(chunk_id="chunk1", content="Content 1", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk2", content="Content 2", similarity_score=0.8, document_metadata={})
        ]
        search_results_2 = [
            SearchResult(chunk_id="chunk1", content="Content 1", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk3", content="Content 3", similarity_score=0.7, document_metadata={})
        ]
        
        all_results = [search_results_1, search_results_2]
        
        can_share = query_processor._can_use_shared_context(all_results)
        
        assert can_share == True  # Should detect overlap
    
    def test_can_use_shared_context_low_overlap(self, query_processor):
        """Test shared context detection with low overlap"""
        search_results_1 = [
            SearchResult(chunk_id="chunk1", content="Content 1", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk2", content="Content 2", similarity_score=0.8, document_metadata={})
        ]
        search_results_2 = [
            SearchResult(chunk_id="chunk3", content="Content 3", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk4", content="Content 4", similarity_score=0.8, document_metadata={})
        ]
        
        all_results = [search_results_1, search_results_2]
        
        can_share = query_processor._can_use_shared_context(all_results)
        
        assert can_share == False  # Should not detect significant overlap
    
    def test_merge_search_results(self, query_processor):
        """Test merging search results from multiple queries"""
        search_results_1 = [
            SearchResult(chunk_id="chunk1", content="Content 1", similarity_score=0.9, document_metadata={}),
            SearchResult(chunk_id="chunk2", content="Content 2", similarity_score=0.8, document_metadata={})
        ]
        search_results_2 = [
            SearchResult(chunk_id="chunk1", content="Content 1", similarity_score=0.7, document_metadata={}),  # Lower score
            SearchResult(chunk_id="chunk3", content="Content 3", similarity_score=0.85, document_metadata={})
        ]
        
        all_results = [search_results_1, search_results_2]
        
        merged = query_processor._merge_search_results(all_results)
        
        assert len(merged) == 3  # chunk1, chunk2, chunk3
        # Should keep higher similarity score for chunk1
        chunk1_result = next(r for r in merged if r.chunk_id == "chunk1")
        assert chunk1_result.similarity_score == 0.9
        
        # Should be sorted by similarity score
        scores = [r.similarity_score for r in merged]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    @patch('app.main.db_manager')
    async def test_log_query_batch(self, mock_db_manager, query_processor, sample_request):
        """Test query batch logging"""
        answers = ["Answer 1", "Answer 2"]
        processing_time = 1500
        
        mock_db_manager.log_query.return_value = True
        
        await query_processor._log_query_batch(sample_request, answers, processing_time)
        
        assert mock_db_manager.log_query.call_count == 2  # One for each question
    
    @pytest.mark.asyncio
    @patch('app.main.db_manager')
    async def test_log_failed_query(self, mock_db_manager, query_processor, sample_request):
        """Test failed query logging"""
        error = "Processing failed"
        processing_time = 500
        
        mock_db_manager.log_query.return_value = True
        
        await query_processor._log_failed_query(sample_request, error, processing_time)
        
        mock_db_manager.log_query.assert_called_once()
        call_args = mock_db_manager.log_query.call_args[0][0]
        assert "ERROR" in call_args["response"]
        assert error in call_args["response"]


class TestFastAPIApp:
    """Test FastAPI application endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": "Bearer test_token_123"}
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    @patch('app.main.db_manager')
    @patch('app.main.vector_store')
    @patch('app.main.llm_service')
    def test_health_check_healthy(self, mock_llm_service, mock_vector_store, mock_db_manager, client):
        """Test health check endpoint when all services are healthy"""
        mock_db_manager.health_check.return_value = True
        mock_vector_store.health_check.return_value = True
        mock_llm_service.health_check.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == True
        assert data["vector_store"] == True
        assert data["llm_service"] == True
    
    @patch('app.main.db_manager')
    @patch('app.main.vector_store')
    @patch('app.main.llm_service')
    def test_health_check_unhealthy(self, mock_llm_service, mock_vector_store, mock_db_manager, client):
        """Test health check endpoint when services are unhealthy"""
        mock_db_manager.health_check.return_value = False
        mock_vector_store.health_check.return_value = True
        mock_llm_service.health_check.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["database"] == False
    
    def test_process_query_no_auth(self, client):
        """Test query endpoint without authentication"""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        response = client.post("/hackrx/run", json=request_data)
        
        assert response.status_code == 401  # Unauthorized
    
    @patch('app.main.settings')
    @patch('app.main.query_processor')
    def test_process_query_success(self, mock_query_processor, mock_settings, client, auth_headers):
        """Test successful query processing"""
        mock_settings.bearer_token = "test_token_123"
        
        # Mock query processor
        mock_response = QueryResponse(answers=["Answer 1", "Answer 2"])
        mock_query_processor.process_query_request.return_value = mock_response
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["Question 1?", "Question 2?"]
        }
        
        response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == 2
        assert data["answers"][0] == "Answer 1"
        assert data["answers"][1] == "Answer 2"
    
    @patch('app.main.settings')
    def test_process_query_missing_document(self, mock_settings, client, auth_headers):
        """Test query processing with missing document URL"""
        mock_settings.bearer_token = "test_token_123"
        
        request_data = {
            "documents": "",
            "questions": ["What is covered?"]
        }
        
        response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
        
        assert response.status_code == 422
        data = response.json()
        assert "Document URL is required" in data["detail"]
    
    @patch('app.main.settings')
    def test_process_query_missing_questions(self, mock_settings, client, auth_headers):
        """Test query processing with missing questions"""
        mock_settings.bearer_token = "test_token_123"
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }
        
        response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
        
        assert response.status_code == 422
        data = response.json()
        assert "At least one question is required" in data["detail"]
    
    @patch('app.main.settings')
    @patch('app.main.query_processor')
    def test_process_query_processing_error(self, mock_query_processor, mock_settings, client, auth_headers):
        """Test query processing with processing error"""
        mock_settings.bearer_token = "test_token_123"
        
        # Mock query processor to raise exception
        mock_query_processor.process_query_request.side_effect = Exception("Processing failed")
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "LLM-Powered Query Retrieval System"
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
    
    def test_docs_endpoint(self, client):
        """Test documentation endpoint"""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint"""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestExceptionHandlers:
    """Test custom exception handlers"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_document_processing_error_handler(self, client):
        """Test DocumentProcessingError handler"""
        from app.core.exceptions import DocumentProcessingError
        
        # This would need to be triggered by an actual endpoint
        # For now, we test that the handler is registered
        assert any(
            handler.exc_class == DocumentProcessingError 
            for handler in app.exception_handlers.values()
        )
    
    def test_embedding_generation_error_handler(self, client):
        """Test EmbeddingGenerationError handler"""
        from app.core.exceptions import EmbeddingGenerationError
        
        assert any(
            handler.exc_class == EmbeddingGenerationError 
            for handler in app.exception_handlers.values()
        )
    
    def test_vector_store_error_handler(self, client):
        """Test VectorStoreError handler"""
        from app.core.exceptions import VectorStoreError
        
        assert any(
            handler.exc_class == VectorStoreError 
            for handler in app.exception_handlers.values()
        )
    
    def test_llm_service_error_handler(self, client):
        """Test LLMServiceError handler"""
        from app.core.exceptions import LLMServiceError
        
        assert any(
            handler.exc_class == LLMServiceError 
            for handler in app.exception_handlers.values()
        )
    
    def test_response_generation_error_handler(self, client):
        """Test ResponseGenerationError handler"""
        from app.core.exceptions import ResponseGenerationError
        
        assert any(
            handler.exc_class == ResponseGenerationError 
            for handler in app.exception_handlers.values()
        )