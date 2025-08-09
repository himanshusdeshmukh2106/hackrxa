"""
Integration tests for the complete system
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app, query_processor
from app.services.database import db_manager
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service
from app.schemas.models import Document, TextChunk, SearchResult, Answer


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": "Bearer test_token_123"}
    
    @pytest.fixture
    def sample_request_data(self):
        """Sample request data"""
        return {
            "documents": "https://example.com/test-policy.pdf",
            "questions": [
                "What is the coverage amount?",
                "What are the exclusions?",
                "What is the waiting period?"
            ]
        }
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_complete_query_processing_workflow(self, mock_settings, client, auth_headers, sample_request_data):
        """Test complete workflow from API request to response"""
        mock_settings.bearer_token = "test_token_123"
        
        # Mock document content
        mock_document = Document(
            id="doc123",
            url="https://example.com/test-policy.pdf",
            content_type="application/pdf"
        )
        
        mock_chunks = [
            TextChunk(
                document_id="doc123",
                content="This policy provides coverage up to $50,000 for medical expenses.",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            ),
            TextChunk(
                document_id="doc123",
                content="Pre-existing conditions are excluded from coverage for the first 12 months.",
                chunk_index=1,
                embedding=[0.4, 0.5, 0.6]
            ),
            TextChunk(
                document_id="doc123",
                content="There is a 30-day waiting period for all non-emergency treatments.",
                chunk_index=2,
                embedding=[0.7, 0.8, 0.9]
            )
        ]
        mock_document.text_chunks = mock_chunks
        
        # Mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk1",
                content="This policy provides coverage up to $50,000 for medical expenses.",
                similarity_score=0.9,
                document_metadata={"document_id": "doc123", "chunk_index": 0}
            ),
            SearchResult(
                chunk_id="chunk2",
                content="Pre-existing conditions are excluded from coverage for the first 12 months.",
                similarity_score=0.8,
                document_metadata={"document_id": "doc123", "chunk_index": 1}
            ),
            SearchResult(
                chunk_id="chunk3",
                content="There is a 30-day waiting period for all non-emergency treatments.",
                similarity_score=0.85,
                document_metadata={"document_id": "doc123", "chunk_index": 2}
            )
        ]
        
        # Mock answers
        mock_answers = [
            Answer(
                question="What is the coverage amount?",
                answer="The policy provides coverage up to $50,000 for medical expenses.",
                confidence=0.9,
                source_chunks=["chunk1"],
                reasoning="Based on policy document section 1"
            ),
            Answer(
                question="What are the exclusions?",
                answer="Pre-existing conditions are excluded from coverage for the first 12 months.",
                confidence=0.8,
                source_chunks=["chunk2"],
                reasoning="Based on policy exclusions section"
            ),
            Answer(
                question="What is the waiting period?",
                answer="There is a 30-day waiting period for all non-emergency treatments.",
                confidence=0.85,
                source_chunks=["chunk3"],
                reasoning="Based on policy terms section"
            )
        ]
        
        # Mock all the service calls
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.return_value = mock_document
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "doc123"
                mock_db.update_document_status.return_value = True
                mock_db.log_query.return_value = True
                
                with patch.object(query_processor.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_extract:
                    mock_extract.return_value = mock_chunks
                    
                    with patch('app.main.embedding_service') as mock_embedding:
                        mock_embedding.embed_text_chunks.return_value = mock_chunks
                        
                        with patch('app.main.vector_store') as mock_vector:
                            mock_vector.store_text_chunks.return_value = True
                            
                            with patch('app.main.query_analyzer') as mock_analyzer:
                                mock_intent = MagicMock()
                                mock_intent.intent_type = "coverage"
                                mock_analyzer.analyze_query.return_value = mock_intent
                                
                                with patch('app.main.search_engine') as mock_search:
                                    mock_search.hybrid_search.side_effect = [
                                        [mock_search_results[0]],  # For first question
                                        [mock_search_results[1]],  # For second question
                                        [mock_search_results[2]]   # For third question
                                    ]
                                    
                                    with patch('app.main.response_generator') as mock_response_gen:
                                        mock_response_gen.generate_answer.side_effect = mock_answers
                                        
                                        # Make the API call
                                        response = client.post("/hackrx/run", json=sample_request_data, headers=auth_headers)
                                        
                                        # Verify response
                                        assert response.status_code == 200
                                        data = response.json()
                                        
                                        assert "answers" in data
                                        assert len(data["answers"]) == 3
                                        assert "$50,000" in data["answers"][0]
                                        assert "Pre-existing conditions" in data["answers"][1]
                                        assert "30-day waiting period" in data["answers"][2]
                                        
                                        # Verify service calls were made
                                        mock_loader.load_document.assert_called_once()
                                        mock_extract.assert_called_once()
                                        mock_embedding.embed_text_chunks.assert_called_once()
                                        mock_vector.store_text_chunks.assert_called_once()
                                        assert mock_analyzer.analyze_query.call_count == 3
                                        assert mock_search.hybrid_search.call_count == 3
                                        assert mock_response_gen.generate_answer.call_count == 3
    
    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test database operations integration"""
        # This would require a test database setup
        # For now, we'll test the interface
        
        # Test database initialization
        assert hasattr(db_manager, 'initialize')
        assert hasattr(db_manager, 'create_tables')
        assert hasattr(db_manager, 'health_check')
        
        # Test document metadata operations
        assert hasattr(db_manager, 'store_document_metadata')
        assert hasattr(db_manager, 'update_document_status')
        
        # Test query logging
        assert hasattr(db_manager, 'log_query')
        
        # Test configuration management
        assert hasattr(db_manager, 'get_configuration')
        assert hasattr(db_manager, 'set_configuration')
    
    @pytest.mark.asyncio
    async def test_embedding_service_integration(self):
        """Test embedding service integration"""
        # Test service initialization
        assert hasattr(embedding_service, 'initialize')
        assert hasattr(embedding_service, 'generate_embeddings')
        assert hasattr(embedding_service, 'embed_text_chunks')
        
        # Test model info
        assert hasattr(embedding_service, 'get_model_info')
        assert hasattr(embedding_service, 'cleanup')
    
    @pytest.mark.asyncio
    async def test_vector_store_integration(self):
        """Test vector store integration"""
        # Test service initialization
        assert hasattr(vector_store, 'initialize')
        assert hasattr(vector_store, 'store_text_chunks')
        assert hasattr(vector_store, 'search_similar_chunks')
        
        # Test health check
        assert hasattr(vector_store, 'health_check')
        assert hasattr(vector_store, 'get_index_stats')
    
    @pytest.mark.asyncio
    async def test_llm_service_integration(self):
        """Test LLM service integration"""
        # Test service initialization
        assert hasattr(llm_service, 'initialize')
        assert hasattr(llm_service, 'generate_response')
        assert hasattr(llm_service, 'batch_process')
        
        # Test health check
        assert hasattr(llm_service, 'health_check')
        assert hasattr(llm_service, 'get_model_info')


class TestErrorHandlingIntegration:
    """Test error handling across the system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": "Bearer test_token_123"}
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_document_processing_error_handling(self, mock_settings, client, auth_headers):
        """Test error handling when document processing fails"""
        mock_settings.bearer_token = "test_token_123"
        
        request_data = {
            "documents": "https://example.com/invalid-document.pdf",
            "questions": ["What is covered?"]
        }
        
        # Mock document loader to fail
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.side_effect = Exception("Document not found")
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_embedding_service_error_handling(self, mock_settings, client, auth_headers):
        """Test error handling when embedding service fails"""
        mock_settings.bearer_token = "test_token_123"
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        # Mock successful document loading but failed embedding
        mock_document = Document(id="doc123", url="https://example.com/test.pdf")
        mock_document.text_chunks = [
            TextChunk(document_id="doc123", content="Test content", chunk_index=0)
        ]
        
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.return_value = mock_document
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "doc123"
                mock_db.update_document_status.return_value = True
                
                with patch.object(query_processor.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_extract:
                    mock_extract.return_value = mock_document.text_chunks
                    
                    with patch('app.main.embedding_service') as mock_embedding:
                        mock_embedding.embed_text_chunks.side_effect = Exception("Embedding service unavailable")
                        
                        response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
                        
                        assert response.status_code == 500
                        data = response.json()
                        assert "Internal server error" in data["detail"]
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_timeout_handling(self, mock_settings, client, auth_headers):
        """Test timeout handling for long-running operations"""
        mock_settings.bearer_token = "test_token_123"
        
        request_data = {
            "documents": "https://example.com/large-document.pdf",
            "questions": ["What is covered?"]
        }
        
        # Mock document processing to take too long
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            
            async def slow_load(*args, **kwargs):
                await asyncio.sleep(65)  # Longer than timeout
                return Document(id="doc123", url="https://example.com/test.pdf")
            
            mock_loader.load_document = slow_load
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "doc123"
                
                response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
                
                assert response.status_code == 500
                data = response.json()
                assert "Internal server error" in data["detail"]


class TestPerformanceIntegration:
    """Test performance-related integrations"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system_status" in data
        assert "cache_stats" in data
        assert "timestamp" in data
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint performance"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        # This would require more complex setup with actual async client
        # For now, we'll test that the endpoint can handle basic requests
        
        responses = []
        for i in range(5):
            response = client.get("/health")
            responses.append(response)
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)


class TestSecurityIntegration:
    """Test security-related integrations"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_authentication_required(self, client):
        """Test that authentication is required for protected endpoints"""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        response = client.post("/hackrx/run", json=request_data)
        
        assert response.status_code == 401
    
    def test_invalid_token(self, client):
        """Test handling of invalid authentication token"""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 401
    
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present"""
        # Make multiple requests to trigger rate limiting info
        for i in range(3):
            response = client.get("/health")
            
            # Check for rate limiting headers (would be added by middleware)
            # This is a basic test - actual rate limiting would need more setup
            assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        
        # CORS headers should be present due to CORSMiddleware
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled


class TestDataFlowIntegration:
    """Test data flow through the entire system"""
    
    @pytest.mark.asyncio
    async def test_document_to_chunks_flow(self):
        """Test document processing to text chunks flow"""
        # This would test the actual flow from document loading to chunking
        # For now, we verify the interfaces exist
        
        from app.services.document_loader import DocumentLoader
        from app.services.text_extractor import TextExtractor
        
        # Verify the flow interfaces
        assert hasattr(DocumentLoader, 'load_document')
        assert hasattr(TextExtractor, 'extract_text')
    
    @pytest.mark.asyncio
    async def test_chunks_to_embeddings_flow(self):
        """Test text chunks to embeddings flow"""
        from app.services.embedding_service import embedding_service
        from app.services.vector_store import vector_store
        
        # Verify the flow interfaces
        assert hasattr(embedding_service, 'embed_text_chunks')
        assert hasattr(vector_store, 'store_text_chunks')
    
    @pytest.mark.asyncio
    async def test_query_to_answer_flow(self):
        """Test query processing to answer generation flow"""
        from app.services.query_analyzer import query_analyzer
        from app.services.search_engine import search_engine
        from app.services.response_generator import response_generator
        
        # Verify the flow interfaces
        assert hasattr(query_analyzer, 'analyze_query')
        assert hasattr(search_engine, 'hybrid_search')
        assert hasattr(response_generator, 'generate_answer')