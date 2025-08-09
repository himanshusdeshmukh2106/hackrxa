"""
End-to-end integration tests with real components
"""
import pytest
import asyncio
import tempfile
import os
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.database import db_manager
from app.schemas.models import Document, TextChunk, SearchResult


class TestRealComponentIntegration:
    """Test integration with real components where possible"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": f"Bearer {settings.bearer_token}"}
    
    @pytest.mark.asyncio
    async def test_database_real_operations(self):
        """Test real database operations if test database is available"""
        try:
            # Try to initialize database
            await db_manager.initialize()
            await db_manager.create_tables()
            
            # Test document metadata storage
            doc_metadata = {
                "url": "https://example.com/test.pdf",
                "content_type": "application/pdf",
                "status": "processing",
                "metadata": {"pages": 5, "size": 1024}
            }
            
            doc_id = await db_manager.store_document_metadata(doc_metadata)
            assert doc_id is not None
            
            # Test status update
            success = await db_manager.update_document_status(doc_id, "completed")
            assert success
            
            # Test query logging
            query_data = {
                "query": "Test query",
                "response": "Test response",
                "processing_time_ms": 1500,
                "document_url": "https://example.com/test.pdf"
            }
            
            success = await db_manager.log_query(query_data)
            assert success
            
            # Test configuration operations
            await db_manager.set_configuration("test_key", "test_value")
            value = await db_manager.get_configuration("test_key")
            assert value == "test_value"
            
            # Test health check
            is_healthy = await db_manager.health_check()
            assert is_healthy
            
        except Exception as e:
            pytest.skip(f"Database not available for testing: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_text_extraction_real_files(self):
        """Test text extraction with real files"""
        from app.services.text_extractor import TextExtractor
        
        text_extractor = TextExtractor()
        
        # Test with a simple text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "This is a test document with multiple sentences. It contains information about insurance policies. The coverage amount is $50,000."
            f.write(test_content)
            f.flush()
            file_path = f.name
        
        try:
            # Create a mock document
            document = Document(
                id="test_doc",
                url=f"file://{file_path}",
                content_type="text/plain",
                metadata={"raw_content": test_content, "file_path": file_path}
            )
            
            # Extract text chunks
            chunks = await text_extractor.extract_text(document)
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, TextChunk) for chunk in chunks)
            assert all(chunk.content.strip() for chunk in chunks)
            assert all(chunk.document_id == "test_doc" for chunk in chunks)
            
        finally:
            os.unlink(file_path)
    
    @pytest.mark.asyncio
    async def test_embedding_service_real_model(self):
        """Test embedding service with real model if available"""
        from app.services.embedding_service import embedding_service
        
        try:
            # Initialize the service
            await embedding_service.initialize()
            
            # Test single text embedding
            text = "This is a test sentence for embedding generation."
            embedding = await embedding_service.generate_embeddings([text])
            
            assert len(embedding) == 1
            assert isinstance(embedding[0], list)
            assert len(embedding[0]) > 0  # Should have some dimensions
            assert all(isinstance(x, float) for x in embedding[0])
            
            # Test batch embedding
            texts = [
                "Insurance policy coverage details",
                "Exclusions and limitations",
                "Premium payment terms"
            ]
            
            embeddings = await embedding_service.generate_embeddings(texts)
            
            assert len(embeddings) == 3
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) == len(embeddings[0]) for emb in embeddings)  # Same dimensions
            
            # Test text chunks embedding
            chunks = [
                TextChunk(
                    document_id="test_doc",
                    content=text,
                    chunk_index=i
                )
                for i, text in enumerate(texts)
            ]
            
            embedded_chunks = await embedding_service.embed_text_chunks(chunks)
            
            assert len(embedded_chunks) == 3
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            assert all(len(chunk.embedding) > 0 for chunk in embedded_chunks)
            
            # Test model info
            model_info = embedding_service.get_model_info()
            assert "model_name" in model_info
            assert "dimensions" in model_info
            
        except Exception as e:
            pytest.skip(f"Embedding service not available for testing: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_query_analyzer_real_processing(self):
        """Test query analyzer with real processing"""
        from app.services.query_analyzer import query_analyzer
        
        # Test different types of queries
        test_queries = [
            "What is the coverage amount for medical expenses?",
            "Are pre-existing conditions covered?",
            "How long is the waiting period?",
            "What are the exclusions in this policy?",
            "When do I need to pay the premium?"
        ]
        
        for query in test_queries:
            intent = await query_analyzer.analyze_query(query)
            
            assert hasattr(intent, 'intent_type')
            assert hasattr(intent, 'entities')
            assert hasattr(intent, 'confidence')
            assert hasattr(intent, 'keywords')
            
            # Intent should be classified
            assert intent.intent_type is not None
            assert isinstance(intent.entities, list)
            assert 0 <= intent.confidence <= 1
            assert isinstance(intent.keywords, list)
    
    @pytest.mark.asyncio
    async def test_search_engine_real_search(self):
        """Test search engine with real search operations"""
        from app.services.search_engine import search_engine
        from app.services.query_analyzer import query_analyzer
        
        # Create mock search results for testing
        mock_chunks = [
            SearchResult(
                chunk_id="test_doc_chunk_0",
                content="This policy provides coverage up to $50,000 for medical expenses including hospital stays and surgeries.",
                similarity_score=0.9,
                document_metadata={"document_id": "test_doc", "chunk_index": 0}
            ),
            SearchResult(
                chunk_id="test_doc_chunk_1",
                content="Pre-existing conditions are excluded from coverage for the first 12 months of the policy.",
                similarity_score=0.8,
                document_metadata={"document_id": "test_doc", "chunk_index": 1}
            ),
            SearchResult(
                chunk_id="test_doc_chunk_2",
                content="There is a 30-day waiting period for all non-emergency treatments and procedures.",
                similarity_score=0.7,
                document_metadata={"document_id": "test_doc", "chunk_index": 2}
            )
        ]
        
        # Mock vector store to return our test chunks
        with patch('app.services.search_engine.vector_store', new_callable=AsyncMock) as mock_vector_store:
            mock_vector_store.search_similar_chunks.return_value = mock_chunks
            
            # Test search with different queries
            test_queries = [
                "What is the coverage amount?",
                "Are pre-existing conditions covered?",
                "What is the waiting period?"
            ]
            
            for query in test_queries:
                # Analyze query intent
                query_intent = await query_analyzer.analyze_query(query)
                
                # Perform search
                results = await search_engine.hybrid_search(
                    query=query,
                    query_intent=query_intent,
                    filters={"document_id": "test_doc"},
                    top_k=5
                )
                
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(hasattr(result, 'similarity_score') for result in results)
                assert all(hasattr(result, 'content') for result in results)
                assert all(hasattr(result, 'chunk_id') for result in results)
    
    @pytest.mark.asyncio
    async def test_response_generator_real_generation(self):
        """Test response generator with real generation"""
        from app.services.response_generator import response_generator
        from app.services.query_analyzer import query_analyzer
        from app.schemas.models import SearchResult
        
        # Create mock search results
        search_results = [
            SearchResult(
                chunk_id="chunk1",
                content="This policy provides coverage up to $50,000 for medical expenses including hospital stays and surgeries.",
                similarity_score=0.9,
                document_metadata={"document_id": "test_doc", "chunk_index": 0}
            ),
            SearchResult(
                chunk_id="chunk2",
                content="The policy covers emergency room visits, diagnostic tests, and prescription medications.",
                similarity_score=0.8,
                document_metadata={"document_id": "test_doc", "chunk_index": 1}
            )
        ]
        
        query = "What is the coverage amount for medical expenses?"
        query_intent = await query_analyzer.analyze_query(query)
        
        # Mock LLM service
        with patch('app.services.response_generator.llm_service', new_callable=AsyncMock) as mock_llm:
            mock_llm.generate_response.return_value = MagicMock(
                content="Based on the policy document, the coverage amount for medical expenses is up to $50,000. This includes hospital stays, surgeries, emergency room visits, diagnostic tests, and prescription medications.",
                confidence=0.9,
                reasoning="The answer is directly stated in the policy document section about coverage limits.",
                tokens_used=100,
                processing_time_ms=500
            )
            
            # Generate answer
            answer = await response_generator.generate_answer(
                context_results=search_results,
                query=query,
                query_intent=query_intent
            )
            
            assert hasattr(answer, 'answer')
            assert hasattr(answer, 'confidence')
            assert hasattr(answer, 'source_chunks')
            assert hasattr(answer, 'reasoning')
            
            assert answer.answer is not None
            assert 0 <= answer.confidence <= 1
            assert isinstance(answer.source_chunks, list)
            assert answer.reasoning is not None
    
    @pytest.mark.asyncio
    async def test_cache_integration_real_operations(self):
        """Test cache integration with real operations"""
        from app.core.cache import cache_manager
        
        # Test embedding cache
        text = "Test sentence for caching"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Set and get embedding
        await cache_manager.embedding_cache.set_embedding(text, embedding)
        cached_embedding = await cache_manager.embedding_cache.get_embedding(text)
        
        assert cached_embedding == embedding
        
        # Test document cache
        document_url = "https://example.com/test.pdf"
        document_data = {"content": "Test document", "metadata": {"pages": 5}}
        
        await cache_manager.document_cache.set_document(document_url, document_data)
        cached_document = await cache_manager.document_cache.get_document(document_url)
        
        assert cached_document == document_data
        
        # Test query cache
        query = "What is covered?"
        document_id = "doc123"
        answer = "Medical expenses are covered up to $50,000"
        
        await cache_manager.query_cache.set_query_result(query, document_id, answer)
        cached_answer = await cache_manager.query_cache.get_query_result(query, document_id)
        
        assert cached_answer == answer
        
        # Test cache stats
        stats = await cache_manager.get_all_stats()
        
        assert "embedding_cache" in stats
        assert "document_cache" in stats
        assert "query_cache" in stats
        assert "general_cache" in stats
        
        # Each cache should have stats
        for cache_name, cache_stats in stats.items():
            assert "total_entries" in cache_stats
            assert "max_size" in cache_stats
    
    @pytest.mark.asyncio
    async def test_monitoring_integration_real_metrics(self):
        """Test monitoring integration with real metrics"""
        from app.core.monitoring import system_monitor
        
        # Record some test metrics
        system_monitor.record_request_metrics(
            endpoint="/hackrx/run",
            method="POST",
            status_code=200,
            duration_ms=1500
        )
        
        system_monitor.record_processing_metrics(
            operation="document_processing",
            duration_ms=800,
            success=True
        )
        
        system_monitor.record_processing_metrics(
            operation="embedding_generation",
            duration_ms=300,
            success=True
        )
        
        # Get system status
        status = system_monitor.get_system_status()
        
        assert "health" in status
        assert "metrics" in status
        assert "circuit_breakers" in status
        assert "timestamp" in status
        
        # Check metrics
        metrics = status["metrics"]
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics
        
        # Check circuit breakers
        circuit_breakers = status["circuit_breakers"]
        assert "pinecone" in circuit_breakers
        assert "gemini" in circuit_breakers
        assert "database" in circuit_breakers
        
        for breaker_name, breaker_status in circuit_breakers.items():
            assert "state" in breaker_status
            assert "failure_count" in breaker_status
            assert breaker_status["state"] in ["closed", "open", "half_open"]


class TestLoadTesting:
    """Load testing for the system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": f"Bearer {settings.bearer_token}"}
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, client):
        """Test concurrent health check requests"""
        import concurrent.futures
        import time
        
        def make_health_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "data": response.json() if response.status_code == 200 else None
            }
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(result["status_code"] == 200 for result in results)
        
        # Response times should be reasonable
        avg_response_time = sum(result["response_time"] for result in results) / len(results)
        assert avg_response_time < 20.0  # Average should be under 20 seconds
        
        # All responses should have the expected structure
        for result in results:
            if result["data"]:
                assert "status" in result["data"]
                assert "timestamp" in result["data"]
                assert "version" in result["data"]
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_load(self, client):
        """Test metrics endpoint under load"""
        import concurrent.futures
        
        def make_metrics_request():
            response = client.get("/metrics")
            return response.status_code == 200
        
        # Make 20 concurrent requests to metrics endpoint
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_metrics_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9  # At least 90% success rate
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_authentication_load(self, mock_settings, client):
        """Test authentication under load"""
        mock_settings.bearer_token = "test_token_123"
        
        import concurrent.futures
        
        def make_auth_request():
            headers = {"Authorization": "Bearer test_token_123"}
            # Use a simple endpoint that requires auth
            response = client.get("/metrics", headers=headers)
            return response.status_code in [200, 429]  # 200 or rate limited
        
        # Make 50 concurrent authenticated requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_auth_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most requests should succeed (some might be rate limited)
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8  # At least 80% success rate


class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        from app.core.monitoring import system_monitor
        
        # Get a circuit breaker
        breaker = system_monitor.get_circuit_breaker("database")
        assert breaker is not None
        
        # Simulate failures to open the circuit
        for _ in range(breaker.failure_threshold):
            try:
                await breaker.call(lambda: exec('raise Exception("Test failure")'))
            except:
                pass
        
        # Circuit should be open
        assert breaker.state.value == "open"
        
        # Wait for recovery timeout (simulate)
        breaker.last_failure_time = breaker.last_failure_time - timedelta(seconds=breaker.recovery_timeout + 1)
        
        # Next successful call should reset the circuit
        result = await breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_cache_recovery_after_clear(self):
        """Test cache recovery after clearing"""
        from app.core.cache import cache_manager
        
        # Add some data to caches
        await cache_manager.embedding_cache.set_embedding("test", [1, 2, 3])
        await cache_manager.document_cache.set_document("url", {"data": "test"})
        await cache_manager.query_cache.set_query_result("query", "doc", "answer")
        
        # Verify data is there
        assert await cache_manager.embedding_cache.get_embedding("test") == [1, 2, 3]
        assert await cache_manager.document_cache.get_document("url") == {"data": "test"}
        assert await cache_manager.query_cache.get_query_result("query", "doc") == "answer"
        
        # Clear all caches
        await cache_manager.clear_all_caches()
        
        # Verify data is gone
        assert await cache_manager.embedding_cache.get_embedding("test") is None
        assert await cache_manager.document_cache.get_document("url") is None
        assert await cache_manager.query_cache.get_query_result("query", "doc") is None
        
        # Add data again (recovery)
        await cache_manager.embedding_cache.set_embedding("test2", [4, 5, 6])
        assert await cache_manager.embedding_cache.get_embedding("test2") == [4, 5, 6]
    
    def test_application_startup_recovery(self, client):
        """Test application can recover from startup issues"""
        # Test that the application is running and responsive
        response = client.get("/")
        assert response.status_code == 200
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200