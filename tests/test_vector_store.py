"""
Unit tests for vector store service
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.vector_store import VectorStore, Vector, Match
from app.core.exceptions import VectorStoreError
from app.schemas.models import TextChunk, SearchResult


class TestVectorStore:
    """Test VectorStore functionality"""
    
    @pytest.fixture
    def vector_store(self):
        """Create VectorStore instance"""
        return VectorStore()
    
    @pytest.fixture
    def mock_pinecone_client(self):
        """Create mock Pinecone client"""
        client = MagicMock()
        client.list_indexes.return_value = []
        client.create_index = MagicMock()
        client.describe_index.return_value = MagicMock(status=MagicMock(ready=True))
        
        # Mock index
        mock_index = MagicMock()
        client.Index.return_value = mock_index
        
        return client, mock_index
    
    @pytest.mark.asyncio
    @patch('app.services.vector_store.Pinecone')
    async def test_initialize_success(self, mock_pinecone_class, vector_store):
        """Test successful initialization"""
        mock_client, mock_index = self.mock_pinecone_client()
        mock_pinecone_class.return_value = mock_client
        
        await vector_store.initialize()
        
        assert vector_store._initialized == True
        assert vector_store.client is not None
        assert vector_store.index is not None
    
    @pytest.mark.asyncio
    @patch('app.services.vector_store.Pinecone')
    async def test_initialize_create_index(self, mock_pinecone_class, vector_store):
        """Test initialization with index creation"""
        mock_client, mock_index = self.mock_pinecone_client()
        mock_client.list_indexes.return_value = []  # No existing indexes
        mock_pinecone_class.return_value = mock_client
        
        await vector_store.initialize()
        
        # Should create index
        mock_client.create_index.assert_called_once()
        assert vector_store._initialized == True
    
    @pytest.mark.asyncio
    @patch('app.services.vector_store.Pinecone')
    async def test_initialize_failure(self, mock_pinecone_class, vector_store):
        """Test initialization failure"""
        mock_pinecone_class.side_effect = Exception("Pinecone connection failed")
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.initialize()
        
        assert "Pinecone initialization failed" in str(exc_info.value)
        assert vector_store._initialized == False
    
    @pytest.mark.asyncio
    async def test_upsert_vectors_success(self, vector_store):
        """Test successful vector upsert"""
        # Mock initialized state
        vector_store._initialized = True
        mock_index = MagicMock()
        vector_store.index = mock_index
        
        # Create test vectors
        vectors = [
            Vector("vec1", [0.1, 0.2, 0.3], {"content": "test1"}),
            Vector("vec2", [0.4, 0.5, 0.6], {"content": "test2"})
        ]
        
        result = await vector_store.upsert_vectors(vectors)
        
        assert result == True
        mock_index.upsert.assert_called_once()
        
        # Check the call arguments
        call_args = mock_index.upsert.call_args[1]["vectors"]
        assert len(call_args) == 2
        assert call_args[0]["id"] == "vec1"
        assert call_args[0]["values"] == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_upsert_vectors_empty_list(self, vector_store):
        """Test upsert with empty vector list"""
        vector_store._initialized = True
        
        result = await vector_store.upsert_vectors([])
        
        assert result == True
    
    @pytest.mark.asyncio
    async def test_upsert_vectors_failure(self, vector_store):
        """Test vector upsert failure"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.upsert.side_effect = Exception("Upsert failed")
        vector_store.index = mock_index
        
        vectors = [Vector("vec1", [0.1, 0.2, 0.3])]
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.upsert_vectors(vectors)
        
        assert "Failed to upsert vectors" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_similarity_search_success(self, vector_store):
        """Test successful similarity search"""
        vector_store._initialized = True
        mock_index = MagicMock()
        
        # Mock search response
        mock_match = MagicMock()
        mock_match.id = "match1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "test content"}
        
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response
        
        vector_store.index = mock_index
        
        query_vector = [0.1, 0.2, 0.3]
        matches = await vector_store.similarity_search(query_vector, top_k=5)
        
        assert len(matches) == 1
        assert isinstance(matches[0], Match)
        assert matches[0].id == "match1"
        assert matches[0].score == 0.95
        assert matches[0].metadata["content"] == "test content"
        
        # Check query parameters
        mock_index.query.assert_called_once()
        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["vector"] == query_vector
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["include_metadata"] == True
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, vector_store):
        """Test similarity search with metadata filter"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])
        vector_store.index = mock_index
        
        query_vector = [0.1, 0.2, 0.3]
        filter_dict = {"document_id": {"$eq": "doc123"}}
        
        await vector_store.similarity_search(
            query_vector, 
            top_k=10, 
            filter_dict=filter_dict
        )
        
        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == filter_dict
    
    @pytest.mark.asyncio
    async def test_similarity_search_failure(self, vector_store):
        """Test similarity search failure"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.query.side_effect = Exception("Search failed")
        vector_store.index = mock_index
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.similarity_search([0.1, 0.2, 0.3])
        
        assert "Search operation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_delete_vectors_success(self, vector_store):
        """Test successful vector deletion"""
        vector_store._initialized = True
        mock_index = MagicMock()
        vector_store.index = mock_index
        
        vector_ids = ["vec1", "vec2", "vec3"]
        result = await vector_store.delete_vectors(vector_ids)
        
        assert result == True
        mock_index.delete.assert_called_once_with(ids=vector_ids)
    
    @pytest.mark.asyncio
    async def test_delete_vectors_empty_list(self, vector_store):
        """Test delete with empty vector list"""
        vector_store._initialized = True
        
        result = await vector_store.delete_vectors([])
        
        assert result == True
    
    @pytest.mark.asyncio
    async def test_delete_vectors_failure(self, vector_store):
        """Test vector deletion failure"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.delete.side_effect = Exception("Delete failed")
        vector_store.index = mock_index
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.delete_vectors(["vec1"])
        
        assert "Failed to delete vectors" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_store_text_chunks_success(self, vector_store):
        """Test successful text chunk storage"""
        vector_store._initialized = True
        mock_index = MagicMock()
        vector_store.index = mock_index
        
        # Create test chunks with embeddings
        chunks = [
            TextChunk(
                document_id="doc1",
                content="First chunk content",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            ),
            TextChunk(
                document_id="doc1",
                content="Second chunk content",
                chunk_index=1,
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        result = await vector_store.store_text_chunks(chunks)
        
        assert result == True
        mock_index.upsert.assert_called_once()
        
        # Check upserted vectors
        call_args = mock_index.upsert.call_args[1]["vectors"]
        assert len(call_args) == 2
        assert call_args[0]["metadata"]["document_id"] == "doc1"
        assert call_args[0]["metadata"]["chunk_index"] == 0
    
    @pytest.mark.asyncio
    async def test_store_text_chunks_no_embeddings(self, vector_store):
        """Test storing chunks without embeddings"""
        chunks = [
            TextChunk(
                document_id="doc1",
                content="Chunk without embedding",
                chunk_index=0
            )
        ]
        
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.store_text_chunks(chunks)
        
        assert "No chunks with embeddings found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, vector_store):
        """Test searching for similar chunks"""
        vector_store._initialized = True
        mock_index = MagicMock()
        
        # Mock search response
        mock_match = MagicMock()
        mock_match.id = "chunk1"
        mock_match.score = 0.85
        mock_match.metadata = {
            "content": "Similar chunk content",
            "document_id": "doc1",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100
        }
        
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response
        
        vector_store.index = mock_index
        
        query_embedding = [0.1, 0.2, 0.3]
        results = await vector_store.search_similar_chunks(query_embedding, top_k=5)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chunk1"
        assert results[0].similarity_score == 0.85
        assert results[0].content == "Similar chunk content"
        assert results[0].document_metadata["document_id"] == "doc1"
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks_with_document_filter(self, vector_store):
        """Test searching chunks with document ID filter"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])
        vector_store.index = mock_index
        
        query_embedding = [0.1, 0.2, 0.3]
        await vector_store.search_similar_chunks(
            query_embedding, 
            document_id="doc123", 
            top_k=5
        )
        
        # Check that filter was applied
        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"document_id": {"$eq": "doc123"}}
    
    @pytest.mark.asyncio
    async def test_get_index_stats(self, vector_store):
        """Test getting index statistics"""
        vector_store._initialized = True
        mock_index = MagicMock()
        
        # Mock stats response
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 1000
        mock_stats.dimension = 768
        mock_stats.index_fullness = 0.1
        mock_stats.namespaces = {}
        
        mock_index.describe_index_stats.return_value = mock_stats
        vector_store.index = mock_index
        
        stats = await vector_store.get_index_stats()
        
        assert stats["total_vector_count"] == 1000
        assert stats["dimension"] == 768
        assert stats["index_fullness"] == 0.1
        assert stats["namespaces"] == {}
    
    @pytest.mark.asyncio
    async def test_get_index_stats_failure(self, vector_store):
        """Test index stats failure"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.describe_index_stats.side_effect = Exception("Stats failed")
        vector_store.index = mock_index
        
        stats = await vector_store.get_index_stats()
        
        assert stats == {}
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, vector_store):
        """Test successful health check"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = MagicMock()
        vector_store.index = mock_index
        
        is_healthy = await vector_store.health_check()
        
        assert is_healthy == True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, vector_store):
        """Test health check failure"""
        vector_store._initialized = True
        mock_index = MagicMock()
        mock_index.describe_index_stats.side_effect = Exception("Health check failed")
        vector_store.index = mock_index
        
        is_healthy = await vector_store.health_check()
        
        assert is_healthy == False
    
    def test_vector_class(self):
        """Test Vector data class"""
        vector = Vector("test-id", [0.1, 0.2, 0.3], {"key": "value"})
        
        assert vector.id == "test-id"
        assert vector.values == [0.1, 0.2, 0.3]
        assert vector.metadata == {"key": "value"}
    
    def test_vector_class_default_metadata(self):
        """Test Vector with default metadata"""
        vector = Vector("test-id", [0.1, 0.2, 0.3])
        
        assert vector.metadata == {}
    
    def test_match_class(self):
        """Test Match data class"""
        match = Match("match-id", 0.95, [0.1, 0.2], {"content": "test"})
        
        assert match.id == "match-id"
        assert match.score == 0.95
        assert match.values == [0.1, 0.2]
        assert match.metadata == {"content": "test"}
    
    def test_match_class_defaults(self):
        """Test Match with default values"""
        match = Match("match-id", 0.95)
        
        assert match.values == []
        assert match.metadata == {}