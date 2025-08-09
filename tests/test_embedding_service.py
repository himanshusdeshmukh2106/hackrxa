"""
Unit tests for embedding service
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.embedding_service import EmbeddingService
from app.core.exceptions import EmbeddingGenerationError
from app.schemas.models import TextChunk


class TestEmbeddingService:
    """Test EmbeddingService functionality"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create EmbeddingService instance"""
        return EmbeddingService("test-model")
    
    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model"""
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        model.device = "cpu"
        model.max_seq_length = 512
        return model
    
    @pytest.mark.asyncio
    @patch('app.services.embedding_service.SentenceTransformer')
    async def test_initialize_success(self, mock_transformer, embedding_service):
        """Test successful model initialization"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        await embedding_service.initialize()
        
        assert embedding_service._model_loaded == True
        assert embedding_service.model is not None
        mock_transformer.assert_called_once_with("test-model")
    
    @pytest.mark.asyncio
    @patch('app.services.embedding_service.SentenceTransformer')
    async def test_initialize_failure(self, mock_transformer, embedding_service):
        """Test model initialization failure"""
        mock_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await embedding_service.initialize()
        
        assert "Model initialization failed" in str(exc_info.value)
        assert embedding_service._model_loaded == False
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service, mock_model):
        """Test successful embedding generation"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        
        texts = ["Hello world", "Test text"]
        embeddings = await embedding_service.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, embedding_service):
        """Test embedding generation with empty list"""
        embedding_service._model_loaded = True
        
        embeddings = await embedding_service.generate_embeddings([])
        
        assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_invalid_input(self, embedding_service):
        """Test embedding generation with invalid input"""
        embedding_service._model_loaded = True
        
        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await embedding_service.generate_embeddings([123, "valid text"])
        
        assert "not a string" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self, embedding_service, mock_model):
        """Test embedding generation with empty text"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        
        # Mock model to handle empty text placeholder
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        texts = ["", "  ", "valid text"]
        embeddings = await embedding_service.generate_embeddings(texts)
        
        # Should replace empty texts with placeholder
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert "[EMPTY]" in call_args
        assert "valid text" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, embedding_service, mock_model):
        """Test single embedding generation"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        
        # Mock to return single embedding
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        embedding = await embedding_service.generate_single_embedding("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_embed_text_chunks(self, embedding_service, mock_model):
        """Test embedding text chunks"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        embedding_service.model_name = "test-model"
        
        # Create test chunks
        chunks = [
            TextChunk(document_id="doc1", content="First chunk", chunk_index=0),
            TextChunk(document_id="doc1", content="Second chunk", chunk_index=1)
        ]
        
        result_chunks = await embedding_service.embed_text_chunks(chunks)
        
        assert len(result_chunks) == 2
        assert result_chunks[0].embedding == [0.1, 0.2, 0.3]
        assert result_chunks[1].embedding == [0.4, 0.5, 0.6]
        assert result_chunks[0].metadata["embedding_model"] == "test-model"
        assert result_chunks[0].metadata["embedding_dimension"] == 3
    
    @pytest.mark.asyncio
    async def test_embed_text_chunks_empty(self, embedding_service):
        """Test embedding empty chunks list"""
        result = await embedding_service.embed_text_chunks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_batch_process_chunks(self, embedding_service, mock_model):
        """Test batch processing of chunks"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        embedding_service.model_name = "test-model"
        
        # Create test chunks
        chunks = [
            TextChunk(document_id="doc1", content=f"Chunk {i}", chunk_index=i)
            for i in range(5)
        ]
        
        # Mock encode to return appropriate number of embeddings
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]
        ])
        
        result_chunks = await embedding_service.batch_process_chunks(chunks, batch_size=2)
        
        assert len(result_chunks) == 5
        assert all(chunk.embedding is not None for chunk in result_chunks)
    
    def test_validate_embedding_dimension(self, embedding_service):
        """Test embedding dimension validation"""
        embedding_service.embedding_dimension = 3
        
        assert embedding_service.validate_embedding_dimension([0.1, 0.2, 0.3]) == True
        assert embedding_service.validate_embedding_dimension([0.1, 0.2]) == False
        assert embedding_service.validate_embedding_dimension([0.1, 0.2, 0.3, 0.4]) == False
    
    def test_calculate_similarity(self, embedding_service):
        """Test cosine similarity calculation"""
        # Identical vectors should have similarity close to 1
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = embedding_service.calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01
        
        # Orthogonal vectors should have similarity close to 0.5 (normalized)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = embedding_service.calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.5) < 0.01
        
        # Opposite vectors should have similarity close to 0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = embedding_service.calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.01
    
    def test_calculate_similarity_zero_vectors(self, embedding_service):
        """Test similarity calculation with zero vectors"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = embedding_service.calculate_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_calculate_similarity_error_handling(self, embedding_service):
        """Test similarity calculation error handling"""
        # Test with invalid input
        similarity = embedding_service.calculate_similarity([], [1.0, 2.0])
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, embedding_service, mock_model):
        """Test getting model information"""
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        embedding_service.model_name = "test-model"
        embedding_service.embedding_dimension = 768
        
        info = await embedding_service.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["embedding_dimension"] == 768
        assert info["loaded"] == True
        assert info["device"] == "cpu"
    
    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    async def test_cleanup(self, mock_empty_cache, mock_cuda_available, embedding_service, mock_model):
        """Test model cleanup"""
        mock_cuda_available.return_value = True
        embedding_service.model = mock_model
        embedding_service._model_loaded = True
        
        await embedding_service.cleanup()
        
        assert embedding_service.model is None
        assert embedding_service._model_loaded == False
        mock_empty_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_initialize_on_generate(self, embedding_service):
        """Test that model auto-initializes when generating embeddings"""
        with patch.object(embedding_service, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            embedding_service._model_loaded = False
            
            # Mock the model after initialization
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            
            async def set_model():
                embedding_service.model = mock_model
                embedding_service._model_loaded = True
            
            mock_init.side_effect = set_model
            
            embeddings = await embedding_service.generate_embeddings(["test"])
            
            mock_init.assert_called_once()
            assert len(embeddings) == 1