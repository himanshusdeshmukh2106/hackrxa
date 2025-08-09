"""
Unit tests for search engine service
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.services.search_engine import SearchEngine
from app.schemas.models import SearchResult, QueryIntent, Entity


class TestSearchEngine:
    """Test SearchEngine functionality"""
    
    @pytest.fixture
    def search_engine(self):
        """Create SearchEngine instance"""
        return SearchEngine()
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results"""
        return [
            SearchResult(
                chunk_id="chunk1",
                content="This policy covers dental treatment and surgery procedures.",
                similarity_score=0.85,
                document_metadata={"document_id": "doc1", "chunk_index": 0},
                relevance_explanation="High similarity match"
            ),
            SearchResult(
                chunk_id="chunk2",
                content="Premium payments are due monthly with a grace period.",
                similarity_score=0.75,
                document_metadata={"document_id": "doc1", "chunk_index": 1},
                relevance_explanation="Medium similarity match"
            ),
            SearchResult(
                chunk_id="chunk3",
                content="Exclusions include pre-existing conditions and cosmetic procedures.",
                similarity_score=0.65,
                document_metadata={"document_id": "doc1", "chunk_index": 2},
                relevance_explanation="Lower similarity match"
            )
        ]
    
    @pytest.fixture
    def sample_query_intent(self):
        """Create sample query intent"""
        return QueryIntent(
            original_query="Does this policy cover dental surgery?",
            intent_type="coverage",
            entities=[
                Entity(text="dental", label="treatment", start=25, end=31, confidence=0.8),
                Entity(text="surgery", label="surgery", start=32, end=39, confidence=0.9)
            ],
            confidence=0.85
        )
    
    @pytest.mark.asyncio
    @patch('app.services.search_engine.embedding_service.generate_single_embedding', new_callable=AsyncMock)
    @patch('app.services.search_engine.vector_store.search_similar_chunks', new_callable=AsyncMock)
    async def test_semantic_search_success(self, mock_search_similar_chunks, mock_generate_single_embedding, search_engine, sample_search_results):
        """Test successful semantic search"""
        # Mock embedding generation
        mock_generate_single_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector store search
        mock_search_similar_chunks.return_value = sample_search_results
        
        query = "Does this policy cover dental treatment?"
        results = await search_engine.semantic_search(query)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.similarity_score >= search_engine.similarity_threshold for r in results)
        
        mock_generate_single_embedding.assert_called_once_with(query)
        mock_search_similar_chunks.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.search_engine.embedding_service.generate_single_embedding', new_callable=AsyncMock)
    async def test_semantic_search_no_embedding(self, mock_generate_single_embedding, search_engine):
        """Test semantic search when embedding generation fails"""
        mock_generate_single_embedding.return_value = []
        
        results = await search_engine.semantic_search("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    @patch('app.services.search_engine.embedding_service.generate_single_embedding', new_callable=AsyncMock)
    @patch('app.services.search_engine.vector_store.search_similar_chunks', new_callable=AsyncMock)
    async def test_semantic_search_with_filters(self, mock_search_similar_chunks, mock_generate_single_embedding, search_engine):
        """Test semantic search with metadata filters"""
        mock_generate_single_embedding.return_value = [0.1, 0.2, 0.3]
        mock_search_similar_chunks.return_value = []
        
        filters = {"document_id": "doc123"}
        await search_engine.semantic_search("test query", filters=filters)
        
        # Check that document_id filter was passed
        call_args = mock_search_similar_chunks.call_args
        assert call_args[1]['document_id'] == "doc123"
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, search_engine):
        """Test keyword search functionality"""
        query = "premium payment grace period"
        
        # Mock the PostgreSQL search (since it's not fully implemented)
        with patch.object(search_engine, '_postgresql_fulltext_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            
            results = await search_engine.keyword_search(query)
            
            assert isinstance(results, list)
            mock_search.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.search_engine.asyncio.create_task')
    async def test_hybrid_search(self, mock_create_task, search_engine, sample_search_results, sample_query_intent):
        """Test hybrid search combining semantic and keyword approaches"""
        # Mock the async tasks
        semantic_task = AsyncMock()
        semantic_task.return_value = sample_search_results[:2]
        keyword_task = AsyncMock()
        keyword_task.return_value = sample_search_results[2:]
        
        mock_create_task.side_effect = [semantic_task, keyword_task]
        
        # Mock asyncio.gather
        with patch('app.services.search_engine.asyncio.gather', new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = (sample_search_results[:2], sample_search_results[2:])
            
            results = await search_engine.hybrid_search("test query", sample_query_intent)
            
            assert isinstance(results, list)
            assert len(results) > 0
    
    def test_extract_keywords(self, search_engine):
        """Test keyword extraction from query"""
        query = "What is the premium for dental coverage?"
        
        keywords = search_engine._extract_keywords(query)
        
        assert "premium" in keywords
        assert "dental" in keywords
        assert "coverage" in keywords
        # Stop words should be removed
        assert "what" not in keywords
        assert "the" not in keywords
        assert "for" not in keywords
    
    def test_extract_keywords_empty_query(self, search_engine):
        """Test keyword extraction from empty query"""
        keywords = search_engine._extract_keywords("")
        assert keywords == []
    
    def test_extract_keywords_stop_words_only(self, search_engine):
        """Test keyword extraction with only stop words"""
        query = "what is the and or but"
        keywords = search_engine._extract_keywords(query)
        assert keywords == []
    
    def test_combine_search_results(self, search_engine, sample_query_intent):
        """Test combining semantic and keyword search results"""
        semantic_results = [
            SearchResult(
                chunk_id="chunk1",
                content="Semantic result",
                similarity_score=0.8,
                document_metadata={},
                relevance_explanation=""
            )
        ]
        
        keyword_results = [
            SearchResult(
                chunk_id="chunk2",
                content="Keyword result",
                similarity_score=0.7,
                document_metadata={},
                relevance_explanation=""
            ),
            SearchResult(
                chunk_id="chunk1",  # Same chunk as semantic
                content="Overlapping result",
                similarity_score=0.6,
                document_metadata={},
                relevance_explanation=""
            )
        ]
        
        combined = search_engine._combine_search_results(
            semantic_results, keyword_results, "test query", sample_query_intent
        )
        
        # Should have 2 unique chunks
        assert len(combined) == 2
        
        # Check that overlapping chunk has combined score
        chunk1 = next(r for r in combined if r.chunk_id == "chunk1")
        assert chunk1.similarity_score > 0.7  # Should be higher than original semantic score
        assert "Semantic similarity" in chunk1.relevance_explanation
        assert "Keyword match" in chunk1.relevance_explanation
    
    @pytest.mark.asyncio
    async def test_rank_results(self, search_engine, sample_search_results, sample_query_intent):
        """Test result ranking functionality"""
        ranked_results = await search_engine.rank_results(
            sample_search_results, "dental surgery coverage", sample_query_intent
        )
        
        assert len(ranked_results) == len(sample_search_results)
        
        # Results should be sorted by similarity score (descending)
        scores = [r.similarity_score for r in ranked_results]
        assert scores == sorted(scores, reverse=True)
        
        # Check that explanations were updated
        for result in ranked_results:
            assert "final score" in result.relevance_explanation
    
    @pytest.mark.asyncio
    async def test_rank_results_empty_list(self, search_engine):
        """Test ranking empty results list"""
        ranked_results = await search_engine.rank_results([], "test query")
        assert ranked_results == []
    
    def test_calculate_length_factor(self, search_engine):
        """Test content length factor calculation"""
        # Optimal length should get high score
        assert search_engine._calculate_length_factor(500) > 0.9
        
        # Too short should get lower score
        assert search_engine._calculate_length_factor(30) == 0.5
        
        # Too long should get medium score
        assert search_engine._calculate_length_factor(3000) == 0.7
        
        # Reasonable length should get good score
        assert search_engine._calculate_length_factor(800) >= 0.7
    
    def test_calculate_tf_factor(self, search_engine):
        """Test term frequency factor calculation"""
        content = "This policy covers dental treatment and dental surgery procedures."
        query = "dental surgery"
        
        tf_factor = search_engine._calculate_tf_factor(content, query)
        
        assert tf_factor > 0.0
        assert tf_factor <= 1.0
    
    def test_calculate_tf_factor_no_matches(self, search_engine):
        """Test TF factor with no matching terms"""
        content = "This is about something completely different."
        query = "dental surgery"
        
        tf_factor = search_engine._calculate_tf_factor(content, query)
        
        assert tf_factor == 0.0
    
    def test_calculate_intent_factor_coverage(self, search_engine):
        """Test intent factor calculation for coverage queries"""
        result = SearchResult(
            chunk_id="test",
            content="This policy covers dental treatment and includes benefits.",
            similarity_score=0.8,
            document_metadata={}
        )
        
        query_intent = QueryIntent(
            original_query="Does this cover dental?",
            intent_type="coverage",
            entities=[Entity(text="dental", label="treatment", start=0, end=6, confidence=0.8)],
            confidence=0.8
        )
        
        intent_factor = search_engine._calculate_intent_factor(result, query_intent)
        
        assert intent_factor > 0.5  # Should be boosted for coverage content
    
    def test_calculate_intent_factor_no_intent(self, search_engine):
        """Test intent factor with no query intent"""
        result = SearchResult(
            chunk_id="test",
            content="Some content",
            similarity_score=0.8,
            document_metadata={}
        )
        
        intent_factor = search_engine._calculate_intent_factor(result, None)
        
        assert intent_factor == 0.5  # Default value
    
    def test_calculate_position_factor(self, search_engine):
        """Test position-based factor calculation"""
        # First chunk should get highest score
        result1 = SearchResult(
            chunk_id="test1",
            content="Content",
            similarity_score=0.8,
            document_metadata={"chunk_index": 0}
        )
        assert search_engine._calculate_position_factor(result1) == 1.0
        
        # Early chunk should get high score
        result2 = SearchResult(
            chunk_id="test2",
            content="Content",
            similarity_score=0.8,
            document_metadata={"chunk_index": 3}
        )
        assert search_engine._calculate_position_factor(result2) == 0.9
        
        # Later chunk should get lower score
        result3 = SearchResult(
            chunk_id="test3",
            content="Content",
            similarity_score=0.8,
            document_metadata={"chunk_index": 15}
        )
        assert search_engine._calculate_position_factor(result3) == 0.7
    
    @pytest.mark.asyncio
    async def test_get_search_suggestions(self, search_engine):
        """Test search suggestions functionality"""
        suggestions = await search_engine.get_search_suggestions("coverage")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        assert any("coverage" in suggestion for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_get_search_suggestions_empty(self, search_engine):
        """Test search suggestions with empty input"""
        suggestions = await search_engine.get_search_suggestions("")
        
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_explain_search_results(self, search_engine, sample_search_results):
        """Test search results explanation"""
        explanation = await search_engine.explain_search_results(
            sample_search_results, "test query"
        )
        
        assert "total_results" in explanation
        assert "score_statistics" in explanation
        assert "relevance_distribution" in explanation
        assert "search_method" in explanation
        assert explanation["total_results"] == len(sample_search_results)
    
    @pytest.mark.asyncio
    async def test_explain_search_results_empty(self, search_engine):
        """Test explanation for empty search results"""
        explanation = await search_engine.explain_search_results([], "test query")
        
        assert explanation["total_results"] == 0
        assert "No relevant results found" in explanation["explanation"]
        assert "suggestions" in explanation
