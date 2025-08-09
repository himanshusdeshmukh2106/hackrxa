"""
Unit tests for response generator service
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.response_generator import ResponseGenerator
from app.services.llm_service import LLMResponse
from app.core.exceptions import ResponseGenerationError
from app.schemas.models import SearchResult, QueryIntent, Entity, Answer


class TestResponseGenerator:
    """Test ResponseGenerator functionality"""
    
    @pytest.fixture
    def response_generator(self):
        """Create ResponseGenerator instance"""
        return ResponseGenerator()
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results"""
        return [
            SearchResult(
                chunk_id="chunk1",
                content="This policy covers dental treatment including surgery and routine care. The coverage includes up to $5000 per year.",
                similarity_score=0.85,
                document_metadata={"document_id": "policy1", "chunk_index": 0},
                relevance_explanation="High similarity match"
            ),
            SearchResult(
                chunk_id="chunk2",
                content="Waiting period for dental coverage is 6 months from policy inception date.",
                similarity_score=0.75,
                document_metadata={"document_id": "policy1", "chunk_index": 1},
                relevance_explanation="Medium similarity match"
            ),
            SearchResult(
                chunk_id="chunk3",
                content="Pre-existing dental conditions are excluded from coverage for the first 12 months.",
                similarity_score=0.65,
                document_metadata={"document_id": "policy1", "chunk_index": 2},
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
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response"""
        return LLMResponse(
            content="Yes, this policy covers dental surgery up to $5000 per year. However, there is a 6-month waiting period from policy inception.",
            tokens_used=150,
            processing_time_ms=800,
            model_name="gemini-2.0-flash-exp",
            confidence=0.8,
            metadata={"prompt_template": "coverage_analysis"}
        )
    
    @pytest.mark.asyncio
    @patch('app.services.response_generator.llm_service')
    async def test_generate_answer_success(self, mock_llm_service, response_generator, sample_search_results, sample_query_intent, mock_llm_response):
        """Test successful answer generation"""
        mock_llm_service.generate_response.return_value = mock_llm_response
        
        query = "Does this policy cover dental surgery?"
        answer = await response_generator.generate_answer(sample_search_results, query, sample_query_intent)
        
        assert isinstance(answer, Answer)
        assert answer.question == query
        assert answer.answer == mock_llm_response.content
        assert 0.0 <= answer.confidence <= 1.0
        assert len(answer.source_chunks) > 0
        assert answer.reasoning is not None
        assert "llm_tokens_used" in answer.metadata
        
        mock_llm_service.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_context(self, response_generator):
        """Test answer generation with no context"""
        query = "What is the coverage?"
        answer = await response_generator.generate_answer([], query)
        
        assert isinstance(answer, Answer)
        assert answer.confidence == 0.0
        assert len(answer.source_chunks) == 0
        assert "don't have sufficient information" in answer.answer
        assert answer.metadata.get("no_context") == True
    
    @pytest.mark.asyncio
    @patch('app.services.response_generator.llm_service')
    async def test_generate_answer_llm_failure(self, mock_llm_service, response_generator, sample_search_results):
        """Test answer generation when LLM fails"""
        mock_llm_service.generate_response.side_effect = Exception("LLM service error")
        
        with pytest.raises(ResponseGenerationError) as exc_info:
            await response_generator.generate_answer(sample_search_results, "test query")
        
        assert "Failed to generate answer" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.services.response_generator.llm_service')
    async def test_generate_batch_answers(self, mock_llm_service, response_generator, sample_search_results, mock_llm_response):
        """Test batch answer generation"""
        # Mock batch processing
        mock_llm_service.batch_process.return_value = [
            mock_llm_response,
            LLMResponse("Answer 2", 120, 600, "gemini", 0.7)
        ]
        
        questions = ["Question 1?", "Question 2?"]
        answers = await response_generator.generate_batch_answers(questions, sample_search_results)
        
        assert len(answers) == 2
        assert all(isinstance(answer, Answer) for answer in answers)
        assert all(answer.metadata.get("batch_processed") for answer in answers)
        assert answers[0].metadata.get("batch_index") == 0
        assert answers[1].metadata.get("batch_index") == 1
        
        mock_llm_service.batch_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_batch_answers_no_context(self, response_generator):
        """Test batch answer generation with no context"""
        questions = ["Question 1?", "Question 2?"]
        answers = await response_generator.generate_batch_answers(questions, [])
        
        assert len(answers) == 2
        assert all(answer.confidence == 0.0 for answer in answers)
        assert all("don't have sufficient information" in answer.answer for answer in answers)
    
    def test_select_context_chunks(self, response_generator, sample_search_results):
        """Test context chunk selection"""
        selected = response_generator._select_context_chunks(sample_search_results)
        
        assert len(selected) <= response_generator.max_context_chunks
        assert len(selected) > 0
        
        # Should be sorted by similarity score (descending)
        scores = [chunk.similarity_score for chunk in selected]
        assert scores == sorted(scores, reverse=True)
    
    def test_select_context_chunks_empty(self, response_generator):
        """Test context chunk selection with empty input"""
        selected = response_generator._select_context_chunks([])
        assert selected == []
    
    def test_select_context_chunks_size_limit(self, response_generator):
        """Test context chunk selection respects size limits"""
        # Create chunks with very long content
        large_chunks = []
        for i in range(10):
            chunk = SearchResult(
                chunk_id=f"chunk{i}",
                content="x" * 2000,  # Large content
                similarity_score=0.9 - i * 0.1,
                document_metadata={}
            )
            large_chunks.append(chunk)
        
        selected = response_generator._select_context_chunks(large_chunks)
        
        # Should respect context window size limit
        total_size = sum(len(chunk.content) for chunk in selected)
        assert total_size <= response_generator.context_window_size
    
    def test_build_context_text(self, response_generator, sample_search_results):
        """Test context text building"""
        context_text = response_generator._build_context_text(sample_search_results)
        
        assert isinstance(context_text, str)
        assert len(context_text) > 0
        
        # Should contain chunk headers
        assert "[Context Chunk 1]" in context_text
        assert "[Context Chunk 2]" in context_text
        
        # Should contain relevance scores
        assert "[Relevance:" in context_text
        
        # Should contain actual content
        assert "dental treatment" in context_text
        assert "waiting period" in context_text
    
    def test_build_context_text_empty(self, response_generator):
        """Test context text building with empty input"""
        context_text = response_generator._build_context_text([])
        assert context_text == ""
    
    def test_calculate_final_confidence_high_quality(self, response_generator, sample_search_results, mock_llm_response):
        """Test confidence calculation with high-quality context and response"""
        confidence = response_generator._calculate_final_confidence(mock_llm_response, sample_search_results)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high
    
    def test_calculate_final_confidence_no_context(self, response_generator, mock_llm_response):
        """Test confidence calculation with no context"""
        confidence = response_generator._calculate_final_confidence(mock_llm_response, [])
        
        assert confidence < 0.5  # Should be lower without context
    
    def test_calculate_final_confidence_uncertain_response(self, response_generator, sample_search_results):
        """Test confidence calculation with uncertain response"""
        uncertain_response = LLMResponse(
            content="I don't know if this policy covers that. The information is unclear.",
            tokens_used=50,
            processing_time_ms=400,
            model_name="gemini",
            confidence=0.3
        )
        
        confidence = response_generator._calculate_final_confidence(uncertain_response, sample_search_results)
        
        assert confidence < 0.5  # Should be penalized for uncertainty
    
    def test_calculate_final_confidence_specific_response(self, response_generator, sample_search_results):
        """Test confidence calculation with specific, detailed response"""
        specific_response = LLMResponse(
            content="Yes, this policy covers dental surgery with a maximum benefit of $5,000 per year. There is a 6-month waiting period, and pre-existing conditions are excluded for 12 months. The coverage includes both routine and emergency dental procedures.",
            tokens_used=200,
            processing_time_ms=1000,
            model_name="gemini",
            confidence=0.8
        )
        
        confidence = response_generator._calculate_final_confidence(specific_response, sample_search_results)
        
        assert confidence > 0.7  # Should be boosted for specificity
    
    def test_generate_reasoning(self, response_generator, sample_search_results, mock_llm_response, sample_query_intent):
        """Test reasoning generation"""
        reasoning = response_generator._generate_reasoning(sample_search_results, mock_llm_response, sample_query_intent)
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        
        # Should mention number of sources
        assert "3 relevant document sections" in reasoning
        
        # Should mention query intent
        assert "coverage-related" in reasoning
        
        # Should mention key terms
        assert "dental" in reasoning
        
        # Should mention LLM details
        assert "gemini" in reasoning.lower()
        assert "150 tokens" in reasoning
    
    def test_generate_reasoning_no_context(self, response_generator, mock_llm_response):
        """Test reasoning generation with no context"""
        reasoning = response_generator._generate_reasoning([], mock_llm_response, None)
        
        assert isinstance(reasoning, str)
        # Should still contain LLM processing info
        assert "gemini" in reasoning.lower()
    
    def test_create_no_context_answer(self, response_generator):
        """Test creation of no-context answer"""
        query = "What is the coverage?"
        answer = response_generator._create_no_context_answer(query)
        
        assert isinstance(answer, Answer)
        assert answer.question == query
        assert answer.confidence == 0.0
        assert len(answer.source_chunks) == 0
        assert "don't have sufficient information" in answer.answer
        assert answer.metadata.get("no_context") == True
    
    @pytest.mark.asyncio
    async def test_explain_reasoning(self, response_generator):
        """Test reasoning explanation"""
        answer = Answer(
            question="Test question",
            answer="Test answer with specific details including $1000 amount.",
            confidence=0.75,
            source_chunks=["chunk1", "chunk2"],
            reasoning="Test reasoning",
            metadata={
                "llm_tokens_used": 150,
                "llm_processing_time_ms": 800,
                "context_chunks_count": 2,
                "avg_chunk_similarity": 0.8
            }
        )
        
        explanation = await response_generator.explain_reasoning(answer)
        
        assert "answer_confidence" in explanation
        assert "confidence_level" in explanation
        assert "source_analysis" in explanation
        assert "processing_details" in explanation
        assert "quality_indicators" in explanation
        assert explanation["answer_confidence"] == 0.75
        assert explanation["source_analysis"]["total_sources"] == 2
    
    @pytest.mark.asyncio
    async def test_explain_reasoning_low_confidence(self, response_generator):
        """Test reasoning explanation for low confidence answer"""
        answer = Answer(
            question="Test question",
            answer="Uncertain answer",
            confidence=0.2,
            source_chunks=[],
            reasoning="Low confidence reasoning",
            metadata={}
        )
        
        explanation = await response_generator.explain_reasoning(answer)
        
        assert "improvement_suggestions" in explanation
        assert len(explanation["improvement_suggestions"]) > 0
    
    def test_get_confidence_level(self, response_generator):
        """Test confidence level categorization"""
        assert response_generator._get_confidence_level(0.9) == "Very High"
        assert response_generator._get_confidence_level(0.7) == "High"
        assert response_generator._get_confidence_level(0.5) == "Medium"
        assert response_generator._get_confidence_level(0.3) == "Low"
        assert response_generator._get_confidence_level(0.1) == "Very Low"
    
    def test_analyze_answer_quality_detailed(self, response_generator):
        """Test answer quality analysis for detailed answer"""
        answer = Answer(
            question="Test",
            answer="This is a detailed answer with specific information including $5000 coverage amount, 6-month waiting period, and various conditions that must be met for eligibility.",
            confidence=0.8,
            source_chunks=["chunk1"],
            reasoning="Test reasoning",
            metadata={"avg_chunk_similarity": 0.85}
        )
        
        quality = response_generator._analyze_answer_quality(answer)
        
        assert quality["length_category"] == "detailed"
        assert quality["specificity"]["contains_numbers"] == True
        assert quality["specificity"]["contains_monetary_info"] == True
        assert quality["specificity"]["contains_timeframes"] == True
        assert quality["specificity"]["contains_conditions"] == True
        assert quality["uncertainty_level"] == "low"
    
    def test_analyze_answer_quality_brief(self, response_generator):
        """Test answer quality analysis for brief answer"""
        answer = Answer(
            question="Test",
            answer="Maybe yes.",
            confidence=0.3,
            source_chunks=[],
            reasoning="Brief reasoning",
            metadata={}
        )
        
        quality = response_generator._analyze_answer_quality(answer)
        
        assert quality["length_category"] == "brief"
        assert quality["specificity"]["contains_numbers"] == False
        assert quality["uncertainty_level"] == "medium"  # "maybe" is uncertain
    
    @pytest.mark.asyncio
    async def test_validate_answer_accuracy(self, response_generator, sample_search_results):
        """Test answer accuracy validation"""
        answer = Answer(
            question="Test",
            answer="This policy covers dental surgery up to $5000 per year. There is a 6-month waiting period.",
            confidence=0.8,
            source_chunks=["chunk1", "chunk2"],
            reasoning="Test reasoning",
            metadata={}
        )
        
        validation = await response_generator.validate_answer_accuracy(answer, sample_search_results)
        
        assert "overall_accuracy" in validation
        assert "context_alignment" in validation
        assert "factual_consistency" in validation
        assert validation["overall_accuracy"] in ["high", "medium", "low", "unknown"]
        assert 0.0 <= validation["context_alignment"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_answer_accuracy_inconsistent(self, response_generator, sample_search_results):
        """Test answer accuracy validation with inconsistent answer"""
        answer = Answer(
            question="Test",
            answer="This policy covers unlimited dental surgery with no waiting period and no exclusions.",
            confidence=0.8,
            source_chunks=["chunk1"],
            reasoning="Test reasoning",
            metadata={}
        )
        
        validation = await response_generator.validate_answer_accuracy(answer, sample_search_results)
        
        # Should detect inconsistencies
        assert validation["overall_accuracy"] in ["low", "medium"]
        assert len(validation.get("issues_found", [])) > 0
    
    def test_extract_key_claims(self, response_generator):
        """Test key claims extraction"""
        answer_text = "This policy covers dental surgery. The maximum benefit is $5000 per year. There is a 6-month waiting period. Pre-existing conditions are excluded."
        
        claims = response_generator._extract_key_claims(answer_text)
        
        assert len(claims) > 0
        assert len(claims) <= 5  # Should be limited
        assert any("dental surgery" in claim for claim in claims)
        assert any("$5000" in claim for claim in claims)
    
    def test_validate_claim_against_context(self, response_generator):
        """Test individual claim validation"""
        claim = "This policy covers dental surgery up to $5000"
        context = "This policy covers dental treatment including surgery and routine care. The coverage includes up to $5000 per year."
        
        is_valid = response_generator._validate_claim_against_context(claim, context)
        
        assert is_valid == True
    
    def test_validate_claim_against_context_invalid(self, response_generator):
        """Test individual claim validation with invalid claim"""
        claim = "This policy covers unlimited cosmetic surgery with no restrictions"
        context = "This policy covers dental treatment including surgery and routine care. The coverage includes up to $5000 per year."
        
        is_valid = response_generator._validate_claim_against_context(claim, context)
        
        assert is_valid == False