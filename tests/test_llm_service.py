"""
Unit tests for LLM service
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.llm_service import LLMService, LLMRequest, LLMResponse
from app.core.exceptions import LLMServiceError
from app.schemas.models import QueryIntent, Entity


class TestLLMService:
    """Test LLMService functionality"""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLMService instance"""
        return LLMService()
    
    @pytest.fixture
    def mock_genai_model(self):
        """Create mock Gemini model"""
        mock_response = MagicMock()
        mock_response.text = "This is a test response from Gemini."
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.candidates[0].safety_ratings = []
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        
        return mock_model, mock_response
    
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
    @patch('app.services.llm_service.genai')
    async def test_initialize_success(self, mock_genai, llm_service):
        """Test successful LLM service initialization"""
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        await llm_service.initialize()
        
        assert llm_service._initialized == True
        assert llm_service.model is not None
        mock_genai.configure.assert_called_once()
        mock_genai.GenerativeModel.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.llm_service.genai')
    async def test_initialize_failure(self, mock_genai, llm_service):
        """Test LLM service initialization failure"""
        mock_genai.configure.side_effect = Exception("API key invalid")
        
        with pytest.raises(LLMServiceError) as exc_info:
            await llm_service.initialize()
        
        assert "Gemini initialization failed" in str(exc_info.value)
        assert llm_service._initialized == False
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, llm_service, mock_genai_model, sample_query_intent):
        """Test successful response generation"""
        mock_model, mock_response = mock_genai_model
        llm_service.model = mock_model
        llm_service._initialized = True
        
        context = "This policy covers dental treatments including surgery."
        query = "Does this cover dental surgery?"
        
        response = await llm_service.generate_response(context, query, sample_query_intent)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from Gemini."
        assert response.tokens_used > 0
        assert response.processing_time_ms > 0
        assert response.model_name == llm_service.model_name
        assert 0.0 <= response.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_response_auto_initialize(self, llm_service):
        """Test that response generation auto-initializes the service"""
        llm_service._initialized = False
        
        with patch.object(llm_service, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            
            # Mock the model after initialization
            mock_model, mock_response = self.mock_genai_model()
            
            async def set_model():
                llm_service.model = mock_model
                llm_service._initialized = True
            
            mock_init.side_effect = set_model
            
            response = await llm_service.generate_response("context", "query")
            
            mock_init.assert_called_once()
            assert isinstance(response, LLMResponse)
    
    def test_select_prompt_template_coverage(self, llm_service, sample_query_intent):
        """Test prompt template selection for coverage queries"""
        template = llm_service._select_prompt_template(sample_query_intent)
        
        assert "coverage" in template.lower()
        assert "COVERAGE ANALYSIS:" in template
    
    def test_select_prompt_template_definition(self, llm_service):
        """Test prompt template selection for definition queries"""
        intent = QueryIntent(
            original_query="What does 'deductible' mean?",
            intent_type="definition",
            entities=[],
            confidence=0.8
        )
        
        template = llm_service._select_prompt_template(intent)
        
        assert "definition" in template.lower()
        assert "DEFINITION:" in template
    
    def test_select_prompt_template_default(self, llm_service):
        """Test default prompt template selection"""
        template = llm_service._select_prompt_template(None)
        
        assert "ANSWER:" in template
        assert "document analyst" in template.lower()
    
    def test_optimize_context_short(self, llm_service):
        """Test context optimization with short context"""
        short_context = "This is a short context."
        
        optimized = llm_service._optimize_context(short_context)
        
        assert optimized == short_context
    
    def test_optimize_context_long(self, llm_service):
        """Test context optimization with long context"""
        # Create context longer than max_context_length
        long_context = "This is a sentence. " * 100000  # Much longer than limit
        
        optimized = llm_service._optimize_context(long_context)
        
        assert len(optimized) <= llm_service.max_context_length
        assert optimized.endswith('.')  # Should end at sentence boundary
    
    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, llm_service, mock_genai_model):
        """Test successful generation with retry logic"""
        mock_model, mock_response = mock_genai_model
        llm_service.model = mock_model
        
        response = await llm_service._generate_with_retry("test prompt")
        
        assert response.text == "This is a test response from Gemini."
        mock_model.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_retry_failure(self, llm_service):
        """Test generation failure with retry logic"""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API error")
        llm_service.model = mock_model
        
        with pytest.raises(LLMServiceError) as exc_info:
            await llm_service._generate_with_retry("test prompt", max_retries=2)
        
        assert "Failed after 2 attempts" in str(exc_info.value)
        assert mock_model.generate_content.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_with_retry_empty_response(self, llm_service):
        """Test handling of empty response"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""  # Empty response
        mock_model.generate_content.return_value = mock_response
        llm_service.model = mock_model
        
        with pytest.raises(LLMServiceError) as exc_info:
            await llm_service._generate_with_retry("test prompt")
        
        assert "Empty response from Gemini" in str(exc_info.value)
    
    def test_estimate_token_usage(self, llm_service):
        """Test token usage estimation"""
        prompt = "This is a test prompt with some words."
        response = "This is a test response with more words."
        
        tokens = llm_service._estimate_token_usage(prompt, response)
        
        assert tokens > 0
        assert isinstance(tokens, int)
        # Should be roughly (len(prompt) + len(response)) / 4
        expected = (len(prompt) + len(response)) // 4
        assert abs(tokens - expected) <= 1
    
    def test_calculate_confidence_normal_completion(self, llm_service, mock_genai_model):
        """Test confidence calculation for normal completion"""
        _, mock_response = mock_genai_model
        
        confidence = llm_service._calculate_confidence(mock_response)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.7  # Should be high for normal completion
    
    def test_calculate_confidence_safety_issues(self, llm_service):
        """Test confidence calculation with safety issues"""
        mock_response = MagicMock()
        mock_response.text = "Response text"
        mock_response.candidates = [MagicMock()]
        
        # Mock safety rating with high probability
        mock_rating = MagicMock()
        mock_rating.probability = MagicMock()
        mock_rating.probability.name = "HIGH"
        
        mock_response.candidates[0].safety_ratings = [mock_rating]
        
        confidence = llm_service._calculate_confidence(mock_response)
        
        assert confidence == 0.3  # Low confidence due to safety concerns
    
    def test_calculate_confidence_short_response(self, llm_service):
        """Test confidence calculation for short response"""
        mock_response = MagicMock()
        mock_response.text = "Short"
        mock_response.candidates = []
        
        confidence = llm_service._calculate_confidence(mock_response)
        
        assert confidence == 0.3  # Low confidence for very short response
    
    def test_calculate_confidence_uncertain_response(self, llm_service):
        """Test confidence calculation for uncertain response"""
        mock_response = MagicMock()
        mock_response.text = "I don't know the answer to this question."
        mock_response.candidates = []
        
        confidence = llm_service._calculate_confidence(mock_response)
        
        assert confidence == 0.5  # Medium confidence for uncertain response
    
    def test_parse_batch_response_numbered(self, llm_service):
        """Test parsing numbered batch response"""
        response_text = """1. This is the first answer.
2. This is the second answer.
3. This is the third answer."""
        
        answers = llm_service._parse_batch_response(response_text, 3)
        
        assert len(answers) == 3
        assert "first answer" in answers[0]
        assert "second answer" in answers[1]
        assert "third answer" in answers[2]
    
    def test_parse_batch_response_insufficient_answers(self, llm_service):
        """Test parsing batch response with insufficient answers"""
        response_text = "1. Only one answer provided."
        
        answers = llm_service._parse_batch_response(response_text, 3)
        
        assert len(answers) == 3
        assert "Only one answer" in answers[0]
        assert "Unable to extract answer" in answers[1]
        assert "Unable to extract answer" in answers[2]
    
    @pytest.mark.asyncio
    async def test_batch_process_different_contexts(self, llm_service, mock_genai_model):
        """Test batch processing with different contexts"""
        mock_model, mock_response = mock_genai_model
        llm_service.model = mock_model
        llm_service._initialized = True
        
        requests = [
            LLMRequest(context="Context 1", query="Query 1"),
            LLMRequest(context="Context 2", query="Query 2")
        ]
        
        with patch('asyncio.gather') as mock_gather:
            mock_responses = [
                LLMResponse("Answer 1", 100, 500, "gemini", 0.8),
                LLMResponse("Answer 2", 120, 600, "gemini", 0.7)
            ]
            mock_gather.return_value = mock_responses
            
            responses = await llm_service.batch_process(requests)
            
            assert len(responses) == 2
            assert all(isinstance(r, LLMResponse) for r in responses)
    
    @pytest.mark.asyncio
    async def test_batch_process_same_context(self, llm_service, mock_genai_model):
        """Test batch processing with same context"""
        mock_model, mock_response = mock_genai_model
        mock_response.text = "1. Answer one.\n2. Answer two."
        llm_service.model = mock_model
        llm_service._initialized = True
        
        requests = [
            LLMRequest(context="Same context", query="Query 1"),
            LLMRequest(context="Same context", query="Query 2")
        ]
        
        responses = await llm_service.batch_process(requests)
        
        assert len(responses) == 2
        assert all(isinstance(r, LLMResponse) for r in responses)
        assert all(r.metadata.get("batch_processed") for r in responses)
    
    @pytest.mark.asyncio
    async def test_batch_process_empty_list(self, llm_service):
        """Test batch processing with empty request list"""
        responses = await llm_service.batch_process([])
        
        assert responses == []
    
    @pytest.mark.asyncio
    async def test_batch_process_with_exceptions(self, llm_service):
        """Test batch processing with some requests failing"""
        requests = [
            LLMRequest(context="Context 1", query="Query 1"),
            LLMRequest(context="Context 2", query="Query 2")
        ]
        
        with patch('asyncio.gather') as mock_gather:
            # First request succeeds, second fails
            mock_gather.return_value = [
                LLMResponse("Answer 1", 100, 500, "gemini", 0.8),
                Exception("API error")
            ]
            
            responses = await llm_service.batch_process(requests)
            
            assert len(responses) == 2
            assert responses[0].content == "Answer 1"
            assert "Error processing request" in responses[1].content
            assert responses[1].metadata.get("error") == True
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, llm_service):
        """Test getting model information"""
        info = await llm_service.get_model_info()
        
        assert "model_name" in info
        assert "max_context_length" in info
        assert "initialized" in info
        assert "available_templates" in info
        assert info["model_name"] == llm_service.model_name
        assert isinstance(info["available_templates"], list)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service, mock_genai_model):
        """Test successful health check"""
        mock_model, mock_response = mock_genai_model
        llm_service.model = mock_model
        llm_service._initialized = True
        
        is_healthy = await llm_service.health_check()
        
        assert is_healthy == True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service):
        """Test health check failure"""
        llm_service._initialized = True
        
        with patch.object(llm_service, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("Health check failed")
            
            is_healthy = await llm_service.health_check()
            
            assert is_healthy == False
    
    def test_prompt_templates_initialization(self, llm_service):
        """Test that prompt templates are properly initialized"""
        templates = llm_service.prompt_templates
        
        assert "answer_generation" in templates
        assert "coverage_analysis" in templates
        assert "definition_extraction" in templates
        assert "timeline_analysis" in templates
        assert "amount_extraction" in templates
        assert "batch_processing" in templates
        
        # Check that templates contain expected placeholders
        assert "{context}" in templates["answer_generation"]
        assert "{question}" in templates["answer_generation"]
        assert "ANSWER:" in templates["answer_generation"]
    
    def test_llm_request_dataclass(self):
        """Test LLMRequest dataclass"""
        request = LLMRequest(
            context="Test context",
            query="Test query",
            system_prompt="System prompt",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert request.context == "Test context"
        assert request.query == "Test query"
        assert request.system_prompt == "System prompt"
        assert request.temperature == 0.5
        assert request.max_tokens == 1000
    
    def test_llm_request_defaults(self):
        """Test LLMRequest with default values"""
        request = LLMRequest(context="Test", query="Test")
        
        assert request.system_prompt is None
        assert request.temperature == 0.1
        assert request.max_tokens is None
    
    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass"""
        response = LLMResponse(
            content="Test response",
            tokens_used=100,
            processing_time_ms=500,
            model_name="gemini-2.0-flash-exp",
            confidence=0.8,
            metadata={"key": "value"}
        )
        
        assert response.content == "Test response"
        assert response.tokens_used == 100
        assert response.processing_time_ms == 500
        assert response.model_name == "gemini-2.0-flash-exp"
        assert response.confidence == 0.8
        assert response.metadata == {"key": "value"}
    
    def test_llm_response_defaults(self):
        """Test LLMResponse with default values"""
        response = LLMResponse(
            content="Test",
            tokens_used=100,
            processing_time_ms=500,
            model_name="gemini"
        )
        
        assert response.confidence == 0.0
        assert response.metadata is None