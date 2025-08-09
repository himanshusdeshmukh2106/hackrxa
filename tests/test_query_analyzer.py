"""
Unit tests for query analyzer service
"""
import pytest

from app.services.query_analyzer import QueryAnalyzer, QueryType, DomainEntity
from app.schemas.models import QueryIntent, Entity


class TestQueryAnalyzer:
    """Test QueryAnalyzer functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create QueryAnalyzer instance"""
        return QueryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_coverage_query(self, analyzer):
        """Test analysis of coverage-related query"""
        query = "Does this policy cover knee surgery?"
        
        result = await analyzer.analyze_query(query)
        
        assert isinstance(result, QueryIntent)
        assert result.original_query == query
        assert result.intent_type == QueryType.COVERAGE.value
        assert result.confidence > 0.5
        assert any(entity.label == DomainEntity.SURGERY.value for entity in result.entities)
    
    @pytest.mark.asyncio
    async def test_analyze_exclusion_query(self, analyzer):
        """Test analysis of exclusion-related query"""
        query = "What is excluded from the coverage?"
        
        result = await analyzer.analyze_query(query)
        
        assert result.intent_type == QueryType.EXCLUSION.value
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_analyze_definition_query(self, analyzer):
        """Test analysis of definition-related query"""
        query = "What is a pre-existing disease?"
        
        result = await analyzer.analyze_query(query)
        
        assert result.intent_type == QueryType.DEFINITION.value
        assert any(entity.label == DomainEntity.DISEASE.value for entity in result.entities)
    
    @pytest.mark.asyncio
    async def test_analyze_timeline_query(self, analyzer):
        """Test analysis of timeline-related query"""
        query = "What is the waiting period for maternity benefits?"
        
        result = await analyzer.analyze_query(query)
        
        assert result.intent_type == QueryType.TIMELINE.value
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_analyze_amount_query(self, analyzer):
        """Test analysis of amount-related query"""
        query = "How much is the premium for this policy?"
        
        result = await analyzer.analyze_query(query)
        
        assert result.intent_type == QueryType.AMOUNT.value
        assert any(entity.label == DomainEntity.PREMIUM.value for entity in result.entities)
    
    @pytest.mark.asyncio
    async def test_analyze_general_query(self, analyzer):
        """Test analysis of general query"""
        query = "Tell me about this document"
        
        result = await analyzer.analyze_query(query)
        
        assert result.intent_type == QueryType.GENERAL.value
        assert result.confidence >= 0.5
    
    def test_preprocess_query(self, analyzer):
        """Test query preprocessing"""
        query = "What's the   coverage for knee-surgery???"
        
        processed = analyzer._preprocess_query(query)
        
        assert processed == "what is the coverage for knee surgery"
        assert "what's" not in processed
        assert "  " not in processed
        assert "???" not in processed
    
    def test_preprocess_query_contractions(self, analyzer):
        """Test contraction handling in preprocessing"""
        query = "Can't I get coverage? Won't it cover surgery?"
        
        processed = analyzer._preprocess_query(query)
        
        assert "cannot" in processed
        assert "will not" in processed
        assert "can't" not in processed
        assert "won't" not in processed
    
    def test_classify_query_type_coverage(self, analyzer):
        """Test query type classification for coverage"""
        query = "does this policy cover dental treatment"
        
        query_type = analyzer._classify_query_type(query)
        
        assert query_type == QueryType.COVERAGE
    
    def test_classify_query_type_exclusion(self, analyzer):
        """Test query type classification for exclusion"""
        query = "what is not covered by this policy"
        
        query_type = analyzer._classify_query_type(query)
        
        assert query_type == QueryType.EXCLUSION
    
    def test_classify_query_type_definition(self, analyzer):
        """Test query type classification for definition"""
        query = "what is the meaning of deductible"
        
        query_type = analyzer._classify_query_type(query)
        
        assert query_type == QueryType.DEFINITION
    
    def test_classify_query_type_general(self, analyzer):
        """Test query type classification for general queries"""
        query = "tell me something about this"
        
        query_type = analyzer._classify_query_type(query)
        
        assert query_type == QueryType.GENERAL
    
    def test_extract_entities_medical(self, analyzer):
        """Test extraction of medical entities"""
        query = "does the policy cover heart surgery and medication"
        
        entities = analyzer._extract_entities(query)
        
        surgery_entities = [e for e in entities if e.label == DomainEntity.SURGERY.value]
        medication_entities = [e for e in entities if e.label == DomainEntity.MEDICATION.value]
        
        assert len(surgery_entities) > 0
        assert len(medication_entities) > 0
        assert surgery_entities[0].text == "surgery"
        assert medication_entities[0].text == "medication"
    
    def test_extract_entities_insurance(self, analyzer):
        """Test extraction of insurance entities"""
        query = "what is the premium and deductible for this policy"
        
        entities = analyzer._extract_entities(query)
        
        premium_entities = [e for e in entities if e.label == DomainEntity.PREMIUM.value]
        deductible_entities = [e for e in entities if e.label == DomainEntity.DEDUCTIBLE.value]
        policy_entities = [e for e in entities if e.label == DomainEntity.POLICY.value]
        
        assert len(premium_entities) > 0
        assert len(deductible_entities) > 0
        assert len(policy_entities) > 0
    
    def test_extract_numerical_entities_currency(self, analyzer):
        """Test extraction of currency amounts"""
        query = "the premium is $500 per month"
        
        entities = analyzer._extract_numerical_entities(query)
        
        amount_entities = [e for e in entities if e.label == DomainEntity.AMOUNT.value]
        assert len(amount_entities) > 0
        assert "$500" in amount_entities[0].text
        assert amount_entities[0].confidence == 0.9
    
    def test_extract_numerical_entities_percentage(self, analyzer):
        """Test extraction of percentages"""
        query = "there is a 20% co-payment required"
        
        entities = analyzer._extract_numerical_entities(query)
        
        percentage_entities = [e for e in entities if e.label == DomainEntity.PERCENTAGE.value]
        assert len(percentage_entities) > 0
        assert "20%" in percentage_entities[0].text
    
    def test_extract_numerical_entities_general_numbers(self, analyzer):
        """Test extraction of general numbers"""
        query = "the waiting period is 30 days"
        
        entities = analyzer._extract_numerical_entities(query)
        
        amount_entities = [e for e in entities if e.label == DomainEntity.AMOUNT.value]
        assert len(amount_entities) > 0
        assert "30" in amount_entities[0].text
    
    def test_extract_date_entities_numeric_date(self, analyzer):
        """Test extraction of numeric date formats"""
        query = "the policy starts on 01/15/2025"
        
        entities = analyzer._extract_date_entities(query)
        
        date_entities = [e for e in entities if e.label == DomainEntity.DATE.value]
        assert len(date_entities) > 0
        assert "01/15/2025" in date_entities[0].text
    
    def test_extract_date_entities_text_date(self, analyzer):
        """Test extraction of text date formats"""
        query = "coverage begins on January 15, 2025"
        
        entities = analyzer._extract_date_entities(query)
        
        date_entities = [e for e in entities if e.label == DomainEntity.DATE.value]
        assert len(date_entities) > 0
        assert "January 15, 2025" in date_entities[0].text
    
    def test_extract_date_entities_period(self, analyzer):
        """Test extraction of time periods"""
        query = "there is a 6 months waiting period"
        
        entities = analyzer._extract_date_entities(query)
        
        period_entities = [e for e in entities if e.label == DomainEntity.PERIOD.value]
        assert len(period_entities) > 0
        assert "6 months" in period_entities[0].text
    
    def test_deduplicate_entities(self, analyzer):
        """Test entity deduplication"""
        entities = [
            Entity(text="surgery", label="surgery", start=10, end=17, confidence=0.8),
            Entity(text="surgery", label="treatment", start=10, end=17, confidence=0.6),  # Overlapping
            Entity(text="policy", label="policy", start=25, end=31, confidence=0.9)
        ]
        
        deduplicated = analyzer._deduplicate_entities(entities)
        
        # Should keep the higher confidence entity
        assert len(deduplicated) == 2
        surgery_entity = next(e for e in deduplicated if e.text == "surgery")
        assert surgery_entity.confidence == 0.8
        assert surgery_entity.label == "surgery"
    
    def test_calculate_confidence_high(self, analyzer):
        """Test confidence calculation for well-matched query"""
        query = "does this insurance policy cover heart surgery"
        query_type = QueryType.COVERAGE
        entities = [
            Entity(text="insurance", label="insurance", start=0, end=9, confidence=0.8),
            Entity(text="policy", label="policy", start=10, end=16, confidence=0.8),
            Entity(text="surgery", label="surgery", start=20, end=27, confidence=0.8)
        ]
        
        confidence = analyzer._calculate_confidence(query, query_type, entities)
        
        assert confidence > 0.7  # Should be high confidence
    
    def test_calculate_confidence_low(self, analyzer):
        """Test confidence calculation for poorly matched query"""
        query = "tell me something"
        query_type = QueryType.GENERAL
        entities = []
        
        confidence = analyzer._calculate_confidence(query, query_type, entities)
        
        assert confidence <= 0.6  # Should be lower confidence
    
    def test_get_query_suggestions(self, analyzer):
        """Test query suggestions"""
        partial_query = "what is"
        
        suggestions = analyzer.get_query_suggestions(partial_query)
        
        assert len(suggestions) > 0
        assert len(suggestions) <= 5
        assert any("what is" in suggestion.lower() for suggestion in suggestions)
    
    def test_get_query_suggestions_coverage(self, analyzer):
        """Test query suggestions for coverage-related partial query"""
        partial_query = "does this policy"
        
        suggestions = analyzer.get_query_suggestions(partial_query)
        
        assert len(suggestions) > 0
        assert any("cover" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_analyze_query_error_handling(self, analyzer):
        """Test error handling in query analysis"""
        # Test with None input (should be handled gracefully)
        with pytest.raises(AttributeError):
            await analyzer.analyze_query(None)
        
        # Test with empty string
        result = await analyzer.analyze_query("")
        assert result.intent_type == QueryType.GENERAL.value
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_complex_query(self, analyzer):
        """Test analysis of complex multi-part query"""
        query = "What is the premium for a 35-year-old male, and does it cover pre-existing heart disease with a $1000 deductible?"
        
        result = await analyzer.analyze_query(query)
        
        # Should detect multiple entity types
        entity_labels = [e.label for e in result.entities]
        assert DomainEntity.PREMIUM.value in entity_labels
        assert DomainEntity.DISEASE.value in entity_labels
        assert DomainEntity.AMOUNT.value in entity_labels
        
        # Should have reasonable confidence
        assert result.confidence > 0.6