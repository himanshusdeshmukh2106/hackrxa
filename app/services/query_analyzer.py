"""
Query analysis service for natural language query processing
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from app.core.logging import LoggerMixin
from app.schemas.models import QueryIntent, Entity


class QueryType(str, Enum):
    """Types of queries the system can handle"""
    COVERAGE = "coverage"  # What does the policy cover?
    EXCLUSION = "exclusion"  # What is excluded?
    CONDITION = "condition"  # What are the conditions?
    PROCEDURE = "procedure"  # How to do something?
    DEFINITION = "definition"  # What is X?
    COMPARISON = "comparison"  # Compare X and Y
    ELIGIBILITY = "eligibility"  # Who is eligible?
    TIMELINE = "timeline"  # When/how long?
    AMOUNT = "amount"  # How much?
    GENERAL = "general"  # General question


class DomainEntity(str, Enum):
    """Domain-specific entities"""
    # Insurance entities
    PREMIUM = "premium"
    DEDUCTIBLE = "deductible"
    COVERAGE = "coverage"
    CLAIM = "claim"
    POLICY = "policy"
    BENEFICIARY = "beneficiary"
    
    # Medical entities
    DISEASE = "disease"
    TREATMENT = "treatment"
    SURGERY = "surgery"
    MEDICATION = "medication"
    HOSPITAL = "hospital"
    DOCTOR = "doctor"
    
    # Legal entities
    CONTRACT = "contract"
    CLAUSE = "clause"
    OBLIGATION = "obligation"
    LIABILITY = "liability"
    JURISDICTION = "jurisdiction"
    
    # HR entities
    EMPLOYEE = "employee"
    BENEFIT = "benefit"
    LEAVE = "leave"
    SALARY = "salary"
    PERFORMANCE = "performance"
    
    # Time entities
    DATE = "date"
    PERIOD = "period"
    DEADLINE = "deadline"
    
    # Financial entities
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"


class QueryAnalyzer(LoggerMixin):
    """Service for analyzing and preprocessing natural language queries"""
    
    def __init__(self):
        self.domain_patterns = self._initialize_domain_patterns()
        self.query_type_patterns = self._initialize_query_type_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keyword patterns"""
        return {
            "insurance": [
                "policy", "premium", "coverage", "claim", "deductible", "beneficiary",
                "insurance", "insured", "policyholder", "underwriting", "actuarial"
            ],
            "medical": [
                "disease", "treatment", "surgery", "medication", "hospital", "doctor",
                "patient", "diagnosis", "therapy", "medical", "health", "clinical"
            ],
            "legal": [
                "contract", "clause", "agreement", "liability", "jurisdiction", "law",
                "legal", "court", "judge", "attorney", "litigation", "compliance"
            ],
            "hr": [
                "employee", "benefit", "leave", "salary", "performance", "hiring",
                "termination", "promotion", "training", "workplace", "personnel"
            ]
        }
    
    def _initialize_query_type_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize query type detection patterns"""
        return {
            QueryType.COVERAGE: [
                "cover", "covers", "covered", "coverage", "include", "includes", "included"
            ],
            QueryType.EXCLUSION: [
                "exclude", "excludes", "excluded", "exclusion", "not cover", "not covered",
                "except", "exception", "limitation", "restrict"
            ],
            QueryType.CONDITION: [
                "condition", "conditions", "requirement", "requirements", "criteria",
                "prerequisite", "terms", "stipulation"
            ],
            QueryType.PROCEDURE: [
                "how to", "process", "procedure", "steps", "method", "way to",
                "instructions", "guide", "apply", "submit"
            ],
            QueryType.DEFINITION: [
                "what is", "what are", "define", "definition", "meaning", "means",
                "refer to", "refers to", "considered"
            ],
            QueryType.COMPARISON: [
                "compare", "comparison", "difference", "different", "versus", "vs",
                "better", "worse", "similar", "contrast"
            ],
            QueryType.ELIGIBILITY: [
                "eligible", "eligibility", "qualify", "qualifies", "qualified",
                "who can", "who is", "allowed", "permitted"
            ],
            QueryType.TIMELINE: [
                "when", "how long", "duration", "period", "time", "deadline",
                "waiting period", "grace period", "effective date"
            ],
            QueryType.AMOUNT: [
                "how much", "amount", "cost", "price", "fee", "charge",
                "limit", "maximum", "minimum", "percentage"
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[DomainEntity, List[str]]:
        """Initialize entity recognition patterns"""
        return {
            # Insurance entities
            DomainEntity.PREMIUM: ["premium", "premiums", "payment", "installment"],
            DomainEntity.DEDUCTIBLE: ["deductible", "deductibles", "excess"],
            DomainEntity.COVERAGE: ["coverage", "cover", "benefit", "protection"],
            DomainEntity.CLAIM: ["claim", "claims", "reimbursement", "settlement"],
            DomainEntity.POLICY: ["policy", "policies", "plan", "scheme"],
            DomainEntity.BENEFICIARY: ["beneficiary", "beneficiaries", "nominee"],
            
            # Medical entities
            DomainEntity.DISEASE: ["disease", "illness", "condition", "disorder", "syndrome"],
            DomainEntity.TREATMENT: ["treatment", "therapy", "care", "intervention"],
            DomainEntity.SURGERY: ["surgery", "operation", "procedure", "surgical"],
            DomainEntity.MEDICATION: ["medication", "medicine", "drug", "prescription"],
            DomainEntity.HOSPITAL: ["hospital", "clinic", "medical center", "facility"],
            DomainEntity.DOCTOR: ["doctor", "physician", "specialist", "practitioner"],
            
            # Time entities
            DomainEntity.DATE: ["date", "day", "month", "year"],
            DomainEntity.PERIOD: ["period", "duration", "term", "span"],
            DomainEntity.DEADLINE: ["deadline", "due date", "expiry", "expiration"],
            
            # Financial entities
            DomainEntity.AMOUNT: ["amount", "sum", "total", "value"],
            DomainEntity.PERCENTAGE: ["percent", "percentage", "%", "rate"],
            DomainEntity.CURRENCY: ["dollar", "rupee", "euro", "currency", "$", "₹", "€"]
        }
    
    async def analyze_query(self, query: str) -> QueryIntent:
        """
        Analyze a natural language query and extract intent
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryIntent object with analysis results
        """
        try:
            self.logger.info(f"Analyzing query: {query[:100]}...")
            
            # Clean and preprocess query
            processed_query = self._preprocess_query(query)
            
            # Determine query type
            query_type = self._classify_query_type(processed_query)
            
            # Extract entities
            entities = self._extract_entities(processed_query)
            
            # Calculate confidence based on pattern matches
            confidence = self._calculate_confidence(processed_query, query_type, entities)
            
            # Extract keywords (simple implementation)
            keywords = list(set(re.findall(r'\b\w+\b', processed_query.lower())))

            query_intent = QueryIntent(
                original_query=query,
                intent_type=query_type.value,
                entities=entities,
                confidence=confidence,
                processed_query=processed_query,
                keywords=keywords
            )
            
            self.logger.info(f"Query analysis complete: type={query_type.value}, confidence={confidence:.2f}")
            return query_intent
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {str(e)}")
            # Return default intent on failure
            return QueryIntent(
                original_query=query,
                intent_type=QueryType.GENERAL.value,
                entities=[],
                confidence=0.5,
                processed_query=query.lower().strip()
            )
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text for analysis
        
        Args:
            query: Raw query string
            
        Returns:
            Preprocessed query string
        """
        # Convert to lowercase
        processed = query.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove punctuation except question marks and periods
        processed = re.sub(r'[^\w\s\?\.]', ' ', processed)
        
        # Handle common contractions
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "when's": "when is",
            "who's": "who is",
            "why's": "why is",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not"
        }
        
        for contraction, expansion in contractions.items():
            processed = processed.replace(contraction, expansion)
        
        return processed.strip()
    
    def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify the type of query based on patterns
        
        Args:
            query: Preprocessed query string
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower()
        
        # Score each query type based on pattern matches
        type_scores = {}
        
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    # Give higher score for exact matches
                    if f" {pattern} " in f" {query_lower} ":
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                type_scores[query_type] = score
        
        # Return the highest scoring type, or GENERAL if no matches
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryType.GENERAL
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """
        Extract domain-specific entities from query
        
        Args:
            query: Preprocessed query string
            
        Returns:
            List of Entity objects
        """
        entities = []
        query_lower = query.lower()
        
        # Extract entities based on patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                # Find all occurrences of the pattern
                matches = list(re.finditer(r'\b' + re.escape(pattern) + r'\b', query_lower))
                
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=entity_type.value,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8  # Base confidence for pattern matches
                    )
                    entities.append(entity)
        
        # Extract numerical entities
        entities.extend(self._extract_numerical_entities(query))
        
        # Extract date entities
        entities.extend(self._extract_date_entities(query))
        
        # Remove duplicates and overlapping entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_numerical_entities(self, query: str) -> List[Entity]:
        """Extract numerical entities (amounts, percentages, etc.)"""
        entities = []
        
        # Pattern for amounts with currency symbols
        currency_pattern = r'(\$|₹|€|USD|INR|EUR)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        for match in re.finditer(currency_pattern, query, re.IGNORECASE):
            entity = Entity(
                text=match.group(),
                label=DomainEntity.AMOUNT.value,
                start=match.start(),
                end=match.end(),
                confidence=0.9
            )
            entities.append(entity)
        
        # Pattern for percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(percentage_pattern, query):
            entity = Entity(
                text=match.group(),
                label=DomainEntity.PERCENTAGE.value,
                start=match.start(),
                end=match.end(),
                confidence=0.9
            )
            entities.append(entity)
        
        # Pattern for general numbers
        number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
        for match in re.finditer(number_pattern, query):
            # Skip if already captured as currency or percentage
            if not any(e.start <= match.start() < e.end for e in entities):
                entity = Entity(
                    text=match.group(),
                    label=DomainEntity.AMOUNT.value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.6
                )
                entities.append(entity)
        
        return entities
    
    def _extract_date_entities(self, query: str) -> List[Entity]:
        """Extract date and time entities"""
        entities = []
        
        # Pattern for dates (various formats)
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4})\b',
            r'\b((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})\b',
            r'\b(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entity = Entity(
                    text=match.group(1),
                    label=DomainEntity.DATE.value,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.8
                )
                entities.append(entity)
        
        # Pattern for time periods
        period_patterns = [
            r'\b(\d+\s+(days?|weeks?|months?|years?))\b',
            r'\b(\d+\s*-\s*\d+\s+(days?|weeks?|months?|years?))\b'
        ]
        
        for pattern in period_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entity = Entity(
                    text=match.group(1),
                    label=DomainEntity.PERIOD.value,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _calculate_confidence(
        self, 
        query: str, 
        query_type: QueryType, 
        entities: List[Entity]
    ) -> float:
        """
        Calculate confidence score for the query analysis
        
        Args:
            query: Preprocessed query
            query_type: Detected query type
            entities: Extracted entities
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5
        
        # Boost confidence based on query type pattern matches
        type_patterns = self.query_type_patterns.get(query_type, [])
        type_matches = sum(1 for pattern in type_patterns if pattern in query.lower())
        type_boost = min(type_matches * 0.1, 0.3)
        
        # Boost confidence based on entity extraction
        entity_boost = min(len(entities) * 0.05, 0.2)
        
        # Boost confidence based on domain relevance
        domain_boost = 0.0
        for domain, keywords in self.domain_patterns.items():
            domain_matches = sum(1 for keyword in keywords if keyword in query.lower())
            if domain_matches > 0:
                domain_boost = min(domain_matches * 0.05, 0.15)
                break
        
        # Calculate final confidence
        confidence = base_confidence + type_boost + entity_boost + domain_boost
        
        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0)
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested completions
        """
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        # Common query starters
        starters = [
            "What is the coverage for",
            "Does this policy cover",
            "What are the conditions for",
            "How much is the premium for",
            "What is the waiting period for",
            "Are there any exclusions for",
            "What is the definition of",
            "How do I claim for",
            "What documents are required for",
            "What is the grace period for"
        ]
        
        # Filter starters that match the partial query
        for starter in starters:
            if starter.lower().startswith(partial_lower) or partial_lower in starter.lower():
                suggestions.append(starter)
        
        return suggestions[:5]  # Return top 5 suggestions


# Global query analyzer instance
query_analyzer = QueryAnalyzer()