"""
Response generation system with explainability and confidence scoring
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import ResponseGenerationError
from app.schemas.models import SearchResult, QueryIntent, Answer
from app.services.llm_service import llm_service, LLMRequest


class ResponseGenerator(LoggerMixin):
    """Service for generating contextual answers with explainability"""
    
    def __init__(self):
        self.max_context_chunks = settings.max_context_chunks
        self.min_confidence_threshold = settings.min_confidence_threshold
        self.context_window_size = 8000  # Characters for context window
    
    async def generate_answer(
        self, 
        context_results: List[SearchResult], 
        query: str,
        query_intent: Optional[QueryIntent] = None
    ) -> Answer:
        """
        Generate a contextual answer from search results
        
        Args:
            context_results: List of relevant search results
            query: Original user query
            query_intent: Optional analyzed query intent
            
        Returns:
            Answer object with explainability
        """
        try:
            self.logger.info(f"Generating answer for query: {query[:100]}...")
            
            if not context_results:
                return self._create_no_context_answer(query)
            
            # Select and rank context chunks
            selected_chunks = self._select_context_chunks(context_results)
            
            if not selected_chunks:
                return self._create_no_context_answer(query)
            
            # Build context string
            context_text = self._build_context_text(selected_chunks)
            
            # Generate response using LLM
            llm_response = await llm_service.generate_response(
                context=context_text,
                query=query,
                query_intent=query_intent
            )
            
            # Create answer with explainability
            answer = Answer(
                question=query,
                answer=llm_response.content,
                confidence=self._calculate_final_confidence(llm_response, selected_chunks),
                source_chunks=[chunk.chunk_id for chunk in selected_chunks],
                reasoning=self._generate_reasoning(selected_chunks, llm_response, query_intent),
                metadata={
                    "llm_tokens_used": llm_response.tokens_used,
                    "llm_processing_time_ms": llm_response.processing_time_ms,
                    "context_chunks_count": len(selected_chunks),
                    "avg_chunk_similarity": sum(c.similarity_score for c in selected_chunks) / len(selected_chunks),
                    "query_intent_type": query_intent.intent_type if query_intent else None,
                    "llm_confidence": llm_response.confidence
                }
            )
            
            self.logger.info(f"Generated answer with confidence {answer.confidence:.3f}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}")
            raise ResponseGenerationError(f"Failed to generate answer: {str(e)}")
    
    async def generate_batch_answers(
        self, 
        questions: List[str],
        context_results: List[SearchResult],
        query_intents: Optional[List[QueryIntent]] = None
    ) -> List[Answer]:
        """
        Generate answers for multiple questions using the same context
        
        Args:
            questions: List of questions to answer
            context_results: Shared context results
            query_intents: Optional list of query intents
            
        Returns:
            List of Answer objects
        """
        try:
            self.logger.info(f"Generating batch answers for {len(questions)} questions")
            
            if not context_results:
                return [self._create_no_context_answer(q) for q in questions]
            
            # Select context chunks (same for all questions)
            selected_chunks = self._select_context_chunks(context_results)
            context_text = self._build_context_text(selected_chunks)
            
            # Create LLM requests
            requests = []
            for i, question in enumerate(questions):
                intent = query_intents[i] if query_intents and i < len(query_intents) else None
                request = LLMRequest(
                    context=context_text,
                    query=question,
                    temperature=0.1
                )
                requests.append(request)
            
            # Process batch
            llm_responses = await llm_service.batch_process(requests)
            
            # Create answers
            answers = []
            for i, (question, llm_response) in enumerate(zip(questions, llm_responses)):
                intent = query_intents[i] if query_intents and i < len(query_intents) else None
                
                answer = Answer(
                    question=question,
                    answer=llm_response.content,
                    confidence=self._calculate_final_confidence(llm_response, selected_chunks),
                    source_chunks=[chunk.chunk_id for chunk in selected_chunks],
                    reasoning=self._generate_reasoning(selected_chunks, llm_response, intent),
                    metadata={
                        "llm_tokens_used": llm_response.tokens_used,
                        "llm_processing_time_ms": llm_response.processing_time_ms,
                        "context_chunks_count": len(selected_chunks),
                        "batch_processed": True,
                        "batch_index": i,
                        "query_intent_type": intent.intent_type if intent else None
                    }
                )
                answers.append(answer)
            
            self.logger.info(f"Generated {len(answers)} batch answers")
            
            return answers
            
        except Exception as e:
            self.logger.error(f"Batch answer generation failed: {str(e)}")
            raise ResponseGenerationError(f"Failed to generate batch answers: {str(e)}")
    
    def _select_context_chunks(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Select the most relevant context chunks within size limits
        
        Args:
            search_results: List of search results to select from
            
        Returns:
            List of selected SearchResult objects
        """
        if not search_results:
            return []
        
        # Sort by similarity score
        sorted_results = sorted(search_results, key=lambda x: x.similarity_score, reverse=True)
        
        selected_chunks = []
        total_context_size = 0
        
        for result in sorted_results:
            # Check if adding this chunk would exceed limits
            chunk_size = len(result.content)
            
            if (len(selected_chunks) >= self.max_context_chunks or 
                total_context_size + chunk_size > self.context_window_size):
                break
            
            selected_chunks.append(result)
            total_context_size += chunk_size
        
        self.logger.info(f"Selected {len(selected_chunks)} context chunks ({total_context_size} chars)")
        
        return selected_chunks
    
    def _build_context_text(self, chunks: List[SearchResult]) -> str:
        """
        Build formatted context text from selected chunks
        
        Args:
            chunks: List of selected search result chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Add chunk header with metadata
            chunk_header = f"[Context Chunk {i+1}]"
            
            # Add document metadata if available
            doc_metadata = chunk.document_metadata
            if doc_metadata.get("document_id"):
                chunk_header += f" (Document: {doc_metadata['document_id']}"
                if doc_metadata.get("chunk_index") is not None:
                    chunk_header += f", Section: {doc_metadata['chunk_index'] + 1}"
                chunk_header += ")"
            
            # Add similarity score
            chunk_header += f" [Relevance: {chunk.similarity_score:.3f}]"
            
            context_parts.append(chunk_header)
            context_parts.append(chunk.content.strip())
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _calculate_final_confidence(
        self, 
        llm_response, 
        context_chunks: List[SearchResult]
    ) -> float:
        """
        Calculate final confidence score combining LLM and context factors
        
        Args:
            llm_response: Response from LLM service
            context_chunks: Context chunks used for generation
            
        Returns:
            Final confidence score (0.0 to 1.0)
        """
        # Base confidence from LLM
        llm_confidence = llm_response.confidence
        
        # Context quality factor
        if not context_chunks:
            context_confidence = 0.0
        else:
            # Average similarity score of context chunks
            avg_similarity = sum(chunk.similarity_score for chunk in context_chunks) / len(context_chunks)
            
            # Boost confidence if we have multiple high-quality chunks
            chunk_count_factor = min(1.0, len(context_chunks) / 3.0)  # Optimal around 3 chunks
            
            context_confidence = avg_similarity * chunk_count_factor
        
        # Response quality factors
        response_text = llm_response.content.lower()
        
        # Penalty for uncertain language
        uncertainty_penalty = 0.0
        uncertain_phrases = [
            "i don't know", "cannot determine", "unclear", "insufficient information",
            "not enough context", "unable to answer", "not specified", "not mentioned"
        ]
        
        for phrase in uncertain_phrases:
            if phrase in response_text:
                uncertainty_penalty += 0.1
        
        uncertainty_penalty = min(0.3, uncertainty_penalty)  # Cap at 30% penalty
        
        # Boost for specific, detailed answers
        specificity_boost = 0.0
        if len(llm_response.content) > 100:  # Detailed response
            specificity_boost += 0.1
        
        # Look for specific details (numbers, dates, amounts)
        if re.search(r'\d+', llm_response.content):  # Contains numbers
            specificity_boost += 0.05
        
        if re.search(r'\$|₹|%', llm_response.content):  # Contains monetary/percentage values
            specificity_boost += 0.05
        
        # Combine all factors
        final_confidence = (
            llm_confidence * 0.4 +
            context_confidence * 0.4 +
            specificity_boost * 0.2 -
            uncertainty_penalty
        )
        
        # Ensure within bounds
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_confidence
    
    def _generate_reasoning(
        self, 
        context_chunks: List[SearchResult],
        llm_response,
        query_intent: Optional[QueryIntent] = None
    ) -> str:
        """
        Generate explanation for how the answer was derived
        
        Args:
            context_chunks: Context chunks used
            llm_response: LLM response object
            query_intent: Optional query intent
            
        Returns:
            Reasoning explanation string
        """
        reasoning_parts = []
        
        # Context analysis
        if context_chunks:
            reasoning_parts.append(f"Answer derived from {len(context_chunks)} relevant document sections:")
            
            for i, chunk in enumerate(context_chunks[:3]):  # Show top 3 sources
                doc_info = ""
                if chunk.document_metadata.get("document_id"):
                    doc_info = f" from document {chunk.document_metadata['document_id']}"
                
                reasoning_parts.append(
                    f"  {i+1}. Section with {chunk.similarity_score:.1%} relevance{doc_info}"
                )
            
            if len(context_chunks) > 3:
                reasoning_parts.append(f"  ... and {len(context_chunks) - 3} additional sections")
        
        # Query intent analysis
        if query_intent:
            intent_explanation = {
                "coverage": "Analyzed for coverage-related information",
                "exclusion": "Searched for exclusions and limitations",
                "definition": "Extracted relevant definitions and explanations",
                "timeline": "Focused on time-related requirements and deadlines",
                "amount": "Identified monetary amounts and financial terms"
            }
            
            if query_intent.intent_type in intent_explanation:
                reasoning_parts.append(f"Query intent: {intent_explanation[query_intent.intent_type]}")
            
            if query_intent.entities:
                entity_texts = [entity.text for entity in query_intent.entities[:3]]
                reasoning_parts.append(f"Key terms identified: {', '.join(entity_texts)}")
        
        # LLM processing info
        reasoning_parts.append(
            f"Response generated using {llm_response.model_name} "
            f"({llm_response.tokens_used} tokens, {llm_response.processing_time_ms}ms)"
        )
        
        # Confidence explanation
        confidence_level = "high" if llm_response.confidence > 0.7 else "medium" if llm_response.confidence > 0.4 else "low"
        reasoning_parts.append(f"Answer confidence: {confidence_level} ({llm_response.confidence:.1%})")
        
        return " | ".join(reasoning_parts)
    
    def _create_no_context_answer(self, query: str) -> Answer:
        """
        Create answer when no relevant context is found
        
        Args:
            query: Original query
            
        Returns:
            Answer indicating no context found
        """
        return Answer(
            question=query,
            answer="I don't have sufficient information in the provided documents to answer this question. Please ensure the relevant documents are uploaded and try rephrasing your question.",
            confidence=0.0,
            source_chunks=[],
            reasoning="No relevant context found in the document collection",
            metadata={
                "no_context": True,
                "suggestion": "Try rephrasing the question or check if relevant documents are available"
            }
        )
    
    async def explain_reasoning(self, answer: Answer) -> Dict[str, Any]:
        """
        Provide detailed explanation of answer reasoning
        
        Args:
            answer: Answer object to explain
            
        Returns:
            Dictionary with detailed reasoning explanation
        """
        try:
            explanation = {
                "answer_confidence": answer.confidence,
                "confidence_level": self._get_confidence_level(answer.confidence),
                "source_analysis": {
                    "total_sources": len(answer.source_chunks),
                    "source_chunks": answer.source_chunks
                },
                "processing_details": {
                    "llm_tokens_used": answer.metadata.get("llm_tokens_used", 0),
                    "processing_time_ms": answer.metadata.get("llm_processing_time_ms", 0),
                    "context_chunks_used": answer.metadata.get("context_chunks_count", 0)
                },
                "quality_indicators": self._analyze_answer_quality(answer),
                "reasoning_breakdown": answer.reasoning
            }
            
            # Add suggestions for low confidence answers
            if answer.confidence < self.min_confidence_threshold:
                explanation["improvement_suggestions"] = [
                    "Try rephrasing the question with more specific terms",
                    "Ensure relevant documents are uploaded to the system",
                    "Check if the question relates to content actually present in the documents",
                    "Consider breaking complex questions into simpler parts"
                ]
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to explain reasoning: {str(e)}")
            return {
                "error": "Failed to generate reasoning explanation",
                "answer_confidence": answer.confidence,
                "basic_reasoning": answer.reasoning
            }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _analyze_answer_quality(self, answer: Answer) -> Dict[str, Any]:
        """
        Analyze various quality indicators of the answer
        
        Args:
            answer: Answer to analyze
            
        Returns:
            Dictionary with quality indicators
        """
        quality_indicators = {}
        
        # Length analysis
        answer_length = len(answer.answer)
        quality_indicators["answer_length"] = answer_length
        quality_indicators["length_category"] = (
            "detailed" if answer_length > 200 else
            "moderate" if answer_length > 50 else
            "brief"
        )
        
        # Specificity indicators
        answer_lower = answer.answer.lower()
        
        # Check for specific details
        has_numbers = bool(re.search(r'\d+', answer.answer))
        has_monetary = bool(re.search(r'\$|₹|%|cost|price|premium|fee', answer_lower))
        has_timeframes = bool(re.search(r'day|week|month|year|period|deadline', answer_lower))
        has_conditions = bool(re.search(r'if|when|provided|subject to|condition', answer_lower))
        
        quality_indicators["specificity"] = {
            "contains_numbers": has_numbers,
            "contains_monetary_info": has_monetary,
            "contains_timeframes": has_timeframes,
            "contains_conditions": has_conditions
        }
        
        # Uncertainty indicators
        uncertainty_phrases = [
            "may", "might", "could", "possibly", "potentially", "unclear",
            "not specified", "depends on", "varies"
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
        quality_indicators["uncertainty_level"] = (
            "high" if uncertainty_count > 3 else
            "medium" if uncertainty_count > 1 else
            "low"
        )
        
        # Source utilization
        quality_indicators["source_utilization"] = {
            "sources_used": len(answer.source_chunks),
            "avg_source_relevance": answer.metadata.get("avg_chunk_similarity", 0.0)
        }
        
        return quality_indicators
    
    async def validate_answer_accuracy(
        self, 
        answer: Answer, 
        context_chunks: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Validate answer accuracy against source context
        
        Args:
            answer: Generated answer to validate
            context_chunks: Original context chunks
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                "overall_accuracy": "unknown",
                "context_alignment": 0.0,
                "factual_consistency": True,
                "issues_found": []
            }
            
            # Check if answer content aligns with context
            answer_lower = answer.answer.lower()
            context_text = " ".join([chunk.content.lower() for chunk in context_chunks])
            
            # Extract key claims from answer
            key_claims = self._extract_key_claims(answer.answer)
            
            # Validate each claim against context
            validated_claims = 0
            for claim in key_claims:
                if self._validate_claim_against_context(claim, context_text):
                    validated_claims += 1
                else:
                    validation_results["issues_found"].append(f"Claim not supported by context: {claim[:100]}...")
            
            if key_claims:
                validation_results["context_alignment"] = validated_claims / len(key_claims)
            
            # Overall accuracy assessment
            if validation_results["context_alignment"] >= 0.8:
                validation_results["overall_accuracy"] = "high"
            elif validation_results["context_alignment"] >= 0.6:
                validation_results["overall_accuracy"] = "medium"
            else:
                validation_results["overall_accuracy"] = "low"
                validation_results["factual_consistency"] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Answer validation failed: {str(e)}")
            return {
                "overall_accuracy": "unknown",
                "error": str(e)
            }
    
    def _extract_key_claims(self, answer_text: str) -> List[str]:
        """Extract key factual claims from answer text"""
        # Simple sentence splitting for key claims
        sentences = re.split(r'[.!?]+', answer_text)
        
        # Filter out very short sentences and common phrases
        key_claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                not sentence.lower().startswith(('however', 'therefore', 'additionally', 'furthermore'))):
                key_claims.append(sentence)
        
        return key_claims[:5]  # Limit to top 5 claims
    
    def _validate_claim_against_context(self, claim: str, context_text: str) -> bool:
        """Validate a single claim against context text"""
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        # Remove common words and focus on content words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        claim_words = [word for word in re.findall(r'\w+', claim_lower) if word not in stop_words and len(word) > 2]
        
        # Check if significant portion of claim words appear in context
        if not claim_words:
            return True  # Empty claim is considered valid
        
        found_words = sum(1 for word in claim_words if word in context_text)
        
        # Require at least 60% of key words to be found in context
        return found_words / len(claim_words) >= 0.6


# Global response generator instance
response_generator = ResponseGenerator()