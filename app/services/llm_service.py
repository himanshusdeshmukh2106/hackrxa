"""
Gemini 2.5 Pro LLM service wrapper
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import LLMServiceError
from app.schemas.models import SearchResult, QueryIntent


@dataclass
class LLMRequest:
    """Request model for LLM operations"""
    context: str
    query: str
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """Response model from LLM operations"""
    content: str
    tokens_used: int
    processing_time_ms: int
    model_name: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


class LLMService(LoggerMixin):
    """Service for interacting with Gemini 2.5 Pro LLM"""
    
    def __init__(self):
        self.model_name = "gemini-2.5-pro"
        self.model = None
        self._initialized = False
        self.max_context_length = 1000000  # Gemini 2.5 Pro context window
        self.prompt_templates = self._initialize_prompt_templates()
    
    async def initialize(self) -> None:
        """Initialize Gemini client"""
        try:
            self.logger.info("Initializing Gemini 2.5 Pro LLM service")
            
            # Configure Gemini
            genai.configure(api_key=settings.gemini_api_key)
            
            # Initialize model with safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=4096,
            )
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            self._initialized = True
            self.logger.info(f"Gemini LLM service initialized with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise LLMServiceError(f"Gemini initialization failed: {str(e)}")
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different query types"""
        return {
            "answer_generation": """You are an expert document analyst specializing in insurance, legal, HR, and compliance documents. 
Your task is to provide accurate, precise answers based solely on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say "The provided context does not contain sufficient information to answer this question."
3. Be specific and cite relevant sections when possible
4. Use clear, professional language
5. If there are multiple relevant pieces of information, organize them logically
6. Include specific details like amounts, timeframes, conditions, and limitations when mentioned in the context

ANSWER:""",

            "coverage_analysis": """You are an insurance policy expert. Analyze the provided context to determine coverage details.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Determine if the item/condition is covered, not covered, or partially covered
2. List any conditions, limitations, or requirements for coverage
3. Mention any waiting periods, deductibles, or sub-limits
4. Cite specific policy sections or clauses when available
5. If coverage depends on specific circumstances, explain those circumstances

COVERAGE ANALYSIS:""",

            "definition_extraction": """You are a legal document expert. Extract and explain definitions from the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide the exact definition as stated in the document
2. If multiple definitions exist, list all relevant ones
3. Include any conditions or qualifications mentioned
4. Explain the definition in simpler terms if it's complex
5. Note if the definition has specific scope or limitations

DEFINITION:""",

            "timeline_analysis": """You are a policy expert specializing in timelines and deadlines. Analyze time-related information.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Extract all relevant time periods, deadlines, or waiting periods
2. Specify what triggers the timeline (e.g., policy start, claim filing, etc.)
3. Note any conditions that might affect the timeline
4. Distinguish between different types of periods (waiting, grace, notice, etc.)
5. Include any consequences of missing deadlines

TIMELINE ANALYSIS:""",

            "amount_extraction": """You are a financial analyst expert. Extract and analyze monetary amounts and financial terms.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Extract all relevant amounts, fees, premiums, or financial figures
2. Specify what each amount relates to (premium, deductible, coverage limit, etc.)
3. Note any conditions that affect the amounts
4. Include percentage-based calculations if mentioned
5. Mention any variations based on different scenarios

FINANCIAL ANALYSIS:""",

            "batch_processing": """You are a document analysis expert. Answer multiple questions based on the provided context.

CONTEXT:
{context}

QUESTIONS:
{questions}

INSTRUCTIONS:
1. Answer each question separately and clearly
2. Number your answers to correspond with the question numbers
3. Use only the information provided in the context
4. If a question cannot be answered from the context, state this clearly
5. Be concise but complete in your answers
6. Maintain consistency across all answers

ANSWERS:"""
        }
    
    def _get_response_text(self, response: Any) -> str:
        """Safely extract text from a Gemini response."""
        try:
            if hasattr(response, 'parts') and response.parts:
                return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
        except Exception as e:
            self.logger.warning(f"Could not extract text from response: {e}")
            return ""

    async def generate_response(
        self, 
        context: str, 
        query: str, 
        query_intent: Optional[QueryIntent] = None,
        temperature: float = 0.1
    ) -> LLMResponse:
        """
        Generate response using Gemini 2.5 Pro
        
        Args:
            context: Document context for the query
            query: User question
            query_intent: Optional analyzed query intent
            temperature: Generation temperature
            
        Returns:
            LLMResponse object
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            start_time = datetime.utcnow()
            
            # Select appropriate prompt template
            prompt_template = self._select_prompt_template(query_intent)
            
            # Optimize context length
            optimized_context = self._optimize_context(context)
            
            # Format prompt
            formatted_prompt = prompt_template.format(
                context=optimized_context,
                question=query
            )
            
            self.logger.info(f"Generating response for query: {query[:100]}...")
            
            # Generate response
            response = await self._generate_with_retry(formatted_prompt, temperature)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response_text = self._get_response_text(response)
            # Extract token usage (approximate)
            tokens_used = self._estimate_token_usage(formatted_prompt, response_text)
            
            llm_response = LLMResponse(
                content=response_text.strip(),
                tokens_used=tokens_used,
                processing_time_ms=int(processing_time),
                model_name=self.model_name,
                confidence=self._calculate_confidence(response),
                metadata={
                    "prompt_template": prompt_template.__name__ if hasattr(prompt_template, '__name__') else "default",
                    "context_length": len(optimized_context),
                    "query_intent": query_intent.intent_type if query_intent else None
                }
            )
            
            self.logger.info(f"Generated response in {processing_time:.0f}ms using {tokens_used} tokens")
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")
    
    async def batch_process(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Process multiple requests efficiently
        
        Args:
            requests: List of LLM requests
            
        Returns:
            List of LLM responses
        """
        if not requests:
            return []
        
        try:
            self.logger.info(f"Processing batch of {len(requests)} requests")
            
            # For single context with multiple questions, use batch template
            if len(requests) > 1 and all(req.context == requests[0].context for req in requests):
                return await self._batch_process_same_context(requests)
            
            # Process individual requests concurrently
            tasks = []
            for request in requests:
                task = asyncio.create_task(
                    self.generate_response(
                        context=request.context,
                        query=request.query,
                        temperature=request.temperature
                    )
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Request {i} failed: {str(response)}")
                    # Create error response
                    error_response = LLMResponse(
                        content=f"Error processing request: {str(response)}",
                        tokens_used=0,
                        processing_time_ms=0,
                        model_name=self.model_name,
                        confidence=0.0,
                        metadata={"error": True}
                    )
                    processed_responses.append(error_response)
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise LLMServiceError(f"Batch processing failed: {str(e)}")
    
    async def _batch_process_same_context(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple questions with the same context efficiently"""
        try:
            context = requests[0].context
            questions = [f"{i+1}. {req.query}" for i, req in enumerate(requests)]
            questions_text = "\n".join(questions)
            
            # Use batch processing template
            prompt_template = self.prompt_templates["batch_processing"]
            optimized_context = self._optimize_context(context)
            
            formatted_prompt = prompt_template.format(
                context=optimized_context,
                questions=questions_text
            )
            
            start_time = datetime.utcnow()
            
            # Generate single response for all questions
            response = await self._generate_with_retry(formatted_prompt, requests[0].temperature)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            response_text = self._get_response_text(response)
            tokens_used = self._estimate_token_usage(formatted_prompt, response_text)
            
            # Parse individual answers from the response
            answers = self._parse_batch_response(response_text, len(requests))
            
            # Create individual responses
            responses = []
            for i, answer in enumerate(answers):
                llm_response = LLMResponse(
                    content=answer.strip(),
                    tokens_used=tokens_used // len(requests),  # Distribute token usage
                    processing_time_ms=int(processing_time // len(requests)),
                    model_name=self.model_name,
                    confidence=self._calculate_confidence(response),
                    metadata={
                        "batch_processed": True,
                        "batch_size": len(requests),
                        "question_index": i
                    }
                )
                responses.append(llm_response)
            
            self.logger.info(f"Batch processed {len(requests)} questions in {processing_time:.0f}ms")
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch processing same context failed: {str(e)}")
            # Fallback to individual processing
            return await self.batch_process(requests)
    
    def _select_prompt_template(self, query_intent: Optional[QueryIntent]) -> str:
        """Select appropriate prompt template based on query intent"""
        if not query_intent:
            return self.prompt_templates["answer_generation"]
        
        intent_type = query_intent.intent_type.lower()
        
        if intent_type == "coverage":
            return self.prompt_templates["coverage_analysis"]
        elif intent_type == "definition":
            return self.prompt_templates["definition_extraction"]
        elif intent_type == "timeline":
            return self.prompt_templates["timeline_analysis"]
        elif intent_type == "amount":
            return self.prompt_templates["amount_extraction"]
        else:
            return self.prompt_templates["answer_generation"]
    
    def _optimize_context(self, context: str) -> str:
        """Optimize context length for token efficiency"""
        if len(context) <= self.max_context_length:
            return context
        
        # Simple truncation strategy - in production, use more sophisticated methods
        # like extractive summarization or chunk ranking
        
        self.logger.warning(f"Context too long ({len(context)} chars), truncating to {self.max_context_length}")
        
        # Try to truncate at sentence boundaries
        truncated = context[:self.max_context_length]
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > self.max_context_length * 0.8:
            return truncated[:last_sentence + 1]
        else:
            return truncated
    
    async def _generate_with_retry(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> Any:
        """Generate response with retry logic"""
        for attempt in range(max_retries):
            try:
                # Update generation config for this request
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=4096,
                )
                
                # Generate response
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                )
                
                if response.text:
                    return response
                else:
                    raise LLMServiceError("Empty response from Gemini")
                
            except Exception as e:
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise LLMServiceError(f"Failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise LLMServiceError("Maximum retries exceeded")
    
    def _estimate_token_usage(self, prompt: str, response: str) -> int:
        """Estimate token usage (approximate)"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score for the response"""
        # This is a simplified confidence calculation
        # In production, you might use more sophisticated methods
        
        try:
            # Check if response has safety ratings or other confidence indicators
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check for safety issues
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        if rating.probability.name in ['HIGH', 'MEDIUM']:
                            return 0.3  # Low confidence due to safety concerns
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason.name == 'STOP':
                        return 0.8  # Normal completion
                    elif candidate.finish_reason.name == 'MAX_TOKENS':
                        return 0.6  # Truncated response
                    else:
                        return 0.4  # Other issues
            
            # Default confidence based on response length and content
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            if len(response_text) < 10:
                return 0.3  # Very short response
            elif "I don't know" in response_text or "cannot answer" in response_text:
                return 0.5  # Uncertain response
            elif len(response_text) > 50:
                return 0.8  # Detailed response
            else:
                return 0.7  # Standard response
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5  # Default confidence
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse batch response into individual answers"""
        try:
            # Look for numbered answers
            import re
            
            # Pattern to match numbered answers
            pattern = r'(\d+)\.\s*(.*?)(?=\n\d+\.|$)'
            matches = re.findall(pattern, response_text, re.DOTALL)
            
            if len(matches) >= expected_count:
                return [match[1].strip() for match in matches[:expected_count]]
            
            # Fallback: split by double newlines
            parts = response_text.split('\n\n')
            if len(parts) >= expected_count:
                return parts[:expected_count]
            
            # Last resort: split by single newlines and group
            lines = response_text.split('\n')
            answers = []
            current_answer = []
            
            for line in lines:
                if re.match(r'^\d+\.', line.strip()) and current_answer:
                    answers.append('\n'.join(current_answer).strip())
                    current_answer = [line]
                else:
                    current_answer.append(line)
            
            if current_answer:
                answers.append('\n'.join(current_answer).strip())
            
            # Pad with empty answers if needed
            while len(answers) < expected_count:
                answers.append("Unable to extract answer from batch response.")
            
            return answers[:expected_count]
            
        except Exception as e:
            self.logger.error(f"Failed to parse batch response: {str(e)}")
            # Return generic answers
            return [f"Answer {i+1}: Unable to parse response." for i in range(expected_count)]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model"""
        return {
            "model_name": self.model_name,
            "max_context_length": self.max_context_length,
            "initialized": self._initialized,
            "available_templates": list(self.prompt_templates.keys())
        }
    
    async def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Test with a simple query
            test_response = await self.generate_response(
                context="This is a test document.",
                query="What is this document about?",
                temperature=0.1
            )
            
            return bool(test_response.content.strip())
            
        except Exception as e:
            self.logger.error(f"LLM health check failed: {str(e)}")
            return False


# Global LLM service instance
llm_service = LLMService()