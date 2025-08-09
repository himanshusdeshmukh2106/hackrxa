"""
Semantic search engine combining vector similarity and keyword matching
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter
import math

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import VectorStoreError
from app.schemas.models import SearchResult, QueryIntent
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.services.database import db_manager


class SearchEngine(LoggerMixin):
    """Hybrid semantic search engine"""
    
    def __init__(self):
        self.similarity_threshold = settings.similarity_threshold
        self.max_results = settings.max_search_results
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7
    
    async def semantic_search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query string
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.max_results
        
        self.logger.info(f"Performing semantic search for: {query[:100]}...")
        
        query_embedding = await embedding_service.generate_single_embedding(query)
        
        if not query_embedding:
            self.logger.warning("Failed to generate query embedding")
            return []
        
        # Search in vector store
                # Search in vector store
        query_embedding = await embedding_service.generate_single_embedding(query)
        
        if not query_embedding:
            self.logger.warning("Failed to generate query embedding")
            return []
        
        # Search in vector store
        matches = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            document_id=filters.get("document_id") if filters else None,
            top_k=top_k * 2  # Get more results for reranking
        )
        
        return matches[:top_k]
    
    async def keyword_search(
        self, 
        query: str, 
        document_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform keyword-based search using PostgreSQL full-text search
        
        Args:
            query: Search query string
            document_id: Optional document ID filter
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            top_k = top_k or self.max_results
            
            self.logger.info(f"Performing keyword search for: {query[:100]}...")
            
            keywords = self._extract_keywords(query)
            if not keywords:
                return []

            return await self._postgresql_fulltext_search(keywords, document_id, top_k)
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        query_intent: Optional[QueryIntent] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword approaches
        
        Args:
            query: Search query string
            query_intent: Optional analyzed query intent
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of ranked SearchResult objects
        """
        try:
            top_k = top_k or self.max_results
            
            self.logger.info(f"Performing hybrid search for: {query[:100]}...")
            self.logger.info(f"Filters: {filters}")
            self.logger.info(f"Similarity threshold: {self.similarity_threshold}")
            
            semantic_results = await self.semantic_search(query, filters, top_k * 2)
            keyword_results = await self.keyword_search(
                query, filters.get("document_id") if filters else None, top_k
            )
            
            # Handle exceptions
            if isinstance(semantic_results, Exception):
                self.logger.error(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            
            if isinstance(keyword_results, Exception):
                self.logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            
            self.logger.info(f"Semantic results: {len(semantic_results)}")
            self.logger.info(f"Keyword results: {len(keyword_results)}")
            
            # Log first semantic result for debugging
            if semantic_results:
                first_result = semantic_results[0]
                self.logger.info(f"First semantic result score: {first_result.similarity_score}")
                self.logger.info(f"First semantic result content: {first_result.content[:100]}...")
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, query, query_intent
            )
            
            # Apply final ranking
                        # Apply final ranking
            ranked_results = await self.rank_results(combined_results, query, query_intent)
            
            self.logger.info(f"Hybrid search returned {len(ranked_results)} results")
            
            return ranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Convert to lowercase and remove punctuation
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into words
        words = clean_query.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'what', 'when', 'where', 'why', 'how', 'who', 'which'
        }
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return keywords
    
    async def _postgresql_fulltext_search(
        self, 
        keywords: List[str], 
        document_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform PostgreSQL full-text search (fallback implementation)
        
        Args:
            keywords: List of keywords to search
            document_id: Optional document ID filter
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # This is a simplified implementation
        # In a real system, this would use PostgreSQL's full-text search capabilities
        
        try:
            # For now, return empty results as this requires actual database implementation
            # In production, this would execute SQL queries like:
            # SELECT * FROM text_chunks WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            
            self.logger.info(f"PostgreSQL search for keywords: {keywords}")
            return []
            
        except Exception as e:
            self.logger.error(f"PostgreSQL search failed: {str(e)}")
            return []
    
    def _combine_search_results(
        self, 
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        query: str,
        query_intent: Optional[QueryIntent] = None
    ) -> List[SearchResult]:
        """
        Combine results from semantic and keyword searches
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            query: Original query
            query_intent: Analyzed query intent
            
        Returns:
            Combined list of SearchResult objects
        """
        # Create a dictionary to merge results by chunk_id
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            result.relevance_explanation = f"Semantic similarity: {result.similarity_score:.3f}"
            combined[result.chunk_id] = result
        
        # Add keyword results, combining scores if chunk already exists
        for result in keyword_results:
            if result.chunk_id in combined:
                # Combine scores using weighted average
                existing = combined[result.chunk_id]
                existing.similarity_score = (
                    existing.similarity_score * self.semantic_weight +
                    result.similarity_score * self.keyword_weight
                )
                existing.relevance_explanation += f" + Keyword match: {result.similarity_score:.3f}"
            else:
                result.relevance_explanation = f"Keyword match: {result.similarity_score:.3f}"
                combined[result.chunk_id] = result
        
        return list(combined.values())
    
    async def rank_results(
        self, 
        results: List[SearchResult],
        query: str,
        query_intent: Optional[QueryIntent] = None
    ) -> List[SearchResult]:
        """
        Apply advanced ranking to search results
        
        Args:
            results: List of search results to rank
            query: Original query
            query_intent: Analyzed query intent
            
        Returns:
            Ranked list of SearchResult objects
        """
        if not results:
            return results
        
        try:
            # Calculate additional ranking factors
            for result in results:
                # Base score from similarity
                base_score = result.similarity_score
                
                # Content length factor (prefer moderate length chunks)
                content_length = len(result.content)
                length_factor = self._calculate_length_factor(content_length)
                
                # Query term frequency factor
                tf_factor = self._calculate_tf_factor(result.content, query)
                
                # Intent relevance factor
                intent_factor = self._calculate_intent_factor(result, query_intent)
                
                # Position factor (if available in metadata)
                position_factor = self._calculate_position_factor(result)
                
                # Combine all factors
                final_score = (
                    base_score * 0.4 +
                    length_factor * 0.1 +
                    tf_factor * 0.2 +
                    intent_factor * 0.2 +
                    position_factor * 0.1
                )
                
                result.similarity_score = final_score
                
                # Update explanation
                result.relevance_explanation += (
                    f" (final score: {final_score:.3f}, "
                    f"length: {length_factor:.2f}, "
                    f"tf: {tf_factor:.2f}, "
                    f"intent: {intent_factor:.2f})"
                )
            
            # Sort by final score
            ranked_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
            
            self.logger.info(f"Ranked {len(ranked_results)} search results")
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Result ranking failed: {str(e)}")
            # Return original results if ranking fails
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def _calculate_length_factor(self, content_length: int) -> float:
        """Calculate scoring factor based on content length"""
        # Prefer chunks that are not too short or too long
        optimal_length = 500
        
        if content_length < 50:
            return 0.5  # Too short
        elif content_length > 2000:
            return 0.7  # Too long
        else:
            # Gaussian-like curve with peak at optimal length
            diff = abs(content_length - optimal_length)
            return max(0.5, 1.0 - (diff / optimal_length) * 0.5)
    
    def _calculate_tf_factor(self, content: str, query: str) -> float:
        """Calculate term frequency factor"""
        query_terms = self._extract_keywords(query)
        
        if not query_terms:
            return 0.5
        
        content_lower = content.lower()
        
        # Count term occurrences
        term_counts = []
        for term in query_terms:
            count = content_lower.count(term.lower())
            if count > 0:
                # Use log to dampen high frequencies
                term_counts.append(math.log(1 + count))
        
        if not term_counts:
            return 0.0
        
        # Average term frequency score
        avg_tf = sum(term_counts) / len(query_terms)
        
        # Normalize to 0-1 range
        return min(1.0, avg_tf / 3.0)
    
    def _calculate_intent_factor(
        self, 
        result: SearchResult, 
        query_intent: Optional[QueryIntent]
    ) -> float:
        """Calculate intent relevance factor"""
        if not query_intent:
            return 0.5
        
        content_lower = result.content.lower()
        intent_boost = 0.5
        
        # Boost based on intent type
        if query_intent.intent_type == "coverage":
            coverage_terms = ["cover", "coverage", "include", "benefit", "protection"]
            if any(term in content_lower for term in coverage_terms):
                intent_boost += 0.3
        
        elif query_intent.intent_type == "exclusion":
            exclusion_terms = ["exclude", "exclusion", "not cover", "limitation", "restriction"]
            if any(term in content_lower for term in exclusion_terms):
                intent_boost += 0.3
        
        elif query_intent.intent_type == "definition":
            definition_terms = ["means", "defined as", "refers to", "definition"]
            if any(term in content_lower for term in definition_terms):
                intent_boost += 0.3
        
        elif query_intent.intent_type == "amount":
            amount_terms = ["amount", "cost", "price", "fee", "premium", "$", "₹"]
            if any(term in content_lower for term in amount_terms):
                intent_boost += 0.3
        
        elif query_intent.intent_type == "timeline":
            time_terms = ["period", "time", "days", "months", "years", "when", "duration"]
            if any(term in content_lower for term in time_terms):
                intent_boost += 0.3
        
        # Boost based on extracted entities
        for entity in query_intent.entities:
            if entity.text.lower() in content_lower:
                intent_boost += 0.1
        
        return min(1.0, intent_boost)
    
    def _calculate_position_factor(self, result: SearchResult) -> float:
        """Calculate position-based factor"""
        # Prefer chunks from the beginning of documents
        chunk_index = result.document_metadata.get("chunk_index", 0)
        
        if chunk_index == 0:
            return 1.0  # First chunk
        elif chunk_index < 5:
            return 0.9  # Early chunks
        elif chunk_index < 10:
            return 0.8  # Middle chunks
        else:
            return 0.7  # Later chunks
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on partial query
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested completions
        """
        try:
            # This would typically query a suggestions index or use autocomplete
            # For now, return some common search patterns
            
            suggestions = []
            partial_lower = partial_query.lower().strip()
            
            common_patterns = [
                "coverage for",
                "waiting period for",
                "premium for",
                "exclusions for",
                "benefits of",
                "conditions for",
                "definition of",
                "how to claim",
                "documents required for",
                "eligibility for"
            ]
            
            for pattern in common_patterns:
                if pattern.startswith(partial_lower) or partial_lower in pattern:
                    suggestions.append(pattern)
            
            return suggestions[:5]
            
        except Exception as e:
            self.logger.error(f"Search suggestions failed: {str(e)}")
            return []
    
    async def explain_search_results(
        self, 
        results: List[SearchResult], 
        query: str
    ) -> Dict[str, Any]:
        """
        Provide explanation for search results
        
        Args:
            results: Search results to explain
            query: Original query
            
        Returns:
            Dictionary with search explanation
        """
        if not results:
            return {
                "total_results": 0,
                "explanation": "No relevant results found for the query.",
                "suggestions": await self.get_search_suggestions(query)
            }
        
        # Analyze result distribution
        score_distribution = [r.similarity_score for r in results]
        avg_score = sum(score_distribution) / len(score_distribution)
        max_score = max(score_distribution)
        min_score = min(score_distribution)
        
        # Count results by score ranges
        high_relevance = len([s for s in score_distribution if s >= 0.8])
        medium_relevance = len([s for s in score_distribution if 0.6 <= s < 0.8])
        low_relevance = len([s for s in score_distribution if s < 0.6])
        
        explanation = {
            "total_results": len(results),
            "score_statistics": {
                "average": round(avg_score, 3),
                "maximum": round(max_score, 3),
                "minimum": round(min_score, 3)
            },
            "relevance_distribution": {
                "high_relevance": high_relevance,
                "medium_relevance": medium_relevance,
                "low_relevance": low_relevance
            },
            "search_method": "hybrid_semantic_keyword",
            "explanation": f"Found {len(results)} relevant chunks with average relevance score of {avg_score:.3f}"
        }
        
        return explanation


# Global search engine instance
search_engine = SearchEngine()