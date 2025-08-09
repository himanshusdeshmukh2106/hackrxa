"""
Embedding generation service using sentence transformers
"""
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import EmbeddingGenerationError
from app.schemas.models import TextChunk


class EmbeddingService(LoggerMixin):
    """Service for generating text embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize embedding service
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension = settings.embedding_dimension
        self._model_loaded = False
    
    async def initialize(self) -> None:
        """Initialize the embedding model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            
            # Verify embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            actual_dimension = test_embedding.shape[1]
            
            if actual_dimension != self.embedding_dimension:
                self.logger.warning(
                    f"Model dimension {actual_dimension} differs from configured {self.embedding_dimension}"
                )
                self.embedding_dimension = actual_dimension
            
            self._model_loaded = True
            self.logger.info(f"Embedding model loaded successfully (dimension: {self.embedding_dimension})")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingGenerationError(f"Model initialization failed: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not self._model_loaded:
            await self.initialize()
        
        if not texts:
            return []
        
        try:
            # Check cache first if enabled
            from app.core.cache import cache_manager
            cached_embeddings = []
            texts_to_generate = []
            cache_keys = []
            
            if hasattr(cache_manager, 'embedding_cache'):
                for text in texts:
                    cache_key = f"embedding:{hash(text.strip())}"
                    cache_keys.append(cache_key)
                    cached = await cache_manager.embedding_cache.get(cache_key)
                    if cached is not None:
                        cached_embeddings.append(cached)
                        texts_to_generate.append(None)  # Placeholder
                    else:
                        cached_embeddings.append(None)
                        texts_to_generate.append(text)
            else:
                texts_to_generate = texts
                cache_keys = []
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if any(t is not None for t in texts_to_generate):
                # Validate and clean texts
                clean_texts = []
                for text in texts_to_generate:
                    if text is None:
                        continue
                    
                    if not isinstance(text, str):
                        raise EmbeddingGenerationError(f"Text is not a string: {type(text)}")
                    
                    clean_text = text.strip()
                    if not clean_text:
                        self.logger.warning(f"Empty text, using placeholder")
                        clean_text = "[EMPTY]"
                    
                    clean_texts.append(clean_text)
                
                if clean_texts:
                    self.logger.info(f"Generating embeddings for {len(clean_texts)} texts")
                    
                    # Generate embeddings in thread pool
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(
                            clean_texts,
                            convert_to_numpy=True,
                            show_progress_bar=len(clean_texts) > 10,
                            batch_size=64
                        )
                    )
                    
                    # Convert to list of lists
                    new_embeddings = embeddings.tolist()
                    
                    # Validate dimensions
                    for i, embedding in enumerate(new_embeddings):
                        if len(embedding) != self.embedding_dimension:
                            raise EmbeddingGenerationError(
                                f"Embedding {i} has dimension {len(embedding)}, expected {self.embedding_dimension}"
                            )
                    
                    # Cache new embeddings
                    if hasattr(cache_manager, 'embedding_cache') and cache_keys:
                        new_idx = 0
                        for i, text in enumerate(texts_to_generate):
                            if text is not None and new_idx < len(new_embeddings):
                                await cache_manager.embedding_cache.set(
                                    cache_keys[i], 
                                    new_embeddings[new_idx]
                                )
                                new_idx += 1
            
            # Combine cached and new embeddings
            final_embeddings = []
            new_idx = 0
            
            if cache_keys:
                for i, cached in enumerate(cached_embeddings):
                    if cached is not None:
                        final_embeddings.append(cached)
                    elif new_idx < len(new_embeddings):
                        final_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
            else:
                final_embeddings = new_embeddings
            
            self.logger.info(f"Successfully generated {len(final_embeddings)} embeddings")
            return final_embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {str(e)}")
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def embed_text_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for text chunks and update them in place
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of TextChunk objects with embeddings added
        """
        if not chunks:
            return chunks
        
        try:
            # Extract texts from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                chunk.metadata["embedding_model"] = self.model_name
                chunk.metadata["embedding_dimension"] = len(embedding)
            
            self.logger.info(f"Added embeddings to {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to embed text chunks: {str(e)}")
            raise EmbeddingGenerationError(f"Chunk embedding failed: {str(e)}")
    
    async def embed_text_chunks_fast(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Fast embedding generation with optimizations
        """
        if not chunks:
            return []
        
        try:
            # Use larger batch size for speed
            batch_size = min(96, len(chunks))  # Even larger batches for speed
            
            # Process in parallel batches
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                task = asyncio.create_task(self._process_batch_fast(batch))
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            result_chunks = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self.logger.error(f"Batch processing failed: {batch_result}")
                    continue
                result_chunks.extend(batch_result)
            
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"Fast chunk embedding failed: {str(e)}")
            # Fallback to regular method
            return await self.embed_text_chunks(chunks)
    
    async def _process_batch_fast(self, batch: List[TextChunk]) -> List[TextChunk]:
        """Process a batch of chunks quickly"""
        texts = [chunk.content for chunk in batch]
        embeddings = await self.generate_embeddings(texts)
        
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding
        
        return batch
    
    async def batch_process_chunks(
        self, 
        chunks: List[TextChunk], 
        batch_size: int = 50
    ) -> List[TextChunk]:
        """
        Process chunks in batches for memory efficiency
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Number of chunks to process per batch
            
        Returns:
            List of TextChunk objects with embeddings
        """
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                embedded_batch = await self.embed_text_chunks(batch)
                processed_chunks.extend(embedded_batch)
                
                # Small delay between batches to prevent overload
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Batch processing failed for batch starting at {i}: {str(e)}")
                # Continue with next batch instead of failing completely
                processed_chunks.extend(batch)  # Add without embeddings
        
        return processed_chunks
    
    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        """
        Validate that embedding has correct dimension
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if dimension is correct, False otherwise
        """
        return len(embedding) == self.embedding_dimension
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            await self.initialize()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
            "loaded": self._model_loaded
        }
    
    async def cleanup(self) -> None:
        """Clean up model resources"""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self._model_loaded = False
            self.logger.info("Embedding model resources cleaned up")


# Global embedding service instance
embedding_service = EmbeddingService()