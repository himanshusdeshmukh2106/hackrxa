"""
Pinecone vector store integration service with performance optimizations
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uuid
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import VectorStoreError
from app.schemas.models import TextChunk, SearchResult


class Vector:
    """Vector data model for Pinecone operations"""
    
    def __init__(self, id: str, values: List[float], metadata: Dict[str, Any] = None):
        self.id = id
        self.values = values
        self.metadata = metadata or {}


class Match:
    """Search match result from Pinecone"""
    
    def __init__(self, id: str, score: float, values: List[float] = None, metadata: Dict[str, Any] = None):
        self.id = id
        self.score = score
        self.values = values or []
        self.metadata = metadata or {}


class VectorStore(LoggerMixin):
    """Pinecone vector store service with async optimizations"""
    
    def __init__(self):
        self.sync_client = None  # For management operations
        self.index = None  # Direct index reference for faster access
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.embedding_dimension
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=12)  # Further increased for single question speed
        self._lock = threading.Lock()
        self._query_cache = {}  # Simple query cache
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    async def initialize(self) -> None:
        """Initialize Pinecone client and index with optimizations"""
        if self._initialized:
            return
            
        try:
            self.logger.info("Initializing Pinecone vector store")
            
            # Initialize sync Pinecone client for management operations
            self.sync_client = Pinecone(api_key=settings.pinecone_api_key)
            
            # Check if index exists, create if not (run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._ensure_index_exists_sync
            )
            
            # Get direct index reference for faster operations
            self.index = self.sync_client.Index(self.index_name)
            
            self._initialized = True
            self.logger.info(f"Pinecone vector store initialized with index: {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise VectorStoreError(f"Pinecone initialization failed: {str(e)}")
    
    def _ensure_index_exists_sync(self) -> None:
        """Synchronous version of index creation for thread pool execution"""
        try:
            # Check if index exists (sync operation)
            if self.index_name not in self.sync_client.list_indexes().names():
                self.logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create index with serverless spec (sync operation)
                self.sync_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.pinecone_environment
                    ),
                    deletion_protection="disabled"
                )
                
                self.logger.info(f"Pinecone index created: {self.index_name}")
            else:
                self.logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure index exists: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")
    
    async def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if not"""
        try:
            # Check if index exists (sync operation)
            if self.index_name not in self.sync_client.list_indexes().names():
                self.logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create index with serverless spec (sync operation)
                self.sync_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.pinecone_environment
                    ),
                    deletion_protection="disabled"
                )
                
                # Wait for index to be ready
                await self._wait_for_index_ready()
                
                self.logger.info(f"Pinecone index created: {self.index_name}")
            else:
                self.logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure index exists: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")
    
    async def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            try:
                index_info = await self.client.describe_index(self.index_name)
                if index_info.status.ready:
                    return
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error checking index status: {str(e)}")
                await asyncio.sleep(2)
        
        raise VectorStoreError(f"Index {self.index_name} not ready after {timeout} seconds")
    
    async def upsert_vectors(self, vectors: List[Vector]) -> bool:
        """
        Upsert vectors to Pinecone index with async optimization
        
        Args:
            vectors: List of Vector objects to upsert
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            VectorStoreError: If upsert operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not vectors:
            return True
        
        try:
            # Convert Vector objects to Pinecone format
            pinecone_vectors = []
            for vector in vectors:
                pinecone_vector = {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata
                }
                pinecone_vectors.append(pinecone_vector)
            
            self.logger.info(f"Upserting {len(pinecone_vectors)} vectors to Pinecone")
            
            # Use thread pool for async execution
            await asyncio.get_event_loop().run_in_executor(
                self._executor, 
                self._upsert_vectors_sync, 
                pinecone_vectors
            )
            
            self.logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Vector upsert failed: {str(e)}")
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")
    
    def _upsert_vectors_sync(self, pinecone_vectors: List[Dict]) -> None:
        """Synchronous upsert for thread pool execution"""
        # Use batch size of 100 for optimal performance
        batch_size = 100
        for i in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Match]:
        """
        Perform similarity search in Pinecone with async optimization
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            filter_dict: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of Match objects
            
        Raises:
            VectorStoreError: If search operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare query parameters
            query_params = {
                "vector": query_vector,
                "top_k": min(top_k, 50),  # Reduced for speed
                "include_metadata": include_metadata
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Use thread pool for async execution
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._query_sync,
                query_params
            )
            
            # Convert response to Match objects
            matches = []
            for match in response.matches:
                match_obj = Match(
                    id=match.id,
                    score=match.score,
                    values=getattr(match, 'values', []),
                    metadata=getattr(match, 'metadata', {})
                )
                matches.append(match_obj)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(f"Search operation failed: {str(e)}")

    def _query_sync(self, query_params: Dict) -> Any:
        """Synchronous query for thread pool execution with caching"""
        # Simple cache key based on vector and top_k
        cache_key = f"{hash(str(query_params.get('vector', [])[:10]))}-{query_params.get('top_k', 10)}"
        
        # Check cache
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                return cache_entry['result']
        
        # Execute query
        result = self.index.query(**query_params)
        
        # Cache result
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Clean old cache entries (simple cleanup)
        if len(self._query_cache) > 100:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k]['timestamp'])
            del self._query_cache[oldest_key]
        
        return result
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone index
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If delete operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not vector_ids:
            return True
        
        try:
            self.logger.info(f"Deleting {len(vector_ids)} vectors from Pinecone")
            
            # Use sync client for now (will be made async later)
            index = self.sync_client.Index(self.index_name)
            index.delete(ids=vector_ids)
            
            self.logger.info(f"Successfully deleted {len(vector_ids)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Vector deletion failed: {str(e)}")
            raise VectorStoreError(f"Failed to delete vectors: {str(e)}")
    
    async def store_text_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Store text chunks as vectors in Pinecone with fast batch processing
        
        Args:
            chunks: List of TextChunk objects with embeddings
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If storage fails
        """
        if not chunks:
            return True
        
        # Validate that chunks have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
        
        if not chunks_with_embeddings:
            raise VectorStoreError("No chunks with embeddings found")
        
        if len(chunks_with_embeddings) != len(chunks):
            self.logger.warning(
                f"Only {len(chunks_with_embeddings)} of {len(chunks)} chunks have embeddings"
            )
        
        try:
            # Convert chunks to vectors with minimal metadata for speed
            vectors = []
            for chunk in chunks_with_embeddings:
                vector = Vector(
                    id=chunk.id,
                    values=chunk.embedding,
                    metadata={
                        "document_id": chunk.document_id,
                        "content": chunk.content[:500],  # Reduced for speed
                        "chunk_index": chunk.chunk_index,
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
                vectors.append(vector)
            
            # Store vectors with fast batch processing
            return await self.upsert_vectors_fast(vectors)
            
        except Exception as e:
            self.logger.error(f"Failed to store text chunks: {str(e)}")
            raise VectorStoreError(f"Chunk storage failed: {str(e)}")
    
    async def upsert_vectors_fast(self, vectors: List[Vector]) -> bool:
        """
        Fast batch upsert with parallel processing
        """
        if not vectors:
            return True
            
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert to Pinecone format
            pinecone_vectors = [
                {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata
                }
                for vector in vectors
            ]
            
            # Process in parallel batches for maximum speed
            batch_size = 100
            tasks = []
            
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                task = asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._upsert_batch_sync,
                    batch
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
            self.logger.info(f"Fast upserted {len(vectors)} vectors in {len(tasks)} batches")
            return True
            
        except Exception as e:
            self.logger.error(f"Fast vector upsert failed: {str(e)}")
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")
    
    def _upsert_batch_sync(self, batch: List[Dict]) -> None:
        """Synchronous batch upsert for thread pool execution"""
        self.index.upsert(vectors=batch)
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar text chunks
        
        Args:
            query_embedding: Query embedding vector
            document_id: Optional document ID filter
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Prepare filter
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": {"$eq": document_id}}
            
            # Perform search
            matches = await self.similarity_search(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # Convert to SearchResult objects
            search_results = []
            for match in matches:
                result = SearchResult(
                    chunk_id=match.id,
                    content=match.metadata.get("content", ""),
                    similarity_score=match.score,
                    document_metadata={
                        "document_id": match.metadata.get("document_id"),
                        "chunk_index": match.metadata.get("chunk_index"),
                        "start_char": match.metadata.get("start_char"),
                        "end_char": match.metadata.get("end_char")
                    },
                    relevance_explanation=f"Cosine similarity: {match.score:.3f}"
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Chunk search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def _query_sync(self, query_params: Dict) -> Any:
        """Synchronous query for thread pool execution with caching"""
        # Simple cache key based on vector and top_k
        cache_key = f"{hash(str(query_params.get('vector', [])[:10]))}-{query_params.get('top_k', 10)}"
        
        # Check cache
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                return cache_entry['result']
        
        # Execute query
        result = self.index.query(**query_params)
        
        # Cache result
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Clean old cache entries (simple cleanup)
        if len(self._query_cache) > 100:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k]['timestamp'])
            del self._query_cache[oldest_key]
        
        return result
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone index
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If delete operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not vector_ids:
            return True
        
        try:
            self.logger.info(f"Deleting {len(vector_ids)} vectors from Pinecone")
            
            # Use sync client for now (will be made async later)
            index = self.sync_client.Index(self.index_name)
            index.delete(ids=vector_ids)
            
            self.logger.info(f"Successfully deleted {len(vector_ids)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Vector deletion failed: {str(e)}")
            raise VectorStoreError(f"Failed to delete vectors: {str(e)}")
    
    async def store_text_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Store text chunks as vectors in Pinecone with fast batch processing
        
        Args:
            chunks: List of TextChunk objects with embeddings
            
        Returns:
            True if successful
            
        Raises:
            VectorStoreError: If storage fails
        """
        if not chunks:
            return True
        
        # Validate that chunks have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
        
        if not chunks_with_embeddings:
            raise VectorStoreError("No chunks with embeddings found")
        
        if len(chunks_with_embeddings) != len(chunks):
            self.logger.warning(
                f"Only {len(chunks_with_embeddings)} of {len(chunks)} chunks have embeddings"
            )
        
        try:
            # Convert chunks to vectors with minimal metadata for speed
            vectors = []
            for chunk in chunks_with_embeddings:
                vector = Vector(
                    id=chunk.id,
                    values=chunk.embedding,
                    metadata={
                        "document_id": chunk.document_id,
                        "content": chunk.content[:500],  # Reduced for speed
                        "chunk_index": chunk.chunk_index,
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
                vectors.append(vector)
            
            # Store vectors with fast batch processing
            return await self.upsert_vectors_fast(vectors)
            
        except Exception as e:
            self.logger.error(f"Failed to store text chunks: {str(e)}")
            raise VectorStoreError(f"Chunk storage failed: {str(e)}")
    
    async def upsert_vectors_fast(self, vectors: List[Vector]) -> bool:
        """
        Fast batch upsert with parallel processing
        """
        if not vectors:
            return True
            
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert to Pinecone format
            pinecone_vectors = [
                {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata
                }
                for vector in vectors
            ]
            
            # Process in parallel batches for maximum speed
            batch_size = 100
            tasks = []
            
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                task = asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._upsert_batch_sync,
                    batch
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
            self.logger.info(f"Fast upserted {len(vectors)} vectors in {len(tasks)} batches")
            return True
            
        except Exception as e:
            self.logger.error(f"Fast vector upsert failed: {str(e)}")
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")
    
    def _upsert_batch_sync(self, batch: List[Dict]) -> None:
        """Synchronous batch upsert for thread pool execution"""
        self.index.upsert(vectors=batch)
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar text chunks
        
        Args:
            query_embedding: Query embedding vector
            document_id: Optional document ID filter
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Prepare filter
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": {"$eq": document_id}}
            
            # Perform search
            matches = await self.similarity_search(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # Convert to SearchResult objects
            search_results = []
            for match in matches:
                result = SearchResult(
                    chunk_id=match.id,
                    content=match.metadata.get("content", ""),
                    similarity_score=match.score,
                    document_metadata={
                        "document_id": match.metadata.get("document_id"),
                        "chunk_index": match.metadata.get("chunk_index"),
                        "start_char": match.metadata.get("start_char"),
                        "end_char": match.metadata.get("end_char")
                    },
                    relevance_explanation=f"Cosine similarity: {match.score:.3f}"
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Chunk search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get Pinecone index statistics
        
        Returns:
            Dictionary with index statistics
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use sync client for now (will be made async later)
            index = self.sync_client.Index(self.index_name)
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {str(e)}")
            return {}
    
    async def health_check(self) -> bool:
        """
        Check if Pinecone service is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Simple check - verify we can access the index
            if self.index_name in self.sync_client.list_indexes().names():
                return True
            else:
                return False
            
        except Exception as e:
            self.logger.error(f"Pinecone health check failed: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close the Pinecone client connection"""
        try:
            if self.sync_client:
                # Pinecone sync client doesn't need explicit closing
                self.logger.info("Pinecone client connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Pinecone client: {str(e)}")


# Global vector store instance
vector_store = VectorStore()