"""
Performance tests for optimization features
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from app.core.cache import LRUCache, EmbeddingCache, DocumentCache, QueryCache, CacheManager
from app.core.async_utils import (
    AsyncRetry, timeout_after, gather_with_concurrency, 
    AsyncBatch, AsyncPool, async_map
)
from app.core.connection_pool import ConnectionPool, ConnectionPoolManager


class TestLRUCache:
    """Test LRU cache performance and functionality"""
    
    def setup_method(self):
        self.cache = LRUCache(max_size=100, default_ttl_seconds=3600)
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self):
        """Test basic cache operations"""
        # Set and get
        await self.cache.set("key1", "value1")
        result = await self.cache.get("key1")
        assert result == "value1"
        
        # Non-existent key
        result = await self.cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=3)
        
        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        await cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", "value4")
        
        assert await cache.get("key1") == "value1"  # Still there
        assert await cache.get("key2") is None      # Evicted
        assert await cache.get("key3") == "value3"  # Still there
        assert await cache.get("key4") == "value4"  # New key
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = LRUCache(default_ttl_seconds=1)
        
        await cache.set("key1", "value1", ttl_seconds=0.1)
        
        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with many operations"""
        cache = LRUCache(max_size=1000)
        
        # Measure set operations
        start_time = time.time()
        for i in range(1000):
            await cache.set(f"key{i}", f"value{i}")
        set_time = time.time() - start_time
        
        # Measure get operations
        start_time = time.time()
        for i in range(1000):
            await cache.get(f"key{i}")
        get_time = time.time() - start_time
        
        # Performance assertions (should be fast)
        assert set_time < 1.0  # Should set 1000 items in less than 1 second
        assert get_time < 0.5  # Should get 1000 items in less than 0.5 seconds
        
        # Verify stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 1000
        assert stats["total_accesses"] == 1000


class TestSpecializedCaches:
    """Test specialized cache implementations"""
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self):
        """Test embedding cache functionality"""
        cache = EmbeddingCache()
        
        text = "This is a test sentence"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Set and get embedding
        await cache.set_embedding(text, embedding)
        result = await cache.get_embedding(text)
        
        assert result == embedding
        
        # Different text should not match
        result = await cache.get_embedding("Different text")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_document_cache(self):
        """Test document cache functionality"""
        cache = DocumentCache()
        
        url = "https://example.com/document.pdf"
        document = {"content": "Document content", "metadata": {"pages": 10}}
        
        # Set and get document
        await cache.set_document(url, document)
        result = await cache.get_document(url)
        
        assert result == document
    
    @pytest.mark.asyncio
    async def test_query_cache(self):
        """Test query cache functionality"""
        cache = QueryCache()
        
        query = "What is the grace period?"
        document_id = "doc123"
        answer = "The grace period is 30 days"
        
        # Set and get query result
        await cache.set_query_result(query, document_id, answer)
        result = await cache.get_query_result(query, document_id)
        
        assert result == answer
        
        # Different document should not match
        result = await cache.get_query_result(query, "doc456")
        assert result is None


class TestAsyncUtils:
    """Test async utility functions"""
    
    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry with successful operation"""
        retry = AsyncRetry(max_attempts=3, base_delay=0.01)
        
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await retry.execute(flaky_operation)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_timeout_after(self):
        """Test timeout functionality"""
        async def quick_operation():
            await asyncio.sleep(0.01)
            return "done"
        
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "done"
        
        # Quick operation should succeed
        result = await timeout_after(0.1, quick_operation())
        assert result == "done"
        
        # Slow operation should timeout
        with pytest.raises(Exception):  # AsyncTimeout
            await timeout_after(0.05, slow_operation())
    
    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """Test concurrent execution with limits"""
        async def test_coro(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        coroutines = [test_coro(i) for i in range(10)]
        
        start_time = time.time()
        results = await gather_with_concurrency(3, *coroutines)
        duration = time.time() - start_time
        
        assert results == [i * 2 for i in range(10)]
        # With concurrency limit of 3, should take longer than unlimited concurrency
        assert duration > 0.03  # At least 3 batches of 0.01s each
    
    @pytest.mark.asyncio
    async def test_async_map(self):
        """Test async map functionality"""
        async def double_value(x):
            await asyncio.sleep(0.001)
            return x * 2
        
        items = list(range(100))
        
        start_time = time.time()
        results = await async_map(double_value, items, concurrency=10)
        duration = time.time() - start_time
        
        assert results == [x * 2 for x in items]
        # Should be faster than sequential execution
        assert duration < 1.0  # Should complete in less than 1 second


class TestAsyncBatch:
    """Test async batch processing"""
    
    class TestBatch(AsyncBatch):
        """Test implementation of AsyncBatch"""
        
        async def _process_batch(self, items):
            # Simulate batch processing
            await asyncio.sleep(0.01)
            return [item * 2 for item in items]
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality"""
        batch_processor = self.TestBatch(batch_size=5, max_wait_time=0.1)
        
        # Submit items concurrently
        tasks = [batch_processor.add_item(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert results == [i * 2 for i in range(10)]


class TestAsyncPool:
    """Test async worker pool"""
    
    @pytest.mark.asyncio
    async def test_pool_processing(self):
        """Test async pool processing"""
        pool = AsyncPool(worker_count=3, queue_size=10)
        
        await pool.start()
        
        try:
            async def test_task(value):
                await asyncio.sleep(0.01)
                return value * 2
            
            # Submit multiple tasks
            tasks = [pool.submit(test_task, i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            assert results == [i * 2 for i in range(10)]
            
        finally:
            await pool.stop()


class TestConnectionPool:
    """Test connection pooling"""
    
    def setup_method(self):
        self.created_connections = []
        self.closed_connections = []
    
    async def create_connection(self):
        """Mock connection creation"""
        conn = Mock()
        conn.id = len(self.created_connections)
        self.created_connections.append(conn)
        await asyncio.sleep(0.001)  # Simulate connection time
        return conn
    
    async def close_connection(self, conn):
        """Mock connection closing"""
        self.closed_connections.append(conn)
        await asyncio.sleep(0.001)  # Simulate close time
    
    async def health_check(self, conn):
        """Mock health check"""
        return True
    
    @pytest.mark.asyncio
    async def test_connection_pool_basic(self):
        """Test basic connection pool operations"""
        pool = ConnectionPool(
            name="test_pool",
            create_connection=self.create_connection,
            close_connection=self.close_connection,
            health_check=self.health_check,
            min_size=2,
            max_size=5
        )
        
        await pool.initialize()
        
        try:
            # Should have created minimum connections
            assert len(self.created_connections) == 2
            
            # Get connection
            conn1 = await pool.get_connection()
            assert conn1 is not None
            
            # Return connection
            await pool.return_connection(conn1)
            
            # Get connection again (should reuse)
            conn2 = await pool.get_connection()
            assert conn2 == conn1  # Should be the same connection
            
            await pool.return_connection(conn2)
            
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_connection_pool_context_manager(self):
        """Test connection pool context manager"""
        pool = ConnectionPool(
            name="test_pool",
            create_connection=self.create_connection,
            close_connection=self.close_connection,
            min_size=1,
            max_size=3
        )
        
        await pool.initialize()
        
        try:
            async with pool.connection() as conn:
                assert conn is not None
                # Connection should be automatically returned after context
            
            # Verify connection was returned to pool
            stats = pool.get_stats()
            assert stats["pool_size"] >= 1
            
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_connection_pool_concurrency(self):
        """Test connection pool under concurrent load"""
        pool = ConnectionPool(
            name="test_pool",
            create_connection=self.create_connection,
            close_connection=self.close_connection,
            min_size=2,
            max_size=5
        )
        
        await pool.initialize()
        
        try:
            async def use_connection():
                async with pool.connection() as conn:
                    await asyncio.sleep(0.01)
                    return conn.id
            
            # Run multiple concurrent operations
            tasks = [use_connection() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Should have completed all tasks
            assert len(results) == 10
            
            # Should not have created more than max_size connections
            assert len(self.created_connections) <= 5
            
        finally:
            await pool.close_all()


class TestPerformanceIntegration:
    """Integration tests for performance features"""
    
    @pytest.mark.asyncio
    async def test_cache_manager_performance(self):
        """Test cache manager performance"""
        cache_manager = CacheManager()
        
        # Test embedding cache performance
        start_time = time.time()
        for i in range(100):
            text = f"Test sentence {i}"
            embedding = [float(j) for j in range(10)]
            await cache_manager.embedding_cache.set_embedding(text, embedding)
        
        for i in range(100):
            text = f"Test sentence {i}"
            result = await cache_manager.embedding_cache.get_embedding(text)
            assert result is not None
        
        duration = time.time() - start_time
        assert duration < 1.0  # Should complete in less than 1 second
        
        # Check stats
        stats = await cache_manager.get_all_stats()
        assert stats["embedding_cache"]["total_entries"] == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self):
        """Test concurrent cache operations"""
        cache = LRUCache(max_size=1000)
        
        async def cache_operations(start_idx):
            for i in range(start_idx, start_idx + 100):
                await cache.set(f"key{i}", f"value{i}")
                result = await cache.get(f"key{i}")
                assert result == f"value{i}"
        
        # Run concurrent cache operations
        start_time = time.time()
        tasks = [cache_operations(i * 100) for i in range(5)]
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Should complete efficiently
        assert duration < 2.0
        
        # Verify all data is cached
        stats = cache.get_stats()
        assert stats["total_entries"] == 500


@pytest.mark.asyncio
async def test_memory_usage_optimization():
    """Test memory usage optimization"""
    cache = LRUCache(max_size=1000)
    
    # Fill cache with large objects
    large_data = "x" * 1000  # 1KB string
    
    for i in range(1000):
        await cache.set(f"key{i}", large_data)
    
    # Cache should not exceed max size
    stats = cache.get_stats()
    assert stats["total_entries"] <= 1000
    
    # Add more items to trigger eviction
    for i in range(1000, 1500):
        await cache.set(f"key{i}", large_data)
    
    # Should still be at max size
    stats = cache.get_stats()
    assert stats["total_entries"] <= 1000


@pytest.mark.asyncio
async def test_response_time_optimization():
    """Test response time optimization features"""
    # Test concurrent processing
    async def simulate_processing(delay):
        await asyncio.sleep(delay)
        return f"processed_{delay}"
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for delay in [0.01, 0.01, 0.01, 0.01, 0.01]:
        result = await simulate_processing(delay)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Concurrent processing
    start_time = time.time()
    tasks = [simulate_processing(0.01) for _ in range(5)]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    # Concurrent should be significantly faster
    assert concurrent_time < sequential_time
    assert concurrent_time < 0.02  # Should complete in ~0.01s instead of ~0.05s
    
    # Results should be the same
    assert len(concurrent_results) == len(sequential_results)