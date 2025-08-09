"""
Caching utilities for performance optimization
"""
import asyncio
import time
import hashlib
import pickle
from typing import Any, Optional, Dict, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict

from app.core.logging import LoggerMixin


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def access(self) -> Any:
        """Access the cached value and update metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        return self.value


class LRUCache(LoggerMixin):
    """LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            return entry.access()
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            ttl = ttl_seconds or self.default_ttl_seconds
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
            
            entry = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict oldest if over max size
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            "total_entries": total_entries,
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "hit_rate": 0.0 if total_accesses == 0 else total_accesses / max(1, total_entries),
            "oldest_entry": min(
                (entry.created_at for entry in self.cache.values()),
                default=None
            ),
            "newest_entry": max(
                (entry.created_at for entry in self.cache.values()),
                default=None
            )
        }


class EmbeddingCache(LRUCache):
    """Specialized cache for embeddings"""
    
    def __init__(self, max_size: int = 10000, default_ttl_seconds: int = 86400):  # 24 hours
        super().__init__(max_size, default_ttl_seconds)
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    async def get_embedding(self, text: str) -> Optional[list]:
        """Get embedding for text"""
        key = f"embedding:{self._hash_text(text)}"
        return await self.get(key)
    
    async def set_embedding(self, text: str, embedding: list, ttl_seconds: Optional[int] = None) -> None:
        """Set embedding for text"""
        key = f"embedding:{self._hash_text(text)}"
        await self.set(key, embedding, ttl_seconds)


class DocumentCache(LRUCache):
    """Specialized cache for processed documents"""
    
    def __init__(self, max_size: int = 100, default_ttl_seconds: int = 3600):  # 1 hour
        super().__init__(max_size, default_ttl_seconds)
    
    def _hash_url(self, url: str) -> str:
        """Create hash key for document URL"""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
    
    async def get_document(self, url: str) -> Optional[Any]:
        """Get processed document by URL"""
        key = f"document:{self._hash_url(url)}"
        return await self.get(key)
    
    async def set_document(self, url: str, document: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set processed document"""
        key = f"document:{self._hash_url(url)}"
        await self.set(key, document, ttl_seconds)


class QueryCache(LRUCache):
    """Specialized cache for query results"""
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 1800):  # 30 minutes
        super().__init__(max_size, default_ttl_seconds)
    
    def _hash_query(self, query: str, document_id: str) -> str:
        """Create hash key for query and document combination"""
        combined = f"{query}:{document_id}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    async def get_query_result(self, query: str, document_id: str) -> Optional[Any]:
        """Get cached query result"""
        key = f"query:{self._hash_query(query, document_id)}"
        return await self.get(key)
    
    async def set_query_result(self, query: str, document_id: str, result: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set cached query result"""
        key = f"query:{self._hash_query(query, document_id)}"
        await self.set(key, result, ttl_seconds)


def cache_result(cache: LRUCache, key_func: Callable = None, ttl_seconds: int = 3600):
    """Decorator to cache function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:16]
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async cache operations
            # This is a simplified version that doesn't actually cache
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class CacheManager(LoggerMixin):
    """Central cache management"""
    
    def __init__(self):
        self.embedding_cache = EmbeddingCache()
        self.document_cache = DocumentCache()
        self.query_cache = QueryCache()
        self.general_cache = LRUCache()
        
        # Start cleanup task only if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, will start cleanup later
            pass
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "document_cache": self.document_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "general_cache": self.general_cache.get_stats()
        }
    
    async def clear_all_caches(self) -> None:
        """Clear all caches"""
        await asyncio.gather(
            self.embedding_cache.clear(),
            self.document_cache.clear(),
            self.query_cache.clear(),
            self.general_cache.clear()
        )
        self.logger.info("All caches cleared")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired cache entries"""
        while True:
            try:
                # Clean up expired entries in all caches
                results = await asyncio.gather(
                    self.embedding_cache.cleanup_expired(),
                    self.document_cache.cleanup_expired(),
                    self.query_cache.cleanup_expired(),
                    self.general_cache.cleanup_expired(),
                    return_exceptions=True
                )
                
                total_cleaned = sum(r for r in results if isinstance(r, int))
                if total_cleaned > 0:
                    self.logger.info(f"Cleaned up {total_cleaned} expired cache entries")
                
            except Exception as e:
                self.logger.error(f"Error during cache cleanup: {str(e)}")
            
            # Sleep for 5 minutes
            await asyncio.sleep(300)


# Global cache manager instance
cache_manager = CacheManager()