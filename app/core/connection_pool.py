"""
Connection pooling for external services
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections import deque

from app.core.logging import LoggerMixin


@dataclass
class PooledConnection:
    """Represents a pooled connection"""
    connection: Any
    created_at: float
    last_used: float
    use_count: int = 0
    is_healthy: bool = True
    
    def mark_used(self):
        """Mark connection as used"""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_expired(self, max_age_seconds: int) -> bool:
        """Check if connection is expired"""
        return time.time() - self.created_at > max_age_seconds
    
    def is_idle_too_long(self, max_idle_seconds: int) -> bool:
        """Check if connection has been idle too long"""
        return time.time() - self.last_used > max_idle_seconds


class ConnectionPool(LoggerMixin):
    """Generic connection pool implementation"""
    
    def __init__(
        self,
        name: str,
        create_connection: Callable,
        close_connection: Callable,
        health_check: Optional[Callable] = None,
        min_size: int = 2,
        max_size: int = 10,
        max_age_seconds: int = 3600,
        max_idle_seconds: int = 300,
        health_check_interval: int = 60
    ):
        self.name = name
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.health_check = health_check
        self.min_size = min_size
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.max_idle_seconds = max_idle_seconds
        self.health_check_interval = health_check_interval
        
        self.pool: deque[PooledConnection] = deque()
        self.active_connections: Dict[int, PooledConnection] = {}
        self.lock = asyncio.Lock()
        self.stats = {
            "created": 0,
            "destroyed": 0,
            "borrowed": 0,
            "returned": 0,
            "health_checks": 0,
            "health_failures": 0
        }
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def initialize(self):
        """Initialize pool with minimum connections"""
        async with self.lock:
            for _ in range(self.min_size):
                try:
                    conn = await self._create_pooled_connection()
                    self.pool.append(conn)
                except Exception as e:
                    self.logger.error(f"Failed to create initial connection for pool {self.name}: {str(e)}")
        
        self.logger.info(f"Initialized connection pool {self.name} with {len(self.pool)} connections")
    
    async def get_connection(self, timeout: float = 30.0) -> Any:
        """Get connection from pool"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self.lock:
                # Try to get healthy connection from pool
                while self.pool:
                    pooled_conn = self.pool.popleft()
                    
                    # Check if connection is still valid
                    if not pooled_conn.is_expired(self.max_age_seconds) and pooled_conn.is_healthy:
                        pooled_conn.mark_used()
                        self.active_connections[id(pooled_conn.connection)] = pooled_conn
                        self.stats["borrowed"] += 1
                        return pooled_conn.connection
                    else:
                        # Connection is expired or unhealthy, close it
                        await self._close_pooled_connection(pooled_conn)
                
                # No available connections, try to create new one
                if len(self.active_connections) + len(self.pool) < self.max_size:
                    try:
                        pooled_conn = await self._create_pooled_connection()
                        pooled_conn.mark_used()
                        self.active_connections[id(pooled_conn.connection)] = pooled_conn
                        self.stats["borrowed"] += 1
                        return pooled_conn.connection
                    except Exception as e:
                        self.logger.error(f"Failed to create connection for pool {self.name}: {str(e)}")
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Failed to get connection from pool {self.name} within {timeout} seconds")
    
    async def return_connection(self, connection: Any, is_healthy: bool = True):
        """Return connection to pool"""
        async with self.lock:
            conn_id = id(connection)
            
            if conn_id not in self.active_connections:
                self.logger.warning(f"Attempted to return unknown connection to pool {self.name}")
                return
            
            pooled_conn = self.active_connections.pop(conn_id)
            pooled_conn.is_healthy = is_healthy
            self.stats["returned"] += 1
            
            if is_healthy and not pooled_conn.is_expired(self.max_age_seconds):
                # Return healthy connection to pool
                self.pool.append(pooled_conn)
            else:
                # Close unhealthy or expired connection
                await self._close_pooled_connection(pooled_conn)
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for getting and returning connections"""
        conn = None
        try:
            conn = await self.get_connection()
            yield conn
        except Exception as e:
            if conn:
                await self.return_connection(conn, is_healthy=False)
            raise
        else:
            if conn:
                await self.return_connection(conn, is_healthy=True)
    
    async def close_all(self):
        """Close all connections and stop maintenance"""
        self._maintenance_task.cancel()
        
        async with self.lock:
            # Close all pooled connections
            while self.pool:
                pooled_conn = self.pool.popleft()
                await self._close_pooled_connection(pooled_conn)
            
            # Close all active connections
            for pooled_conn in list(self.active_connections.values()):
                await self._close_pooled_connection(pooled_conn)
            
            self.active_connections.clear()
        
        self.logger.info(f"Closed all connections for pool {self.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "name": self.name,
            "pool_size": len(self.pool),
            "active_connections": len(self.active_connections),
            "total_connections": len(self.pool) + len(self.active_connections),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "stats": self.stats.copy()
        }
    
    async def _create_pooled_connection(self) -> PooledConnection:
        """Create a new pooled connection"""
        if asyncio.iscoroutinefunction(self.create_connection):
            connection = await self.create_connection()
        else:
            connection = self.create_connection()
        
        pooled_conn = PooledConnection(
            connection=connection,
            created_at=time.time(),
            last_used=time.time()
        )
        
        self.stats["created"] += 1
        return pooled_conn
    
    async def _close_pooled_connection(self, pooled_conn: PooledConnection):
        """Close a pooled connection"""
        try:
            if asyncio.iscoroutinefunction(self.close_connection):
                await self.close_connection(pooled_conn.connection)
            else:
                self.close_connection(pooled_conn.connection)
        except Exception as e:
            self.logger.error(f"Error closing connection in pool {self.name}: {str(e)}")
        
        self.stats["destroyed"] += 1
    
    async def _maintenance_loop(self):
        """Periodic maintenance of connection pool"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in maintenance loop for pool {self.name}: {str(e)}")
    
    async def _perform_maintenance(self):
        """Perform pool maintenance"""
        async with self.lock:
            # Remove expired and idle connections
            healthy_connections = deque()
            
            while self.pool:
                pooled_conn = self.pool.popleft()
                
                if (pooled_conn.is_expired(self.max_age_seconds) or 
                    pooled_conn.is_idle_too_long(self.max_idle_seconds)):
                    await self._close_pooled_connection(pooled_conn)
                elif self.health_check:
                    # Perform health check
                    try:
                        self.stats["health_checks"] += 1
                        if asyncio.iscoroutinefunction(self.health_check):
                            is_healthy = await self.health_check(pooled_conn.connection)
                        else:
                            is_healthy = self.health_check(pooled_conn.connection)
                        
                        if is_healthy:
                            healthy_connections.append(pooled_conn)
                        else:
                            self.stats["health_failures"] += 1
                            await self._close_pooled_connection(pooled_conn)
                    except Exception as e:
                        self.stats["health_failures"] += 1
                        self.logger.warning(f"Health check failed for connection in pool {self.name}: {str(e)}")
                        await self._close_pooled_connection(pooled_conn)
                else:
                    healthy_connections.append(pooled_conn)
            
            self.pool = healthy_connections
            
            # Ensure minimum pool size
            while len(self.pool) < self.min_size:
                try:
                    conn = await self._create_pooled_connection()
                    self.pool.append(conn)
                except Exception as e:
                    self.logger.error(f"Failed to maintain minimum pool size for {self.name}: {str(e)}")
                    break


class ConnectionPoolManager(LoggerMixin):
    """Manager for multiple connection pools"""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
    
    def create_pool(
        self,
        name: str,
        create_connection: Callable,
        close_connection: Callable,
        **kwargs
    ) -> ConnectionPool:
        """Create a new connection pool"""
        if name in self.pools:
            raise ValueError(f"Pool {name} already exists")
        
        pool = ConnectionPool(
            name=name,
            create_connection=create_connection,
            close_connection=close_connection,
            **kwargs
        )
        
        self.pools[name] = pool
        return pool
    
    async def initialize_all(self):
        """Initialize all pools"""
        for pool in self.pools.values():
            await pool.initialize()
    
    async def close_all(self):
        """Close all pools"""
        for pool in self.pools.values():
            await pool.close_all()
        
        self.pools.clear()
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get pool by name"""
        return self.pools.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        return {
            name: pool.get_stats()
            for name, pool in self.pools.items()
        }


# Global connection pool manager
connection_pool_manager = ConnectionPoolManager()