"""
Async utilities and helpers
"""
import asyncio
import time
from typing import Any, Callable, Optional, List, Dict, Coroutine
from functools import wraps
from contextlib import asynccontextmanager

from app.core.logging import LoggerMixin


class AsyncTimeout(Exception):
    """Exception raised when async operation times out"""
    pass


class AsyncRetry(LoggerMixin):
    """Async retry utility with exponential backoff"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions
    
    async def execute(self, coro_func: Callable, *args, **kwargs) -> Any:
        """Execute coroutine with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(coro_func):
                    result = await coro_func(*args, **kwargs)
                else:
                    result = coro_func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except self.exceptions as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    self.logger.error(f"Operation failed after {self.max_attempts} attempts: {str(e)}")
                    break
                
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator version of retry"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        
        return wrapper


async def timeout_after(seconds: float, coro: Coroutine) -> Any:
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise AsyncTimeout(f"Operation timed out after {seconds} seconds")


async def gather_with_concurrency(limit: int, *coroutines) -> List[Any]:
    """Execute coroutines with concurrency limit"""
    semaphore = asyncio.Semaphore(limit)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[sem_coro(coro) for coro in coroutines])


async def gather_with_timeout(timeout_seconds: float, *coroutines) -> List[Any]:
    """Execute coroutines with overall timeout"""
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coroutines),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise AsyncTimeout(f"Batch operation timed out after {timeout_seconds} seconds")


class AsyncBatch(LoggerMixin):
    """Batch processor for async operations"""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self._lock = asyncio.Lock()
        self._batch_task = None
    
    async def add_item(self, item: Any) -> Any:
        """Add item to batch and return future result"""
        async with self._lock:
            future = asyncio.Future()
            self.pending_items.append(item)
            self.pending_futures.append(future)
            
            # Start batch processing if needed
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batches())
            
            return await future
    
    async def _process_batches(self):
        """Process pending items in batches"""
        while True:
            async with self._lock:
                if not self.pending_items:
                    break
                
                # Check if we should process a batch
                should_process = (
                    len(self.pending_items) >= self.batch_size or
                    time.time() - self.last_batch_time >= self.max_wait_time
                )
                
                if not should_process:
                    # Wait a bit more
                    await asyncio.sleep(0.1)
                    continue
                
                # Extract batch
                batch_items = self.pending_items[:self.batch_size]
                batch_futures = self.pending_futures[:self.batch_size]
                
                self.pending_items = self.pending_items[self.batch_size:]
                self.pending_futures = self.pending_futures[self.batch_size:]
                self.last_batch_time = time.time()
            
            # Process batch (override in subclass)
            try:
                results = await self._process_batch(batch_items)
                
                # Set results
                for future, result in zip(batch_futures, results):
                    if not future.done():
                        future.set_result(result)
                        
            except Exception as e:
                # Set exception for all futures
                for future in batch_futures:
                    if not future.done():
                        future.set_exception(e)
    
    async def _process_batch(self, items: List[Any]) -> List[Any]:
        """Override this method to implement batch processing logic"""
        raise NotImplementedError("Subclasses must implement _process_batch")


class AsyncPool(LoggerMixin):
    """Async worker pool for processing tasks"""
    
    def __init__(self, worker_count: int = 5, queue_size: int = 100):
        self.worker_count = worker_count
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.workers = []
        self.running = False
    
    async def start(self):
        """Start worker pool"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.worker_count)
        ]
        
        self.logger.info(f"Started async pool with {self.worker_count} workers")
    
    async def stop(self):
        """Stop worker pool"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.logger.info("Stopped async pool")
    
    async def submit(self, coro_func: Callable, *args, **kwargs) -> Any:
        """Submit task to pool and return future"""
        if not self.running:
            raise RuntimeError("Pool is not running")
        
        future = asyncio.Future()
        task = (coro_func, args, kwargs, future)
        
        await self.queue.put(task)
        return await future
    
    async def _worker(self, worker_id: int):
        """Worker coroutine"""
        self.logger.debug(f"Worker {worker_id} started")
        
        try:
            while self.running:
                try:
                    # Get task from queue
                    task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    coro_func, args, kwargs, future = task
                    
                    try:
                        # Execute task
                        if asyncio.iscoroutinefunction(coro_func):
                            result = await coro_func(*args, **kwargs)
                        else:
                            result = coro_func(*args, **kwargs)
                        
                        # Set result
                        if not future.done():
                            future.set_result(result)
                            
                    except Exception as e:
                        # Set exception
                        if not future.done():
                            future.set_exception(e)
                    
                    finally:
                        self.queue.task_done()
                
                except asyncio.TimeoutError:
                    # No task available, continue
                    continue
                    
        except asyncio.CancelledError:
            self.logger.debug(f"Worker {worker_id} cancelled")
        except Exception as e:
            self.logger.error(f"Worker {worker_id} error: {str(e)}")
        
        self.logger.debug(f"Worker {worker_id} stopped")


@asynccontextmanager
async def async_timer():
    """Context manager to measure async operation time"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Operation took {duration:.3f} seconds")


def run_in_thread_pool(func: Callable, *args, **kwargs):
    """Run blocking function in thread pool"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)


async def async_map(func: Callable, items: List[Any], concurrency: int = 10) -> List[Any]:
    """Apply async function to list of items with concurrency limit"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def sem_func(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return func(item)
    
    return await asyncio.gather(*[sem_func(item) for item in items])


class AsyncLock:
    """Enhanced async lock with timeout and context info"""
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._lock = asyncio.Lock()
        self._acquired_at = None
        self._acquired_by = None
    
    async def acquire(self, timeout: Optional[float] = None):
        """Acquire lock with optional timeout"""
        if timeout:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                raise AsyncTimeout(f"Failed to acquire lock '{self.name}' within {timeout} seconds")
        else:
            await self._lock.acquire()
        
        self._acquired_at = time.time()
        self._acquired_by = asyncio.current_task()
    
    def release(self):
        """Release lock"""
        self._lock.release()
        self._acquired_at = None
        self._acquired_by = None
    
    def locked(self) -> bool:
        """Check if lock is currently held"""
        return self._lock.locked()
    
    def get_lock_info(self) -> Dict[str, Any]:
        """Get information about lock state"""
        return {
            "name": self.name,
            "locked": self.locked(),
            "acquired_at": self._acquired_at,
            "held_for_seconds": time.time() - self._acquired_at if self._acquired_at else None,
            "acquired_by": str(self._acquired_by) if self._acquired_by else None
        }
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()