"""
Abstract base interface for the DSA Memory subsystem.

This module defines the enterprise repository pattern interface that all 
memory backends (Semantic, Episodic, Procedural) must implement. It enforces 
strict contracts for connectivity, CRUD operations, retrieval, lifecycle 
management, and observability.

No concrete implementations are provided here. This is purely an interface.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, ParamSpec

from pydantic import BaseModel

from src.memory.schemas import (
    MemoryRecord,
    MemoryStatistics,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class MemoryBaseError(Exception):
    """Base exception for all memory subsystem errors."""
    pass


class MemoryConnectionError(MemoryBaseError):
    """Raised when a memory backend fails to connect or disconnect."""
    pass


class MemoryStorageError(MemoryBaseError):
    """Raised when a memory backend fails to store or update records."""
    pass


class MemoryRetrievalError(MemoryBaseError):
    """Raised when a memory backend fails to retrieve or search records."""
    pass


class MemoryNotFoundError(MemoryBaseError):
    """Raised when a requested memory record does not exist."""
    pass


class MemoryBackupError(MemoryBaseError):
    """Raised when a memory backend fails to backup or restore data."""
    pass


# ---------------------------------------------------------------------------
# Decorators for Interface Compliance
# ---------------------------------------------------------------------------

P = ParamSpec("P")
T = TypeVar("T")


def timed(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that logs the execution time of a memory operation.
    
    Implementations must wrap all I/O heavy methods with this decorator 
    to maintain observability standards.
    """
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            logger.debug(f"{func.__name__} executed in {duration:.4f}s")
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable:
    """
    Decorator that provides exponential backoff retry logic for transient failures.
    
    Implementations must wrap all network-bound methods with this decorator 
    to ensure resilience against temporary database or network outages.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (MemoryConnectionError, MemoryRetrievalError) as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseMemory(ABC):
    """
    Abstract base class defining the contract for all DSA memory backends.
    
    This interface enforces the Repository Pattern, ensuring that all 
    concrete implementations (e.g., ChromaDB, SQLite, Postgres) provide 
    a consistent API for the Memory Manager.
    
    Thread Safety:
        Implementations must ensure thread safety for all state-mutating 
        operations. Use `asyncio.Lock` for async contexts or threading 
        primitives for synchronous contexts.
        
    Observability:
        All methods must be wrapped with `@timed` and `@retry` decorators 
        (where applicable) to ensure consistent logging and resilience.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes a connection to the underlying memory backend.
        
        Raises:
            MemoryConnectionError: If the connection cannot be established 
                within the configured timeout.
                
        Logging:
            Logs connection attempts and success/failure states.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully closes the connection to the memory backend.
        
        Raises:
            MemoryConnectionError: If the connection cannot be closed cleanly.
            
        Logging:
            Logs disconnection events.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Performs a liveness and readiness check on the memory backend.
        
        Returns:
            A dictionary containing health metrics (e.g., latency, status, version).
            
        Raises:
            MemoryConnectionError: If the backend is unreachable or unhealthy.
            
        Logging:
            Logs health check results and latency.
        """
        ...

    @abstractmethod
    async def store(self, record: MemoryRecord) -> str:
        """
        Persists a single memory record to the backend.
        
        Args:
            record: The memory record to store.
            
        Returns:
            The unique identifier of the stored record.
            
        Raises:
            MemoryStorageError: If the record fails to persist.
            
        Logging:
            Logs storage operations and record IDs.
        """
        ...

    @abstractmethod
    async def store_batch(self, records: list[MemoryRecord]) -> list[str]:
        """
        Persists multiple memory records in a single atomic transaction.
        
        Args:
            records: A list of memory records to store.
            
        Returns:
            A list of unique identifiers for the stored records.
            
        Raises:
            MemoryStorageError: If the batch operation fails.
            
        Logging:
            Logs batch storage operations and total record count.
        """
        ...

    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: dict[str, Any] | None = None
    ) -> RetrievalResult:
        """
        Retrieves memory records matching a semantic or structured query.
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters to narrow the search space.
            
        Returns:
            A RetrievalResult containing scored candidates.
            
        Raises:
            MemoryRetrievalError: If the retrieval operation fails.
            
        Logging:
            Logs query parameters and retrieval latency.
        """
        ...

    @abstractmethod
    async def retrieve_by_id(self, record_id: str) -> MemoryRecord | None:
        """
        Retrieves a specific memory record by its unique identifier.
        
        Args:
            record_id: The unique identifier of the record.
            
        Returns:
            The MemoryRecord if found, otherwise None.
            
        Raises:
            MemoryRetrievalError: If the lookup operation fails.
            
        Logging:
            Logs retrieval attempts by ID.
        """
        ...

    @abstractmethod
    async def update(self, record: MemoryRecord) -> bool:
        """
        Updates an existing memory record in the backend.
        
        Args:
            record: The updated memory record.
            
        Returns:
            True if the update was successful, False otherwise.
            
        Raises:
            MemoryStorageError: If the update operation fails.
            MemoryNotFoundError: If the record ID does not exist.
            
        Logging:
            Logs update operations and version increments.
        """
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """
        Deletes a memory record from the backend.
        
        Args:
            record_id: The unique identifier of the record to delete.
            
        Returns:
            True if the deletion was successful, False otherwise.
            
        Raises:
            MemoryStorageError: If the deletion operation fails.
            
        Logging:
            Logs deletion operations.
        """
        ...

    @abstractmethod
    async def clear(self) -> int:
        """
        Deletes all records from the memory backend.
        
        Returns:
            The total number of records deleted.
            
        Raises:
            MemoryStorageError: If the clear operation fails.
            
        Logging:
            Logs clear operations and total records purged.
        """
        ...

    @abstractmethod
    async def search(
        self, 
        query: str, 
        filters: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        """
        Performs a direct search for memory records without scoring.
        
        Args:
            query: The search query string.
            filters: Optional metadata filters.
            
        Returns:
            A list of matching MemoryRecord objects.
            
        Raises:
            MemoryRetrievalError: If the search operation fails.
            
        Logging:
            Logs search operations.
        """
        ...

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """
        Returns the total number of records in the backend.
        
        Args:
            filters: Optional metadata filters to count a subset of records.
            
        Returns:
            The total count of records.
            
        Raises:
            MemoryRetrievalError: If the count operation fails.
            
        Logging:
            Logs count operations.
        """
        ...

    @abstractmethod
    async def statistics(self) -> MemoryStatistics:
        """
        Computes and returns aggregate statistics about the memory backend.
        
        Returns:
            A MemoryStatistics object containing counts, sizes, and averages.
            
        Raises:
            MemoryRetrievalError: If statistics cannot be computed.
            
        Logging:
            Logs statistics generation.
        """
        ...

    @abstractmethod
    async def backup(self, destination: str) -> str:
        """
        Creates a backup of the memory backend to a specified destination.
        
        Args:
            destination: File path or URI for the backup destination.
            
        Returns:
            The path or URI of the created backup.
            
        Raises:
            MemoryBackupError: If the backup operation fails.
            
        Logging:
            Logs backup operations and destination paths.
        """
        ...

    @abstractmethod
    async def restore(self, source: str) -> bool:
        """
        Restores the memory backend from a specified backup source.
        
        Args:
            source: File path or URI of the backup source.
            
        Returns:
            True if the restore was successful, False otherwise.
            
        Raises:
            MemoryBackupError: If the restore operation fails.
            
        Logging:
            Logs restore operations and source paths.
        """
        ...

    @abstractmethod
    async def optimize(self) -> dict[str, Any]:
        """
        Performs backend-specific optimization (e.g., index rebuilding, VACUUM).
        
        Returns:
            A dictionary containing optimization metrics and status.
            
        Raises:
            MemoryStorageError: If the optimization operation fails.
            
        Logging:
            Logs optimization operations and duration.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Finalizes the backend instance and releases all resources.
        
        This method should be called when the memory backend is no longer 
        needed, ensuring proper garbage collection and resource cleanup.
        
        Raises:
            MemoryConnectionError: If resources cannot be released cleanly.
            
        Logging:
            Logs resource cleanup.
        """
        ...
