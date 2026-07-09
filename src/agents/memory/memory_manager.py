"""
Memory Manager for the Data Samanvayah Agent (DSA).

This module implements the central orchestration layer for the entire memory 
subsystem. It is the single, unified API exposed to all DSA agents (Planner, 
Supervisor, etc.), abstracting away the complexities of the underlying 
semantic, episodic, and procedural memory backends.

Features:
- Centralized orchestration of all memory types.
- Automatic embedding generation for semantic memory.
- Unified, ranked retrieval via the Memory Retrieval Engine.
- Asynchronous caching for high-frequency retrievals.
- Comprehensive lifecycle management (init, shutdown, backup, optimize).
- Detailed observability, logging, and performance metrics.
- Dependency injection and composition-based architecture.
- Singleton access pattern for global state consistency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.memory.base import BaseMemory, MemoryBaseError
from src.memory.embedding_service import EmbeddingConfig, EmbeddingService
from src.memory.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from src.memory.procedural_memory import ProceduralMemory, ProceduralMemoryConfig
from src.memory.retrieval import (
    MemoryRetrievalEngine,
    RetrievalContext,
    ScoringWeights,
    UnifiedRetrievalResult,
)
from src.memory.schemas import (
    EpisodicMemoryRecord,
    MemoryStatistics,
    ProceduralRule,
    SemanticMemoryRecord,
)
from src.memory.semantic_memory import SemanticMemory, SemanticMemoryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Caching
# ---------------------------------------------------------------------------

class MemoryManagerConfig(BaseModel):
    """Configuration for the Memory Manager orchestration layer."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Subsystem Configs
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    semantic: SemanticMemoryConfig = Field(default_factory=SemanticMemoryConfig)
    episodic: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    procedural: ProceduralMemoryConfig = Field(default_factory=ProceduralMemoryConfig)
    
    # Retrieval Engine
    retrieval_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    retrieval_top_k: int = Field(default=10, gt=0)
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=600, gt=0)
    cache_max_size: int = Field(default=500, gt=0)


class AsyncRetrievalCache:
    """Simple thread-safe async LRU cache for retrieval results."""
    
    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._cache: dict[str, tuple[float, UnifiedRetrievalResult]] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash_context(self, context: RetrievalContext) -> str:
        """Creates a deterministic hash for the retrieval context."""
        context_dict = context.model_dump(exclude={"execution_timestamp"})
        serialized = json.dumps(context_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def get(self, context: RetrievalContext) -> Optional[UnifiedRetrievalResult]:
        key = self._hash_context(context)
        async with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self.hits += 1
                    return result
                else:
                    del self._cache[key]
            self.misses += 1
            return None

    async def set(self, context: RetrievalContext, result: UnifiedRetrievalResult) -> None:
        key = self._hash_context(context)
        async with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), result)


# ---------------------------------------------------------------------------
# Memory Manager (The "Brain")
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Central orchestration layer for the DSA Memory Subsystem.
    
    This class coordinates the semantic, episodic, and procedural memory 
    backends. It is the only interface that downstream agents should interact with.
    """
    
    def __init__(self, config: MemoryManagerConfig) -> None:
        self.config = config
        
        # Initialize Services
        self.embedding_service = EmbeddingService(config.embedding)
        
        # Initialize Memory Backends (Composition)
        self.semantic_memory = SemanticMemory(
            config=config.semantic, 
            embedding_service=self.embedding_service
        )
        self.episodic_memory = EpisodicMemory(config=config.episodic)
        self.procedural_memory = ProceduralMemory(config=config.procedural)
        
        # Initialize Retrieval Engine
        self.retrieval_engine = MemoryRetrievalEngine(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            procedural_memory=self.procedural_memory,
            weights=config.retrieval_weights,
            top_k=config.retrieval_top_k
        )
        
        # Initialize Cache
        self.cache = AsyncRetrievalCache(
            max_size=config.cache_max_size, 
            ttl_seconds=config.cache_ttl_seconds
        ) if config.cache_enabled else None
        
        self._is_initialized = False
        logger.info("MemoryManager instantiated.")

    # -----------------------------------------------------------------------
    # Lifecycle Management
    # -----------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Connects to all memory backends and initializes the embedding service.
        Must be called once at application startup.
        """
        if self._is_initialized:
            logger.warning("MemoryManager is already initialized.")
            return
            
        start_time = time.perf_counter()
        logger.info("Initializing Memory Manager subsystems...")
        
        try:
            # Connect backends concurrently
            await asyncio.gather(
                self.semantic_memory.connect(),
                self.episodic_memory.connect(),
                self.procedural_memory.connect(),
            )
            
            # Warm up embedding service (optional, lazy init is also fine)
            await self.embedding_service.get_dimension()
            
            self._is_initialized = True
            duration = time.perf_counter() - start_time
            logger.info(f"Memory Manager initialized successfully in {duration:.2f}s.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise MemoryBaseError(f"Initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Gracefully disconnects from all memory backends and releases resources.
        Must be called at application shutdown.
        """
        if not self._is_initialized:
            return
            
        logger.info("Shutting down Memory Manager subsystems...")
        
        try:
            await asyncio.gather(
                self.semantic_memory.close(),
                self.episodic_memory.close(),
                self.procedural_memory.close(),
            )
            await self.embedding_service.close()
            
            self._is_initialized = False
            logger.info("Memory Manager shut down successfully.")
            
        except Exception as e:
            logger.error(f"Error during Memory Manager shutdown: {e}")

    # -----------------------------------------------------------------------
    # Storage APIs (For Agents to Store Memories)
    # -----------------------------------------------------------------------

    async def store_semantic_memory(self, record: SemanticMemoryRecord) -> str:
        """Stores a semantic memory record, automatically handling embeddings."""
        self._check_initialized()
        logger.info(f"Storing semantic memory for dataset: {record.metadata.dataset_name}")
        return await self.semantic_memory.store(record)

    async def store_episodic_memory(self, record: EpisodicMemoryRecord) -> str:
        """Stores an episodic memory record (execution history)."""
        self._check_initialized()
        logger.info(f"Storing episodic memory for execution: {record.metadata.execution_id}")
        return await self.episodic_memory.store(record)

    async def store_procedural_rule(self, rule: ProceduralRule) -> str:
        """Stores a procedural rule (learned heuristic)."""
        self._check_initialized()
        logger.info(f"Storing procedural rule: {rule.rule_name}")
        return await self.procedural_memory.store(rule)

    # -----------------------------------------------------------------------
    # Retrieval API (The Primary Interface for Agents)
    # -----------------------------------------------------------------------

    async def retrieve_memories(self, context: RetrievalContext) -> UnifiedRetrievalResult:
        """
        The primary retrieval method consumed by the Planner and Supervisor.
        
        Queries all memory backends, merges results, applies the weighted 
        ranking algorithm, and returns a unified, ranked list of memories.
        Includes caching for performance.
        
        Args:
            context: The contextual features of the current execution.
            
        Returns:
            UnifiedRetrievalResult containing ranked memory candidates.
        """
        self._check_initialized()
        
        # Check Cache
        if self.cache:
            cached_result = await self.cache.get(context)
            if cached_result:
                logger.debug(f"Cache hit for memory retrieval. Hits: {self.cache.hits}, Misses: {self.cache.misses}")
                return cached_result
                
        logger.info(f"Executing unified memory retrieval for task: {context.task_type}")
        
        # Execute Retrieval Engine
        result = await self.retrieval_engine.retrieve(context)
        
        # Update Cache
        if self.cache and result.candidates:
            await self.cache.set(context, result)
            
        return result

    # -----------------------------------------------------------------------
    # Update & Delete APIs
    # -----------------------------------------------------------------------

    async def update_memory(self, memory_type: str, record: Any) -> bool:
        """
        Updates an existing memory record in the specified backend.
        
        Args:
            memory_type: One of 'semantic', 'episodic', 'procedural'.
            record: The updated record object.
            
        Returns:
            True if update succeeded.
        """
        self._check_initialized()
        backend = self._get_backend(memory_type)
        return await backend.update(record)

    async def delete_memory(self, memory_type: str, record_id: str) -> bool:
        """
        Deletes a memory record from the specified backend.
        
        Args:
            memory_type: One of 'semantic', 'episodic', 'procedural'.
            record_id: The ID of the record to delete.
            
        Returns:
            True if deletion succeeded.
        """
        self._check_initialized()
        backend = self._get_backend(memory_type)
        return await backend.delete(record_id)

    # -----------------------------------------------------------------------
    # Administrative & Observability APIs
    # -----------------------------------------------------------------------

    async def get_statistics(self) -> dict[str, MemoryStatistics]:
        """
        Retrieves aggregated statistics from all memory backends.
        
        Returns:
            Dictionary mapping backend names to their MemoryStatistics.
        """
        self._check_initialized()
        logger.info("Fetching memory subsystem statistics.")
        
        stats = await asyncio.gather(
            self.semantic_memory.statistics(),
            self.episodic_memory.statistics(),
            self.procedural_memory.statistics(),
        )
        
        return {
            "semantic": stats[0],
            "episodic": stats[1],
            "procedural": stats[2]
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Performs a comprehensive health check across all subsystems.
        
        Returns:
            Dictionary containing health status of each component.
        """
        logger.info("Running Memory Manager health check.")
        
        checks = await asyncio.gather(
            self.semantic_memory.health_check(),
            self.episodic_memory.health_check(),
            self.procedural_memory.health_check(),
            return_exceptions=True
        )
        
        return {
            "initialized": self._is_initialized,
            "cache_enabled": self.cache is not None,
            "cache_stats": self.cache.stats if self.cache else None,
            "semantic_memory": checks[0] if not isinstance(checks[0], Exception) else {"error": str(checks[0])},
            "episodic_memory": checks[1] if not isinstance(checks[1], Exception) else {"error": str(checks[1])},
            "procedural_memory": checks[2] if not isinstance(checks[2], Exception) else {"error": str(checks[2])},
        }

    async def backup(self, destination: str) -> dict[str, str]:
        """
        Creates backups of all memory backends to the specified destination.
        
        Args:
            destination: Base directory for the backups.
            
        Returns:
            Dictionary mapping backend names to their backup file paths.
        """
        self._check_initialized()
        logger.info(f"Initiating backup of all memory backends to {destination}")
        
        results = {}
        try:
            results["semantic"] = await self.semantic_memory.backup(f"{destination}/semantic")
            results["episodic"] = await self.episodic_memory.backup(f"{destination}/episodic.db")
            results["procedural"] = await self.procedural_memory.backup(f"{destination}/procedural.json")
            logger.info("Backup completed successfully.")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
            
        return results

    async def optimize(self) -> dict[str, Any]:
        """
        Triggers optimization routines (e.g., index rebuilding, VACUUM, pruning) 
        across all memory backends.
        
        Returns:
            Dictionary containing optimization results for each backend.
        """
        self._check_initialized()
        logger.info("Running optimization routines for all memory backends.")
        
        results = await asyncio.gather(
            self.semantic_memory.optimize(),
            self.episodic_memory.optimize(),
            self.procedural_memory.optimize(),
        )
        
        return {
            "semantic": results[0],
            "episodic": results[1],
            "procedural": results[2]
        }

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _check_initialized(self) -> None:
        """Ensures the Memory Manager has been initialized before use."""
        if not self._is_initialized:
            raise MemoryBaseError("MemoryManager is not initialized. Call initialize() first.")

    def _get_backend(self, memory_type: str) -> BaseMemory:
        """Returns the correct memory backend based on the type string."""
        backends = {
            "semantic": self.semantic_memory,
            "episodic": self.episodic_memory,
            "procedural": self.procedural_memory,
        }
        if memory_type not in backends:
            raise ValueError(f"Unknown memory type: {memory_type}. Must be one of {list(backends.keys())}")
        return backends[memory_type]


# ---------------------------------------------------------------------------
# Singleton Accessor
# ---------------------------------------------------------------------------

_memory_manager_instance: Optional[MemoryManager] = None

def get_memory_manager(config: MemoryManagerConfig | None = None) -> MemoryManager:
    """
    Returns the singleton instance of the MemoryManager.
    
    If no instance exists, it creates one using the provided config or 
    default settings. This ensures a single source of truth for the 
    memory subsystem across the entire application.
    
    Args:
        config: Optional configuration. Ignored if instance already exists.
        
    Returns:
        The global MemoryManager instance.
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        if config is None:
            config = MemoryManagerConfig()
        _memory_manager_instance = MemoryManager(config)
    return _memory_manager_instance

def reset_memory_manager() -> None:
    """
    Resets the singleton instance. Primarily used for testing purposes.
    """
    global _memory_manager_instance
    _memory_manager_instance = None
