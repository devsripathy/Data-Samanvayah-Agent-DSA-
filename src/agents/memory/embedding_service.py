"""
Embedding Service for the Data Samanvayah Agent (DSA).

This module provides a unified, asynchronous interface for generating text embeddings
using multiple providers (OpenAI, Ollama, Sentence Transformers). It implements the
Strategy Pattern for provider abstraction, includes an async-safe LRU cache,
automatic batching, retry logic, and cosine similarity calculations.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Enums
# ---------------------------------------------------------------------------

class EmbeddingProvider(StrEnum):
    """Supported embedding model providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EmbeddingConfig(BaseModel):
    """Configuration for the Embedding Service."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model_name: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Device configuration (only applies to local models like SentenceTransformers)
    device: str = "cpu"  # "cpu", "cuda", "mps"
    
    # Performance tuning
    batch_size: int = Field(default=32, gt=0)
    cache_max_size: int = Field(default=1000, gt=0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    
    # Resilience
    max_retries: int = Field(default=3, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0)
    
    # Dimensions (can be auto-detected, but useful for overrides)
    expected_dimension: Optional[int] = None


# ---------------------------------------------------------------------------
# Strategy Pattern: Base & Concrete Implementations
# ---------------------------------------------------------------------------

class BaseEmbeddingStrategy(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: Any = None
        self._dimension: Optional[int] = None

    @abstractmethod
    async def _load_model(self) -> None:
        """Lazily loads the underlying embedding model."""
        pass

    @abstractmethod
    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        """Core implementation for embedding a batch of texts."""
        pass

    async def get_dimension(self) -> int:
        """Returns the dimensionality of the embeddings."""
        if self._dimension is None:
            await self._load_model()
            if self.config.expected_dimension:
                self._dimension = self.config.expected_dimension
            else:
                # Fallback: infer from a dummy embedding or provider default
                self._dimension = 1536 if self.config.provider == EmbeddingProvider.OPENAI else 384
        return self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts, ensuring the model is loaded."""
        if self._model is None:
            await self._load_model()
        return await self._embed_batch_impl(texts)


class OpenAIStrategy(BaseEmbeddingStrategy):
    """Strategy for OpenAI embeddings."""

    async def _load_model(self) -> None:
        try:
            from openai import AsyncOpenAI
            self._model = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            logger.info(f"Loaded OpenAI strategy with model: {self.config.model_name}")
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings.")

    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.config.max_retries):
            try:
                response = await self._model.embeddings.create(
                    model=self.config.model_name,
                    input=texts,
                    encoding_format="float"
                )
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"OpenAI embedding failed after {self.config.max_retries} attempts: {e}")
                    raise
                logger.warning(f"OpenAI embedding attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
        return []


class OllamaStrategy(BaseEmbeddingStrategy):
    """Strategy for Ollama embeddings."""

    async def _load_model(self) -> None:
        try:
            import httpx
            self._model = httpx.AsyncClient(base_url=self.config.base_url or "http://localhost:11434")
            logger.info(f"Loaded Ollama strategy with model: {self.config.model_name}")
        except ImportError:
            raise ImportError("httpx package is required for Ollama embeddings.")

    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for attempt in range(self.config.max_retries):
            try:
                # Ollama's /api/embed supports batch input in newer versions
                payload = {"model": self.config.model_name, "input": texts}
                response = await self._model.post("/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Handle both old (embeddings) and new (embeddings) response formats
                return data.get("embeddings", [])
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Ollama embedding failed after {self.config.max_retries} attempts: {e}")
                    raise
                logger.warning(f"Ollama embedding attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
        return embeddings


class SentenceTransformerStrategy(BaseEmbeddingStrategy):
    """Strategy for local Sentence Transformers embeddings."""

    async def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            # Run in thread pool to avoid blocking the event loop during model load
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.config.model_name, device=self.config.device)
            )
            logger.info(f"Loaded SentenceTransformer strategy: {self.config.model_name} on {self.config.device}")
        except ImportError:
            raise ImportError("sentence-transformers package is required for local embeddings.")

    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        # SentenceTransformers encode is synchronous, so we run it in a thread pool
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, show_progress_bar=False).tolist()
        )
        return embeddings


# ---------------------------------------------------------------------------
# Async LRU Cache
# ---------------------------------------------------------------------------

class AsyncEmbeddingCache:
    """Thread-safe, async-compatible LRU cache for embeddings."""

    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._cache: dict[str, tuple[float, list[list[float]]]] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hash_batch(self, texts: list[str]) -> str:
        combined = "".join(texts)
        return self._hash_text(combined)

    async def get(self, texts: list[str]) -> Optional[list[list[float]]]:
        key = self._hash_batch(texts)
        async with self._lock:
            if key in self._cache:
                timestamp, embeddings = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return embeddings
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    async def set(self, texts: list[str], embeddings: list[list[float]]) -> None:
        key = self._hash_batch(texts)
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict oldest item
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), embeddings)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ---------------------------------------------------------------------------
# Main Embedding Service (Facade)
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Main facade for the DSA Embedding subsystem.
    
    Handles provider selection, lazy initialization, batching, caching,
    and similarity calculations.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._strategy: Optional[BaseEmbeddingStrategy] = None
        self._cache = AsyncEmbeddingCache(config.cache_max_size, config.cache_ttl_seconds)
        self._init_lock = asyncio.Lock()
        logger.info(f"EmbeddingService initialized with provider: {config.provider}")

    def _get_strategy(self) -> BaseEmbeddingStrategy:
        """Factory method to instantiate the correct strategy."""
        if self.config.provider == EmbeddingProvider.OPENAI:
            return OpenAIStrategy(self.config)
        elif self.config.provider == EmbeddingProvider.OLLAMA:
            return OllamaStrategy(self.config)
        elif self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerStrategy(self.config)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.provider}")

    async def _ensure_strategy(self) -> BaseEmbeddingStrategy:
        """Ensures the strategy is initialized (lazy loading)."""
        if self._strategy is None:
            async with self._init_lock:
                if self._strategy is None:
                    self._strategy = self._get_strategy()
        return self._strategy

    async def embed_text(self, text: str) -> list[float]:
        """
        Embeds a single text string.
        
        Args:
            text: The input text to embed.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of text strings with automatic batching and caching.
        
        Args:
            texts: List of input texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        start_time = time.perf_counter()
        strategy = await self._ensure_strategy()
        
        # Check cache first
        cached = await self._cache.get(texts)
        if cached is not None:
            logger.debug(f"Cache hit for batch of {len(texts)} texts.")
            return cached

        # Split into chunks if necessary
        all_embeddings: list[list[float]] = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            chunk_embeddings = await strategy.embed_batch(chunk)
            all_embeddings.extend(chunk_embeddings)

        # Store in cache
        await self._cache.set(texts, all_embeddings)
        
        duration = time.perf_counter() - start_time
        logger.info(f"Embedded {len(texts)} texts in {duration:.2f}s. Cache stats: {self._cache.stats}")
        
        return all_embeddings

    async def get_dimension(self) -> int:
        """Returns the dimensionality of the embeddings for the current model."""
        strategy = await self._ensure_strategy()
        return await strategy.get_dimension()

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        
        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def similarity_matrix(self, vectors: list[list[float]]) -> np.ndarray:
        """
        Computes a pairwise cosine similarity matrix for a list of vectors.
        
        Args:
            vectors: List of embedding vectors.
            
        Returns:
            NxN numpy array of similarity scores.
        """
        if not vectors:
            return np.array([])
            
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms == 0] = 1e-8
        normalized = matrix / norms
        
        return np.dot(normalized, normalized.T)

    async def close(self) -> None:
        """Cleans up resources (e.g., closing HTTP clients)."""
        if self._strategy and hasattr(self._strategy, "_model"):
            model = self._strategy._model
            if hasattr(model, "close"):
                await model.close()
            elif hasattr(model, "aclose"):
                await model.aclose()
        logger.info("EmbeddingService resources closed.")
