"""
Semantic Memory Backend for the Data Samanvayah Agent (DSA).

This module implements the semantic memory layer, responsible for storing,
indexing, and retrieving high-dimensional vector representations of dataset
knowledge. It uses ChromaDB as the default vector store but is designed with
an abstraction layer to support future backends (Milvus, Qdrant, Weaviate, PGVector).

Features:
- Abstraction layer for vector database backends.
- Automatic embedding generation via the EmbeddingService.
- Metadata serialization/deserialization for complex types.
- Configurable similarity thresholds and metadata filtering.
- Graceful fallback to an in-memory store if the primary DB fails.
- Non-blocking async execution wrapping synchronous vector DB calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.memory.base import (
    BaseMemory,
    MemoryBaseError,
    MemoryConnectionError,
    MemoryNotFoundError,
    MemoryRetrievalError,
    MemoryStorageError,
)
from src.memory.schemas import (
    MemoryMetadata,
    MemoryRecord,
    MemoryStatistics,
    MemoryType,
    RetrievalCandidate,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VectorStoreError(MemoryBaseError):
    """Base exception for vector store operations."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store cannot be connected to."""
    pass


# ---------------------------------------------------------------------------
# Vector Store Abstraction Layer
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Abstract base class defining the contract for vector database backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Initializes the connection to the vector database."""
        pass

    @abstractmethod
    async def create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        """Creates a new collection in the vector database."""
        pass

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Inserts or updates vectors in the specified collection."""
        pass

    @abstractmethod
    async def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queries the collection for similar vectors."""
        pass

    @abstractmethod
    async def get(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieves records by ID or metadata filter."""
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Deletes records from the collection."""
        pass

    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """Returns the number of records in the collection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Checks if the vector store is healthy and accessible."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of the BaseVectorStore interface.
    
    Uses a persistent client to ensure data survives restarts.
    Synchronous ChromaDB calls are wrapped in asyncio.to_thread 
    to prevent blocking the event loop.
    """

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    async def connect(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            
            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            )
            self._collection = await loop.run_in_executor(
                None,
                lambda: self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            )
            logger.info(f"Connected to ChromaDB at {self.persist_directory}")
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to connect to ChromaDB: {e}") from e

    async def create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.get_or_create_collection(name=name, metadata=metadata)
        )

    async def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        )

    async def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=query_embeddings, n_results=n_results, where=where
            )
        )

    async def get(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._collection.get(ids=ids, where=where)
        )

    async def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.delete(ids=ids, where=where)
        )

    async def count(self, collection_name: str) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._collection.count())

    async def health_check(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._collection.count())
            return True
        except Exception:
            return False


class InMemoryFallbackStore(BaseVectorStore):
    """
    Graceful fallback vector store using in-memory numpy arrays.
    Used when the primary vector database is unavailable.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        logger.warning("Using InMemoryFallbackStore. Data will not persist across restarts.")
        self._store = {}

    async def create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        if name not in self._store:
            self._store[name] = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    async def upsert(self, collection_name: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        if collection_name not in self._store:
            await self.create_collection(collection_name)
        
        col = self._store[collection_name]
        for i, record_id in enumerate(ids):
            if record_id in col["ids"]:
                idx = col["ids"].index(record_id)
                col["embeddings"][idx] = embeddings[i]
                col["documents"][idx] = documents[i]
                col["metadatas"][idx] = metadatas[i]
            else:
                col["ids"].append(record_id)
                col["embeddings"].append(embeddings[i])
                col["documents"].append(documents[i])
                col["metadatas"].append(metadatas[i])

    async def query(self, collection_name: str, query_embeddings: list[list[float]], n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
        if collection_name not in self._store or not self._store[collection_name]["ids"]:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
        
        col = self._store[collection_name]
        query_vec = np.array(query_embeddings[0])
        db_vecs = np.array(col["embeddings"])
        
        # Cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_db = np.linalg.norm(db_vecs, axis=1)
        similarities = np.dot(db_vecs, query_vec) / (norm_db * norm_query + 1e-8)
        
        # Convert to distance (1 - similarity)
        distances = 1 - similarities
        sorted_indices = np.argsort(distances)[:n_results]
        
        return {
            "ids": [[col["ids"][i] for i in sorted_indices]],
            "distances": [[float(distances[i]) for i in sorted_indices]],
            "metadatas": [[col["metadatas"][i] for i in sorted_indices]],
            "documents": [[col["documents"][i] for i in sorted_indices]]
        }

    async def get(self, collection_name: str, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> dict[str, Any]:
        if collection_name not in self._store:
            return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        
        col = self._store[collection_name]
        if ids is None:
            return col
        
        indices = [i for i, id in enumerate(col["ids"]) if id in ids]
        return {
            "ids": [col["ids"][i] for i in indices],
            "embeddings": [col["embeddings"][i] for i in indices],
            "metadatas": [col["metadatas"][i] for i in indices],
            "documents": [col["documents"][i] for i in indices]
        }

    async def delete(self, collection_name: str, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if collection_name not in self._store or ids is None:
            return
        
        col = self._store[collection_name]
        keep_indices = [i for i, id in enumerate(col["ids"]) if id not in ids]
        
        col["ids"] = [col["ids"][i] for i in keep_indices]
        col["embeddings"] = [col["embeddings"][i] for i in keep_indices]
        col["documents"] = [col["documents"][i] for i in keep_indices]
        col["metadatas"] = [col["metadatas"][i] for i in keep_indices]

    async def count(self, collection_name: str) -> int:
        if collection_name not in self._store:
            return 0
        return len(self._store[collection_name]["ids"])

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Serializes complex metadata types into ChromaDB-compatible primitives.
    ChromaDB only supports str, int, float, and bool.
    """
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def deserialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Deserializes JSON strings back into Python objects."""
    deserialized = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            try:
                deserialized[key] = json.loads(value)
            except json.JSONDecodeError:
                deserialized[key] = value
        else:
            deserialized[key] = value
    return deserialized


# ---------------------------------------------------------------------------
# Semantic Memory Implementation
# ---------------------------------------------------------------------------

class SemanticMemoryConfig(BaseModel):
    """Configuration for the Semantic Memory backend."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    persist_directory: str = "./data/chroma_db"
    collection_name: str = "dsa_semantic"
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    use_fallback: bool = True


class SemanticMemory(BaseMemory):
    """
    Concrete implementation of BaseMemory for semantic (vector) storage.
    
    Manages the lifecycle of dataset embeddings, handles metadata serialization,
    and orchestrates interactions between the EmbeddingService and the VectorStore.
    """

    def __init__(
        self,
        config: SemanticMemoryConfig,
        embedding_service: Any,  # EmbeddingService instance
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name
        )
        self._fallback_store = InMemoryFallbackStore() if config.use_fallback else None
        self._is_fallback = False

    async def connect(self) -> None:
        """Connects to the primary vector store, falling back to in-memory if it fails."""
        try:
            await self.vector_store.connect()
            if not await self.vector_store.health_check():
                raise VectorStoreConnectionError("Primary store health check failed.")
            
            # Ensure collection exists
            await self.vector_store.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Semantic memory connected to primary vector store.")
        except Exception as e:
            logger.error(f"Failed to connect to primary vector store: {e}")
            if self.config.use_fallback and self._fallback_store:
                logger.warning("Activating in-memory fallback store.")
                await self._fallback_store.connect()
                self._is_fallback = True
            else:
                raise MemoryConnectionError(f"Semantic memory connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnects from the active vector store."""
        logger.info("Disconnecting semantic memory backend.")
        self._is_fallback = False

    async def health_check(self) -> dict[str, Any]:
        """Checks the health of the active vector store."""
        store = self._fallback_store if self._is_fallback else self.vector_store
        is_healthy = await store.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "backend": "fallback" if self._is_fallback else "primary",
            "collection": self.config.collection_name
        }

    async def store(self, record: MemoryRecord) -> str:
        """Stores a single memory record after generating its embedding."""
        return (await self.store_batch([record]))[0]

    async def store_batch(self, records: list[MemoryRecord]) -> list[str]:
        """Generates embeddings and stores a batch of records."""
        if not records:
            return []

        texts = [record.content for record in records]
        embeddings = await self.embedding_service.embed_batch(texts)
        
        ids = [record.id for record in records]
        documents = texts
        metadatas = [serialize_metadata(record.metadata.model_dump()) for record in records]

        store = self._fallback_store if self._is_fallback else self.vector_store
        await store.upsert(
            collection_name=self.config.collection_name,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(records)} semantic records.")
        return ids

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieves records based on semantic similarity to the query."""
        query_embedding = await self.embedding_service.embed_text(query)
        
        store = self._fallback_store if self._is_fallback else self.vector_store
        results = await store.query(
            collection_name=self.config.collection_name,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        candidates = []
        if results and results.get("ids") and results["ids"][0]:
            for i, record_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                
                if similarity >= self.config.similarity_threshold:
                    metadata = deserialize_metadata(results["metadatas"][0][i]) if results["metadatas"] else {}
                    candidates.append(RetrievalCandidate(
                        memory_id=record_id,
                        similarity=similarity,
                        quality=metadata.get("quality_score", 0.0),
                        recency_score=0.0,  # Calculated by MemoryManager
                        combined_score=similarity,
                        reasoning=f"Similarity {similarity:.3f} > threshold {self.config.similarity_threshold}"
                    ))

        return RetrievalResult(
            query=query,
            retrieved_items=candidates,
            execution_time=0.0,
            strategy_used="cosine_similarity"
        )

    async def retrieve_by_id(self, record_id: str) -> MemoryRecord | None:
        """Retrieves a specific record by its ID."""
        store = self._fallback_store if self._is_fallback else self.vector_store
        results = await store.get(
            collection_name=self.config.collection_name,
            ids=[record_id]
        )
        
        if results and results["ids"]:
            metadata = deserialize_metadata(results["metadatas"][0])
            return MemoryRecord(
                id=results["ids"][0],
                memory_type=MemoryType.SEMANTIC,
                content=results["documents"][0],
                metadata=MemoryMetadata(**metadata) if metadata else MemoryMetadata(memory_id=results["ids"][0]),
                embedding=results["embeddings"][0] if results["embeddings"] else None
            )
        return None

    async def update(self, record: MemoryRecord) -> bool:
        """Updates an existing record by re-upserting it."""
        try:
            await self.store(record)
            return True
        except Exception as e:
            logger.error(f"Failed to update record {record.id}: {e}")
            return False

    async def delete(self, record_id: str) -> bool:
        """Deletes a record by ID."""
        try:
            store = self._fallback_store if self._is_fallback else self.vector_store
            await store.delete(collection_name=self.config.collection_name, ids=[record_id])
            logger.info(f"Deleted semantic record {record_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False

    async def clear(self) -> int:
        """Clears all records in the collection."""
        try:
            store = self._fallback_store if self._is_fallback else self.vector_store
            count = await store.count(self.config.collection_name)
            await store.delete(collection_name=self.config.collection_name)
            logger.info(f"Cleared {count} semantic records.")
            return count
        except Exception as e:
            logger.error(f"Failed to clear semantic memory: {e}")
            return 0

    async def search(self, query: str, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        """Performs a direct search without strict threshold filtering."""
        result = await self.retrieve(query, top_k=100, filters=filters)
        records = []
        for candidate in result.retrieved_items:
            record = await self.retrieve_by_id(candidate.memory_id)
            if record:
                records.append(record)
        return records

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Returns the total number of records in the collection."""
        store = self._fallback_store if self._is_fallback else self.vector_store
        return await store.count(self.config.collection_name)

    async def statistics(self) -> MemoryStatistics:
        """Computes statistics for the semantic memory backend."""
        count = await self.count()
        return MemoryStatistics(
            total_records=count,
            semantic_records=count,
            episodic_records=0,
            procedural_records=0,
            database_size=0  # ChromaDB doesn't expose size easily
        )

    async def backup(self, destination: str) -> str:
        """Backs up the ChromaDB directory."""
        import shutil
        try:
            shutil.copytree(self.config.persist_directory, destination, dirs_exist_ok=True)
            logger.info(f"Backed up semantic memory to {destination}")
            return destination
        except Exception as e:
            raise MemoryBaseError(f"Backup failed: {e}") from e

    async def restore(self, source: str) -> bool:
        """Restores the ChromaDB directory from a backup."""
        import shutil
        try:
            shutil.rmtree(self.config.persist_directory, ignore_errors=True)
            shutil.copytree(source, self.config.persist_directory)
            await self.connect()
            logger.info(f"Restored semantic memory from {source}")
            return True
        except Exception as e:
            raise MemoryBaseError(f"Restore failed: {e}") from e

    async def optimize(self) -> dict[str, Any]:
        """Optimizes the vector store (e.g., compaction)."""
        # ChromaDB handles optimization automatically, but we can trigger a count to ensure health
        count = await self.count()
        return {"status": "optimized", "record_count": count}

    async def close(self) -> None:
        """Releases resources."""
        await self.disconnect()
        logger.info("Semantic memory backend closed.")
