"""
Schema definitions for the DSA Memory subsystem.

This module defines every data model used across the memory subsystem,
including records for semantic, episodic, and procedural memory, along
with retrieval candidates, results, statistics, and configuration.

No database logic, business logic, or retrieval logic is included.
Only strongly typed Pydantic v2 schemas.

Typical usage:
    from src.memory.schemas import SemanticMemoryRecord, MemoryConfiguration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Generic Type for from_dict
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(StrEnum):
    """Enumeration of memory record types within the DSA subsystem."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory record."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryImportance(StrEnum):
    """Priority tier assigned to a memory record for retention and retrieval."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Shared / Composite Metadata Models
# ---------------------------------------------------------------------------

class MemoryMetadata(BaseModel):
    """
    Core identity and provenance metadata shared by every memory record.

    Attributes:
        memory_id: Globally unique identifier for the memory record.
        execution_id: Identifier of the DSA pipeline execution that produced
            or is associated with this record.
        dataset_hash: Deterministic hash of the source dataset contents.
        dataset_name: Human-readable name of the source dataset.
        created_at: UTC timestamp when the record was first created.
        updated_at: UTC timestamp when the record was last modified.
        version: Monotonically increasing version counter for the record.
        owner: Identifier of the agent or subsystem that owns this record.
        tags: Arbitrary string labels for filtering and grouping.
        description: Human-readable summary of the record's purpose.
        status: Current lifecycle status of the record.
        importance: Priority tier governing retention policy.
        memory_type: Discriminator indicating the kind of memory.
    """

    model_config = ConfigDict(
        frozen=False,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    memory_id: str = Field(
        ...,
        min_length=1,
        description="Globally unique identifier for the memory record.",
    )
    execution_id: str = Field(
        default="",
        description="Identifier of the pipeline execution associated with this record.",
    )
    dataset_hash: str = Field(
        default="",
        description="Deterministic content hash of the source dataset.",
    )
    dataset_name: str = Field(
        default="",
        description="Human-readable name of the source dataset.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the record was first created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the record was last modified.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Monotonically increasing version counter.",
    )
    owner: str = Field(
        default="system",
        description="Identifier of the owning agent or subsystem.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary string labels for filtering and grouping.",
    )
    description: str = Field(
        default="",
        description="Human-readable summary of the record's purpose.",
    )
    status: MemoryStatus = Field(
        default=MemoryStatus.ACTIVE,
        description="Current lifecycle status of the record.",
    )
    importance: MemoryImportance = Field(
        default=MemoryImportance.MEDIUM,
        description="Priority tier governing retention policy.",
    )
    memory_type: MemoryType = Field(
        default=MemoryType.SEMANTIC,
        description="Discriminator indicating the kind of memory.",
    )

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_timezone_aware(cls, v: Any) -> datetime:
        """Ensures all datetime values are timezone-aware (UTC)."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryMetadata":
        """Deserializes a dictionary into a MemoryMetadata instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary of the metadata."""
        return (
            f"MemoryMetadata(id={self.memory_id}, type={self.memory_type}, "
            f"status={self.status}, importance={self.importance}, "
            f"version={self.version}, owner={self.owner})"
        )


class EmbeddingMetadata(BaseModel):
    """
    Provenance and quality metadata for a computed embedding vector.

    Attributes:
        embedding_model: Name or identifier of the embedding model used.
        embedding_dimension: Dimensionality of the embedding vector.
        embedding_provider: Backend provider (e.g. openai, ollama, sentence-transformers).
        vector_norm: L2 norm of the embedding vector (for normalization checks).
        token_count: Number of tokens consumed during embedding computation.
        embedding_time_ms: Wall-clock time in milliseconds to compute the embedding.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Name or identifier of the embedding model used.",
    )
    embedding_dimension: int = Field(
        default=1536,
        gt=0,
        description="Dimensionality of the embedding vector.",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Backend provider for embedding computation.",
    )
    vector_norm: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 norm of the embedding vector.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Number of tokens consumed during embedding.",
    )
    embedding_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time in milliseconds to compute the embedding.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingMetadata":
        """Deserializes a dictionary into an EmbeddingMetadata instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"EmbeddingMetadata(model={self.embedding_model}, dim={self.embedding_dimension}, "
            f"provider={self.embedding_provider}, norm={self.vector_norm:.4f}, "
            f"tokens={self.token_count}, time={self.embedding_time_ms:.1f}ms)"
        )


class MemoryQuality(BaseModel):
    """
    Quality and usage tracking metrics for a memory record.

    These scores are updated over time as the record is retrieved and
    evaluated by downstream agents.

    Attributes:
        similarity_score: Average cosine similarity when this record is retrieved.
        quality_score: Composite quality assessment (0.0 to 1.0).
        confidence_score: Confidence in the record's accuracy (0.0 to 1.0).
        usage_count: Total number of times this record has been accessed.
        successful_retrievals: Count of retrievals that led to a positive outcome.
        failed_retrievals: Count of retrievals that led to a negative outcome.
        last_used: UTC timestamp of the most recent access.
        average_improvement: Mean performance improvement attributed to this record.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average cosine similarity when this record is retrieved.",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite quality assessment.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the record's accuracy.",
    )
    usage_count: int = Field(
        default=0,
        ge=0,
        description="Total number of times this record has been accessed.",
    )
    successful_retrievals: int = Field(
        default=0,
        ge=0,
        description="Count of retrievals that led to a positive outcome.",
    )
    failed_retrievals: int = Field(
        default=0,
        ge=0,
        description="Count of retrievals that led to a negative outcome.",
    )
    last_used: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the most recent access.",
    )
    average_improvement: float = Field(
        default=0.0,
        description="Mean performance improvement attributed to this record.",
    )

    @field_validator("last_used", mode="before")
    @classmethod
    def _ensure_timezone_aware(cls, v: Any) -> Optional[datetime]:
        """Ensures datetime values are timezone-aware (UTC)."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryQuality":
        """Deserializes a dictionary into a MemoryQuality instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"MemoryQuality(similarity={self.similarity_score:.3f}, "
            f"quality={self.quality_score:.3f}, confidence={self.confidence_score:.3f}, "
            f"used={self.usage_count}x, success={self.successful_retrievals}, "
            f"fail={self.failed_retrievals})"
        )


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

class SemanticMemoryRecord(BaseModel):
    """
    Schema for a semantic memory record storing vector-indexed dataset knowledge.

    Semantic memory captures the structural and statistical essence of a dataset
    in a form that can be retrieved via similarity search.

    Attributes:
        metadata: Core identity and provenance information.
        embedding_metadata: Provenance of the embedding vector.
        summary: Natural-language summary of the dataset.
        dataset_schema: Mapping of column names to inferred types.
        column_metadata: Per-column statistical metadata.
        dataset_statistics: Aggregate statistics for the dataset.
        embedding: The dense vector representation.
        quality: Quality and usage tracking metrics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    metadata: MemoryMetadata = Field(
        default_factory=lambda: MemoryMetadata(
            memory_id="", memory_type=MemoryType.SEMANTIC
        ),
        description="Core identity and provenance information.",
    )
    embedding_metadata: EmbeddingMetadata = Field(
        default_factory=EmbeddingMetadata,
        description="Provenance of the embedding vector.",
    )
    summary: str = Field(
        default="",
        description="Natural-language summary of the dataset.",
    )
    dataset_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to inferred types.",
    )
    column_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-column statistical metadata (min, max, mean, null_count, etc.).",
    )
    dataset_statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate statistics for the dataset (row count, size, etc.).",
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Dense vector representation of the dataset summary.",
    )
    quality: MemoryQuality = Field(
        default_factory=MemoryQuality,
        description="Quality and usage tracking metrics.",
    )

    @model_validator(mode="after")
    def _enforce_semantic_type(self) -> "SemanticMemoryRecord":
        """Ensures the metadata memory_type is always SEMANTIC."""
        self.metadata.memory_type = MemoryType.SEMANTIC
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticMemoryRecord":
        """Deserializes a dictionary into a SemanticMemoryRecord instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        cols = len(self.dataset_schema)
        emb_dim = len(self.embedding)
        return (
            f"SemanticMemoryRecord(id={self.metadata.memory_id}, "
            f"dataset={self.metadata.dataset_name}, columns={cols}, "
            f"embedding_dim={emb_dim}, quality={self.quality.quality_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

class EpisodeStep(BaseModel):
    """
    Schema for a single step within an episodic memory trace.

    Each step records the action taken by one agent during a pipeline execution.

    Attributes:
        agent_name: Name of the agent that executed this step.
        input_summary: Concise description of the input to this step.
        output_summary: Concise description of the output from this step.
        execution_time: Wall-clock duration of the step in seconds.
        status: Outcome status (e.g. 'success', 'failed', 'skipped').
        timestamp: UTC timestamp when the step began execution.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    agent_name: str = Field(
        ...,
        min_length=1,
        description="Name of the agent that executed this step.",
    )
    input_summary: str = Field(
        default="",
        description="Concise description of the input to this step.",
    )
    output_summary: str = Field(
        default="",
        description="Concise description of the output from this step.",
    )
    execution_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of the step in seconds.",
    )
    status: str = Field(
        default="success",
        description="Outcome status of the step.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the step began execution.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_timezone_aware(cls, v: Any) -> datetime:
        """Ensures datetime values are timezone-aware (UTC)."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeStep":
        """Deserializes a dictionary into an EpisodeStep instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"EpisodeStep(agent={self.agent_name}, status={self.status}, "
            f"time={self.execution_time:.2f}s)"
        )


class EpisodicMemoryRecord(BaseModel):
    """
    Schema for an episodic memory record capturing a complete pipeline execution.

    Episodic memory preserves the full trajectory of a single DSA run so that
    future runs can learn from past successes and failures.

    Attributes:
        metadata: Core identity and provenance information.
        planner_result: Structured output from the Planner agent.
        trainer_result: Structured output from the Trainer agent.
        critic_result: Structured output from the Critic agent.
        execution_metrics: Aggregate timing and resource metrics for the run.
        steps: Ordered list of agent execution steps.
        quality: Quality and usage tracking metrics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    metadata: MemoryMetadata = Field(
        default_factory=lambda: MemoryMetadata(
            memory_id="", memory_type=MemoryType.EPISODIC
        ),
        description="Core identity and provenance information.",
    )
    planner_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output from the Planner agent.",
    )
    trainer_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output from the Trainer agent.",
    )
    critic_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output from the Critic agent.",
    )
    execution_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate timing and resource metrics for the run.",
    )
    steps: list[EpisodeStep] = Field(
        default_factory=list,
        description="Ordered list of agent execution steps.",
    )
    quality: MemoryQuality = Field(
        default_factory=MemoryQuality,
        description="Quality and usage tracking metrics.",
    )

    @model_validator(mode="after")
    def _enforce_episodic_type(self) -> "EpisodicMemoryRecord":
        """Ensures the metadata memory_type is always EPISODIC."""
        self.metadata.memory_type = MemoryType.EPISODIC
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicMemoryRecord":
        """Deserializes a dictionary into an EpisodicMemoryRecord instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"EpisodicMemoryRecord(id={self.metadata.memory_id}, "
            f"dataset={self.metadata.dataset_name}, steps={len(self.steps)}, "
            f"quality={self.quality.quality_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Procedural Memory
# ---------------------------------------------------------------------------

class ProceduralRule(BaseModel):
    """
    Schema for a procedural memory rule (learned IF-THEN heuristic).

    Procedural rules are distilled from successful episodic runs and encode
    reusable decision patterns for the Planner agent.

    Attributes:
        metadata: Core identity and provenance information.
        rule_name: Human-readable name for the rule.
        conditions: Mapping of condition keys to threshold values (IF part).
        actions: Mapping of action keys to recommended values (THEN part).
        confidence: Confidence in the rule's correctness (0.0 to 1.0).
        times_applied: Number of times this rule has been applied in planning.
        success_rate: Fraction of successful outcomes when this rule was followed.
        quality: Quality and usage tracking metrics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    metadata: MemoryMetadata = Field(
        default_factory=lambda: MemoryMetadata(
            memory_id="", memory_type=MemoryType.PROCEDURAL
        ),
        description="Core identity and provenance information.",
    )
    rule_name: str = Field(
        default="",
        description="Human-readable name for the rule.",
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of condition keys to threshold values (IF part).",
    )
    actions: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of action keys to recommended values (THEN part).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the rule's correctness.",
    )
    times_applied: int = Field(
        default=0,
        ge=0,
        description="Number of times this rule has been applied in planning.",
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of successful outcomes when this rule was followed.",
    )
    quality: MemoryQuality = Field(
        default_factory=MemoryQuality,
        description="Quality and usage tracking metrics.",
    )

    @model_validator(mode="after")
    def _enforce_procedural_type(self) -> "ProceduralRule":
        """Ensures the metadata memory_type is always PROCEDURAL."""
        self.metadata.memory_type = MemoryType.PROCEDURAL
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProceduralRule":
        """Deserializes a dictionary into a ProceduralRule instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"ProceduralRule(name={self.rule_name!r}, id={self.metadata.memory_id}, "
            f"confidence={self.confidence:.3f}, applied={self.times_applied}x, "
            f"success_rate={self.success_rate:.3f})"
        )


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------

class RetrievalCandidate(BaseModel):
    """
    Schema for a single candidate produced during memory retrieval.

    Candidates are scored across multiple dimensions before final ranking.

    Attributes:
        memory_id: Identifier of the candidate memory record.
        similarity: Cosine similarity between the query and candidate embedding.
        quality: Quality score of the candidate record.
        recency_score: Normalized recency score (newer records score higher).
        combined_score: Weighted aggregate of all scoring dimensions.
        reasoning: Human-readable explanation of why this candidate was surfaced.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    memory_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the candidate memory record.",
    )
    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the query and candidate embedding.",
    )
    quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score of the candidate record.",
    )
    recency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized recency score.",
    )
    combined_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted aggregate of all scoring dimensions.",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of why this candidate was surfaced.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalCandidate":
        """Deserializes a dictionary into a RetrievalCandidate instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"RetrievalCandidate(id={self.memory_id}, similarity={self.similarity:.3f}, "
            f"quality={self.quality:.3f}, recency={self.recency_score:.3f}, "
            f"combined={self.combined_score:.3f})"
        )


class RetrievalResult(BaseModel):
    """
    Schema for the complete output of a memory retrieval operation.

    Attributes:
        query: The original query string or structured query descriptor.
        retrieved_items: Ordered list of scored retrieval candidates.
        execution_time: Wall-clock time for the retrieval in milliseconds.
        strategy_used: Identifier of the retrieval strategy applied.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    query: str = Field(
        default="",
        description="The original query string or structured query descriptor.",
    )
    retrieved_items: list[RetrievalCandidate] = Field(
        default_factory=list,
        description="Ordered list of scored retrieval candidates.",
    )
    execution_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time for the retrieval in milliseconds.",
    )
    strategy_used: str = Field(
        default="hybrid",
        description="Identifier of the retrieval strategy applied.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """Deserializes a dictionary into a RetrievalResult instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"RetrievalResult(query={self.query!r}, items={len(self.retrieved_items)}, "
            f"time={self.execution_time:.1f}ms, strategy={self.strategy_used})"
        )


# ---------------------------------------------------------------------------
# Statistics & Configuration
# ---------------------------------------------------------------------------

class MemoryStatistics(BaseModel):
    """
    Aggregate statistics describing the current state of the memory subsystem.

    Attributes:
        total_records: Total number of records across all memory types.
        semantic_records: Count of semantic memory records.
        episodic_records: Count of episodic memory records.
        procedural_records: Count of procedural memory records.
        average_similarity: Mean similarity score across recent retrievals.
        average_quality: Mean quality score across all active records.
        database_size: Estimated total storage size in bytes.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    total_records: int = Field(
        default=0,
        ge=0,
        description="Total number of records across all memory types.",
    )
    semantic_records: int = Field(
        default=0,
        ge=0,
        description="Count of semantic memory records.",
    )
    episodic_records: int = Field(
        default=0,
        ge=0,
        description="Count of episodic memory records.",
    )
    procedural_records: int = Field(
        default=0,
        ge=0,
        description="Count of procedural memory records.",
    )
    average_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean similarity score across recent retrievals.",
    )
    average_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean quality score across all active records.",
    )
    database_size: int = Field(
        default=0,
        ge=0,
        description="Estimated total storage size in bytes.",
    )

    @model_validator(mode="after")
    def _validate_total(self) -> "MemoryStatistics":
        """Ensures total_records is at least the sum of sub-type counts."""
        computed = self.semantic_records + self.episodic_records + self.procedural_records
        if self.total_records < computed:
            self.total_records = computed
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStatistics":
        """Deserializes a dictionary into a MemoryStatistics instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"MemoryStatistics(total={self.total_records}, semantic={self.semantic_records}, "
            f"episodic={self.episodic_records}, procedural={self.procedural_records}, "
            f"avg_quality={self.average_quality:.3f}, size={self.database_size}B)"
        )


class MemoryConfiguration(BaseModel):
    """
    Runtime configuration for the memory subsystem.

    Controls retrieval behavior, embedding parameters, caching policy,
    and vector database connectivity.

    Attributes:
        top_k: Maximum number of candidates to retrieve per query.
        similarity_threshold: Minimum cosine similarity for a candidate to be included.
        minimum_quality: Minimum quality score for a record to be retrievable.
        embedding_model: Name of the embedding model to use.
        vector_database: Identifier or connection string for the vector database.
        cache_enabled: Whether in-memory caching of retrieval results is active.
        cache_size: Maximum number of entries in the retrieval cache.
    """

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    top_k: int = Field(
        default=10,
        gt=0,
        description="Maximum number of candidates to retrieve per query.",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for candidate inclusion.",
    )
    minimum_quality: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for a record to be retrievable.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Name of the embedding model to use.",
    )
    vector_database: str = Field(
        default="chromadb",
        description="Identifier or connection string for the vector database.",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether in-memory caching of retrieval results is active.",
    )
    cache_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of entries in the retrieval cache.",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "MemoryConfiguration":
        """Ensures similarity_threshold is not lower than minimum_quality in a degenerate way."""
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfiguration":
        """Deserializes a dictionary into a MemoryConfiguration instance."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Returns a concise human-readable summary."""
        return (
            f"MemoryConfiguration(top_k={self.top_k}, threshold={self.similarity_threshold}, "
            f"min_quality={self.minimum_quality}, model={self.embedding_model!r}, "
            f"vector_db={self.vector_database!r}, cache={self.cache_enabled})"
        )
