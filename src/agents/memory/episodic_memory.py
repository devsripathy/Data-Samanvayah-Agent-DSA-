"""
Episodic Memory Backend for the Data Samanvayah Agent (DSA).

This module implements the episodic memory layer, responsible for storing
and retrieving historical execution records of the DSA pipeline. It uses
SQLModel (SQLAlchemy) for ORM capabilities with SQLite as the default backend
and full PostgreSQL compatibility for production deployments.

Features:
- SQLModel-based ORM with async support
- Automatic schema migrations via Alembic
- Comprehensive indexing for fast queries
- Retention policies for automatic cleanup
- Transaction management for data integrity
- Complex querying by dataset fingerprint, execution ID, date range, etc.
- JSON field support for flexible metadata storage
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import (
    JSON,
    Column,
    create_engine,
    delete,
    select,
    SQLModel,
    Field as SQLModelField,
    Relationship,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.memory.base import (
    BaseMemory,
    MemoryBaseError,
    MemoryConnectionError,
    MemoryNotFoundError,
    MemoryRetrievalError,
    MemoryStorageError,
)
from src.memory.schemas import (
    EpisodicMemoryRecord,
    EpisodeStep,
    MemoryMetadata,
    MemoryQuality,
    MemoryStatistics,
    MemoryType,
    ExecutionOutcome,
    RetrievalResult,
    RetrievalCandidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database Models (SQLModel)
# ---------------------------------------------------------------------------

class ExecutionTable(SQLModel, table=True):
    """
    SQLModel table definition for episodic memory records.
    
    This table stores the complete execution history of DSA pipeline runs,
    including planner decisions, training results, critic feedback, and
    agent execution traces.
    """
    
    __tablename__ = "episodic_memory"
    __table_args__ = (
        # Indexes for common query patterns
        {"sqlite_auto_index": False},
    )
    
    # Primary Key
    id: str = SQLModelField(
        primary_key=True,
        index=True,
        description="Unique execution identifier"
    )
    
    # Core Metadata
    execution_id: str = SQLModelField(
        index=True,
        description="DSA execution ID"
    )
    dataset_hash: str = SQLModelField(
        index=True,
        description="Dataset fingerprint/hash"
    )
    dataset_name: str = SQLModelField(
        index=True,
        description="Human-readable dataset name"
    )
    dataset_rows: int = SQLModelField(default=0, description="Number of rows")
    dataset_columns: int = SQLModelField(default=0, description="Number of columns")
    dataset_size_bytes: int = SQLModelField(default=0, description="Dataset size in bytes")
    
    # Task Information
    task_type: str = SQLModelField(
        default="unknown",
        index=True,
        description="ML task type (classification, regression, etc.)"
    )
    target_column: Optional[str] = SQLModelField(
        default=None,
        description="Target column name"
    )
    
    # Planning & Preprocessing
    suggested_models: list[str] = SQLModelField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Models suggested by planner"
    )
    preprocessing_steps: list[str] = SQLModelField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Applied preprocessing steps"
    )
    planner_reasoning: str = SQLModelField(
        default="",
        description="Planner's reasoning for decisions"
    )
    
    # Training Results
    best_model: Optional[str] = SQLModelField(
        default=None,
        description="Best performing model"
    )
    metrics: dict[str, float] = SQLModelField(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Training metrics (accuracy, f1, etc.)"
    )
    training_time_seconds: float = SQLModelField(
        default=0.0,
        description="Total training time"
    )
    
    # Outcome
    outcome: str = SQLModelField(
        default="success",
        index=True,
        description="Execution outcome (success, failure, partial, timeout)"
    )
    failure_reason: Optional[str] = SQLModelField(
        default=None,
        description="Reason for failure if applicable"
    )
    critic_recommendations: list[str] = SQLModelField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Recommendations from critic agent"
    )
    
    # Quality Indicators
    data_quality_score: float = SQLModelField(
        default=0.0,
        index=True,
        description="Overall data quality score"
    )
    missing_value_ratio: float = SQLModelField(
        default=0.0,
        description="Ratio of missing values"
    )
    
    # Agent Execution Trace
    steps: list[dict[str, Any]] = SQLModelField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Agent execution trace"
    )
    
    # Timestamps
    created_at: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
        description="Record creation timestamp"
    )
    updated_at: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record update timestamp"
    )
    
    # Retention
    ttl_seconds: Optional[int] = SQLModelField(
        default=None,
        description="Time-to-live in seconds"
    )
    
    # Relationships (for future expansion)
    # artifacts: list["ArtifactTable"] = Relationship(back_populates="execution")


class ArtifactTable(SQLModel, table=True):
    """
    SQLModel table for execution artifacts (models, reports, visualizations).
    """
    
    __tablename__ = "execution_artifacts"
    
    id: str = SQLModelField(primary_key=True)
    execution_id: str = SQLModelField(
        index=True,
        foreign_key="episodic_memory.id",
        description="Reference to execution"
    )
    artifact_type: str = SQLModelField(
        index=True,
        description="Type of artifact (model, report, plot, etc.)"
    )
    artifact_path: str = SQLModelField(
        description="File path or URI to artifact"
    )
    artifact_metadata: dict[str, Any] = SQLModelField(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Additional artifact metadata"
    )
    created_at: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Artifact creation timestamp"
    )
    
    # execution: ExecutionTable = Relationship(back_populates="artifacts")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class EpisodicMemoryConfig(BaseModel):
    """Configuration for the episodic memory backend."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Database connection
    database_url: str = "sqlite+aiosqlite:///./data/episodic.db"
    echo_sql: bool = False
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    
    # Retention policy
    retention_days: int = 365
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24
    
    # Indexing
    create_indexes: bool = True
    
    # Migrations
    auto_migrate: bool = True


# ---------------------------------------------------------------------------
# Migration Manager
# ---------------------------------------------------------------------------

class MigrationManager:
    """
    Handles database schema migrations for episodic memory.
    
    In production, this would integrate with Alembic. For now,
    it provides basic schema evolution capabilities.
    """
    
    def __init__(self, engine: Any) -> None:
        self.engine = engine
    
    async def run_migrations(self) -> None:
        """
        Applies pending migrations to bring the database to the latest schema.
        
        Migrations are applied in order:
        1. Create base tables
        2. Create indexes
        3. Apply data transformations (if needed)
        """
        logger.info("Running episodic memory migrations...")
        
        async with self.engine.begin() as conn:
            # Migration 001: Create base tables
            await conn.run_sync(self._create_tables)
            
            # Migration 002: Create indexes
            await conn.run_sync(self._create_indexes)
            
        logger.info("Episodic memory migrations completed successfully.")
    
    def _create_tables(self, conn: Any) -> None:
        """Creates all SQLModel tables."""
        SQLModel.metadata.create_all(conn)
    
    def _create_indexes(self, conn: Any) -> None:
        """Creates performance indexes on commonly queried columns."""
        from sqlalchemy import text
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_execution_dataset_hash ON episodic_memory(dataset_hash)",
            "CREATE INDEX IF NOT EXISTS idx_execution_outcome ON episodic_memory(outcome)",
            "CREATE INDEX IF NOT EXISTS idx_execution_task_type ON episodic_memory(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_execution_created_at ON episodic_memory(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_execution_quality ON episodic_memory(data_quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_artifact_execution ON execution_artifacts(execution_id)",
        ]
        
        for idx_sql in indexes:
            conn.execute(text(idx_sql))


# ---------------------------------------------------------------------------
# Episodic Memory Backend
# ---------------------------------------------------------------------------

class EpisodicMemory(BaseMemory):
    """
    Concrete implementation of BaseMemory for episodic (execution history) storage.
    
    Provides full CRUD operations, complex querying, retention management,
    and transaction support for storing DSA pipeline execution records.
    """
    
    def __init__(self, config: EpisodicMemoryConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._async_session: Any = None
        self._migration_manager: Optional[MigrationManager] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> None:
        """
        Establishes connection to the database and runs migrations.
        
        Creates the async engine and session factory, then applies
        any pending migrations to ensure schema compatibility.
        """
        try:
            # Ensure data directory exists
            if self.config.database_url.startswith("sqlite"):
                db_path = Path(self.config.database_url.replace("sqlite+aiosqlite:///", ""))
                db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create async engine
            self._engine = create_async_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
            )
            
            # Create session factory
            async_session_maker = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            self._async_session = async_session_maker
            
            # Run migrations
            if self.config.auto_migrate:
                self._migration_manager = MigrationManager(self._engine)
                await self._migration_manager.run_migrations()
            
            # Start cleanup task if enabled
            if self.config.auto_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"Episodic memory connected to {self.config.database_url}")
            
        except Exception as e:
            raise MemoryConnectionError(f"Failed to connect to episodic memory: {e}") from e
    
    async def disconnect(self) -> None:
        """
        Gracefully disconnects from the database.
        
        Cancels the cleanup task and disposes of the engine connection pool.
        """
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._engine:
            await self._engine.dispose()
        
        logger.info("Episodic memory disconnected.")
    
    async def health_check(self) -> dict[str, Any]:
        """
        Checks database connectivity and table accessibility.
        
        Returns:
            Dictionary with health status, table count, and latency.
        """
        start_time = datetime.now()
        try:
            async with self._async_session() as session:
                result = await session.exec(select(ExecutionTable).limit(1))
                _ = result.first()
            
            latency = (datetime.now() - start_time).total_seconds()
            return {
                "status": "healthy",
                "backend": "sqlmodel",
                "latency_seconds": latency,
                "database_url": self.config.database_url
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    async def store(self, record: EpisodicMemoryRecord) -> str:
        """
        Stores a single episodic memory record.
        
        Args:
            record: The episodic memory record to store.
            
        Returns:
            The ID of the stored record.
        """
        try:
            async with self._async_session() as session:
                # Convert schema to table model
                table_record = ExecutionTable(
                    id=record.metadata.memory_id,
                    execution_id=record.metadata.execution_id,
                    dataset_hash=record.metadata.dataset_hash,
                    dataset_name=record.metadata.dataset_name,
                    dataset_rows=record.dataset_rows,
                    dataset_columns=record.dataset_columns,
                    dataset_size_bytes=record.dataset_size_bytes,
                    task_type=record.task_type.value,
                    target_column=record.target_column,
                    suggested_models=record.suggested_models,
                    preprocessing_steps=record.preprocessing_steps,
                    planner_reasoning=record.planner_reasoning,
                    best_model=record.best_model,
                    metrics=record.metrics,
                    training_time_seconds=record.training_time_seconds,
                    outcome=record.outcome.value,
                    failure_reason=record.failure_reason,
                    critic_recommendations=record.critic_recommendations,
                    data_quality_score=record.quality.quality_score,
                    missing_value_ratio=record.missing_value_ratio,
                    steps=[step.model_dump() for step in record.steps],
                    created_at=record.metadata.created_at,
                    updated_at=datetime.now(timezone.utc),
                    ttl_seconds=record.metadata.ttl_seconds,
                )
                
                session.add(table_record)
                await session.commit()
                await session.refresh(table_record)
                
            logger.info(f"Stored episodic record {record.metadata.memory_id}")
            return record.metadata.memory_id
            
        except Exception as e:
            raise MemoryStorageError(f"Failed to store episodic record: {e}") from e
    
    async def store_batch(self, records: list[EpisodicMemoryRecord]) -> list[str]:
        """
        Stores multiple episodic memory records in a single transaction.
        
        Args:
            records: List of episodic memory records to store.
            
        Returns:
            List of stored record IDs.
        """
        if not records:
            return []
        
        try:
            async with self._async_session() as session:
                table_records = []
                for record in records:
                    table_record = ExecutionTable(
                        id=record.metadata.memory_id,
                        execution_id=record.metadata.execution_id,
                        dataset_hash=record.metadata.dataset_hash,
                        dataset_name=record.metadata.dataset_name,
                        dataset_rows=record.dataset_rows,
                        dataset_columns=record.dataset_columns,
                        dataset_size_bytes=record.dataset_size_bytes,
                        task_type=record.task_type.value,
                        target_column=record.target_column,
                        suggested_models=record.suggested_models,
                        preprocessing_steps=record.preprocessing_steps,
                        planner_reasoning=record.planner_reasoning,
                        best_model=record.best_model,
                        metrics=record.metrics,
                        training_time_seconds=record.training_time_seconds,
                        outcome=record.outcome.value,
                        failure_reason=record.failure_reason,
                        critic_recommendations=record.critic_recommendations,
                        data_quality_score=record.quality.quality_score,
                        missing_value_ratio=record.missing_value_ratio,
                        steps=[step.model_dump() for step in record.steps],
                        created_at=record.metadata.created_at,
                        updated_at=datetime.now(timezone.utc),
                        ttl_seconds=record.metadata.ttl_seconds,
                    )
                    table_records.append(table_record)
                
                session.add_all(table_records)
                await session.commit()
                
            logger.info(f"Stored {len(records)} episodic records in batch.")
            return [r.metadata.memory_id for r in records]
            
        except Exception as e:
            raise MemoryStorageError(f"Failed to store batch: {e}") from e
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieves episodic records matching the query and filters.
        
        Supports filtering by:
        - dataset_hash: Exact match
        - execution_id: Exact match
        - task_type: Exact match
        - outcome: Exact match
        - min_quality: Minimum quality score
        - date_from: Start date (inclusive)
        - date_to: End date (inclusive)
        
        Args:
            query: Search query (currently unused, reserved for future semantic search)
            top_k: Maximum number of results
            filters: Dictionary of filter criteria
            
        Returns:
            RetrievalResult with matching records
        """
        try:
            async with self._async_session() as session:
                stmt = select(ExecutionTable)
                
                # Apply filters
                if filters:
                    if "dataset_hash" in filters:
                        stmt = stmt.where(ExecutionTable.dataset_hash == filters["dataset_hash"])
                    if "execution_id" in filters:
                        stmt = stmt.where(ExecutionTable.execution_id == filters["execution_id"])
                    if "task_type" in filters:
                        stmt = stmt.where(ExecutionTable.task_type == filters["task_type"])
                    if "outcome" in filters:
                        stmt = stmt.where(ExecutionTable.outcome == filters["outcome"])
                    if "min_quality" in filters:
                        stmt = stmt.where(ExecutionTable.data_quality_score >= filters["min_quality"])
                    if "date_from" in filters:
                        stmt = stmt.where(ExecutionTable.created_at >= filters["date_from"])
                    if "date_to" in filters:
                        stmt = stmt.where(ExecutionTable.created_at <= filters["date_to"])
                
                # Order by recency and quality
                stmt = stmt.order_by(
                    ExecutionTable.data_quality_score.desc(),
                    ExecutionTable.created_at.desc()
                )
                
                # Limit results
                stmt = stmt.limit(top_k)
                
                result = await session.exec(stmt)
                records = result.all()
                
                # Convert to retrieval candidates
                candidates = []
                for record in records:
                    candidates.append(RetrievalCandidate(
                        memory_id=record.id,
                        similarity=1.0,  # Exact match, not semantic
                        quality=record.data_quality_score,
                        recency_score=0.0,  # Calculated by MemoryManager
                        combined_score=record.data_quality_score,
                        reasoning=f"Matched filters: {list(filters.keys()) if filters else 'none'}"
                    ))
                
                return RetrievalResult(
                    query=query,
                    retrieved_items=candidates,
                    execution_time=0.0,
                    strategy_used="sql_filter"
                )
                
        except Exception as e:
            raise MemoryRetrievalError(f"Failed to retrieve episodic records: {e}") from e
    
    async def retrieve_by_id(self, record_id: str) -> EpisodicMemoryRecord | None:
        """
        Retrieves a specific episodic record by ID.
        
        Args:
            record_id: The unique identifier of the record.
            
        Returns:
            The EpisodicMemoryRecord if found, None otherwise.
        """
        try:
            async with self._async_session() as session:
                result = await session.exec(
                    select(ExecutionTable).where(ExecutionTable.id == record_id)
                )
                record = result.first()
                
                if not record:
                    return None
                
                # Convert table model to schema
                return EpisodicMemoryRecord(
                    metadata=MemoryMetadata(
                        memory_id=record.id,
                        execution_id=record.execution_id,
                        dataset_hash=record.dataset_hash,
                        dataset_name=record.dataset_name,
                        created_at=record.created_at,
                        updated_at=record.updated_at,
                        memory_type=MemoryType.EPISODIC,
                        ttl_seconds=record.ttl_seconds,
                    ),
                    dataset_rows=record.dataset_rows,
                    dataset_columns=record.dataset_columns,
                    dataset_size_bytes=record.dataset_size_bytes,
                    task_type=record.task_type,
                    target_column=record.target_column,
                    suggested_models=record.suggested_models,
                    preprocessing_steps=record.preprocessing_steps,
                    planner_reasoning=record.planner_reasoning,
                    best_model=record.best_model,
                    metrics=record.metrics,
                    training_time_seconds=record.training_time_seconds,
                    outcome=record.outcome,
                    failure_reason=record.failure_reason,
                    critic_recommendations=record.critic_recommendations,
                    steps=[EpisodeStep(**step) for step in record.steps],
                    quality=MemoryQuality(
                        quality_score=record.data_quality_score,
                    ),
                    missing_value_ratio=record.missing_value_ratio,
                )
                
        except Exception as e:
            raise MemoryRetrievalError(f"Failed to retrieve record {record_id}: {e}") from e
    
    async def update(self, record: EpisodicMemoryRecord) -> bool:
        """
        Updates an existing episodic record.
        
        Args:
            record: The updated record.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        try:
            async with self._async_session() as session:
                result = await session.exec(
                    select(ExecutionTable).where(ExecutionTable.id == record.metadata.memory_id)
                )
                existing = result.first()
                
                if not existing:
                    raise MemoryNotFoundError(f"Record {record.metadata.memory_id} not found")
                
                # Update fields
                existing.dataset_rows = record.dataset_rows
                existing.dataset_columns = record.dataset_columns
                existing.dataset_size_bytes = record.dataset_size_bytes
                existing.task_type = record.task_type
                existing.target_column = record.target_column
                existing.suggested_models = record.suggested_models
                existing.preprocessing_steps = record.preprocessing_steps
                existing.planner_reasoning = record.planner_reasoning
                existing.best_model = record.best_model
                existing.metrics = record.metrics
                existing.training_time_seconds = record.training_time_seconds
                existing.outcome = record.outcome
                existing.failure_reason = record.failure_reason
                existing.critic_recommendations = record.critic_recommendations
                existing.data_quality_score = record.quality.quality_score
                existing.missing_value_ratio = record.missing_value_ratio
                existing.steps = [step.model_dump() for step in record.steps]
                existing.updated_at = datetime.now(timezone.utc)
                
                session.add(existing)
                await session.commit()
                
            logger.info(f"Updated episodic record {record.metadata.memory_id}")
            return True
            
        except MemoryNotFoundError:
            raise
        except Exception as e:
            raise MemoryStorageError(f"Failed to update record: {e}") from e
    
    async def delete(self, record_id: str) -> bool:
        """
        Deletes an episodic record by ID.
        
        Args:
            record_id: The ID of the record to delete.
            
        Returns:
            True if deletion succeeded, False otherwise.
        """
        try:
            async with self._async_session() as session:
                result = await session.exec(
                    select(ExecutionTable).where(ExecutionTable.id == record_id)
                )
                record = result.first()
                
                if not record:
                    return False
                
                await session.delete(record)
                await session.commit()
                
            logger.info(f"Deleted episodic record {record_id}")
            return True
            
        except Exception as e:
            raise MemoryStorageError(f"Failed to delete record {record_id}: {e}") from e
    
    async def clear(self) -> int:
        """
        Deletes all episodic records.
        
        Returns:
            The number of records deleted.
        """
        try:
            async with self._async_session() as session:
                count_result = await session.exec(select(ExecutionTable))
                count = len(count_result.all())
                
                await session.exec(delete(ExecutionTable))
                await session.commit()
                
            logger.info(f"Cleared {count} episodic records.")
            return count
            
        except Exception as e:
            raise MemoryStorageError(f"Failed to clear episodic memory: {e}") from e
    
    async def search(self, query: str, filters: dict[str, Any] | None = None) -> list[EpisodicMemoryRecord]:
        """
        Searches for episodic records matching the query and filters.
        
        This is a convenience method that returns full records instead of
        retrieval candidates.
        
        Args:
            query: Search query
            filters: Filter criteria
            
        Returns:
            List of matching EpisodicMemoryRecord objects
        """
        result = await self.retrieve(query, top_k=1000, filters=filters)
        records = []
        for candidate in result.retrieved_items:
            record = await self.retrieve_by_id(candidate.memory_id)
            if record:
                records.append(record)
        return records
    
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """
        Counts episodic records matching the filters.
        
        Args:
            filters: Optional filter criteria
            
        Returns:
            Total count of matching records
        """
        try:
            async with self._async_session() as session:
                stmt = select(ExecutionTable)
                
                if filters:
                    if "dataset_hash" in filters:
                        stmt = stmt.where(ExecutionTable.dataset_hash == filters["dataset_hash"])
                    if "task_type" in filters:
                        stmt = stmt.where(ExecutionTable.task_type == filters["task_type"])
                    if "outcome" in filters:
                        stmt = stmt.where(ExecutionTable.outcome == filters["outcome"])
                    if "date_from" in filters:
                        stmt = stmt.where(ExecutionTable.created_at >= filters["date_from"])
                    if "date_to" in filters:
                        stmt = stmt.where(ExecutionTable.created_at <= filters["date_to"])
                
                result = await session.exec(stmt)
                return len(result.all())
                
        except Exception as e:
            raise MemoryRetrievalError(f"Failed to count records: {e}") from e
    
    async def statistics(self) -> MemoryStatistics:
        """
        Computes aggregate statistics for the episodic memory.
        
        Returns:
            MemoryStatistics object with counts and averages
        """
        try:
            async with self._async_session() as session:
                # Total count
                total = await self.count()
                
                # Count by outcome
                success_count = await self.count({"outcome": "success"})
                failure_count = await self.count({"outcome": "failure"})
                
                # Average quality
                result = await session.exec(
                    select(ExecutionTable.data_quality_score)
                )
                scores = [r for r in result.all() if r > 0]
                avg_quality = sum(scores) / len(scores) if scores else 0.0
                
                return MemoryStatistics(
                    total_records=total,
                    semantic_records=0,
                    episodic_records=total,
                    procedural_records=0,
                    average_similarity=0.0,
                    average_quality=avg_quality,
                    database_size=0,  # Would need DB-specific query
                )
                
        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")
            return MemoryStatistics()
    
    async def backup(self, destination: str) -> str:
        """
        Creates a backup of the database.
        
        For SQLite, this copies the database file. For PostgreSQL,
        this would use pg_dump.
        
        Args:
            destination: Path for the backup file
            
        Returns:
            Path to the created backup
        """
        import shutil
        
        try:
            if self.config.database_url.startswith("sqlite"):
                db_path = Path(self.config.database_url.replace("sqlite+aiosqlite:///", ""))
                shutil.copy2(db_path, destination)
                logger.info(f"Backed up episodic memory to {destination}")
                return destination
            else:
                logger.warning("Backup for non-SQLite databases not implemented")
                return ""
                
        except Exception as e:
            raise MemoryBaseError(f"Backup failed: {e}") from e
    
    async def restore(self, source: str) -> bool:
        """
        Restores the database from a backup.
        
        Args:
            source: Path to the backup file
            
        Returns:
            True if restore succeeded
        """
        import shutil
        
        try:
            if self.config.database_url.startswith("sqlite"):
                db_path = Path(self.config.database_url.replace("sqlite+aiosqlite:///", ""))
                shutil.copy2(source, db_path)
                logger.info(f"Restored episodic memory from {source}")
                return True
            else:
                logger.warning("Restore for non-SQLite databases not implemented")
                return False
                
        except Exception as e:
            raise MemoryBaseError(f"Restore failed: {e}") from e
    
    async def optimize(self) -> dict[str, Any]:
        """
        Optimizes the database (VACUUM for SQLite).
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if self.config.database_url.startswith("sqlite"):
                async with self._async_session() as session:
                    await session.exec("VACUUM")
                    await session.commit()
                
                count = await self.count()
                return {"status": "optimized", "record_count": count, "operation": "VACUUM"}
            else:
                return {"status": "skipped", "reason": "Non-SQLite database"}
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def close(self) -> None:
        """
        Closes all connections and cleans up resources.
        """
        await self.disconnect()
        logger.info("Episodic memory backend closed.")
    
    # -----------------------------------------------------------------------
    # Retention Policy Management
    # -----------------------------------------------------------------------
    
    async def _cleanup_loop(self) -> None:
        """
        Background task that periodically cleans up expired records.
        
        Runs every cleanup_interval_hours and removes records that:
        1. Have exceeded their TTL
        2. Are older than retention_days
        """
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                await self._apply_retention_policy()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _apply_retention_policy(self) -> int:
        """
        Applies retention policy to remove old records.
        
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        
        try:
            async with self._async_session() as session:
                # Find expired records
                stmt = select(ExecutionTable).where(
                    (ExecutionTable.ttl_seconds.isnot(None)) &
                    (ExecutionTable.created_at < cutoff_date)
                )
                result = await session.exec(stmt)
                expired = result.all()
                
                if expired:
                    for record in expired:
                        await session.delete(record)
                    await session.commit()
                    logger.info(f"Deleted {len(expired)} expired records.")
                
                return len(expired)
                
        except Exception as e:
            logger.error(f"Failed to apply retention policy: {e}")
            return 0
    
    # -----------------------------------------------------------------------
    # Query Helpers
    # -----------------------------------------------------------------------
    
    async def get_by_dataset_hash(self, dataset_hash: str, limit: int = 10) -> list[EpisodicMemoryRecord]:
        """
        Retrieves execution history for a specific dataset.
        
        Args:
            dataset_hash: Dataset fingerprint
            limit: Maximum number of records
            
        Returns:
            List of execution records for the dataset
        """
        return await self.search(
            query="",
            filters={"dataset_hash": dataset_hash}
        )
    
    async def get_successful_executions(self, limit: int = 100) -> list[EpisodicMemoryRecord]:
        """
        Retrieves successful executions ordered by quality.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of successful execution records
        """
        return await self.search(
            query="",
            filters={"outcome": "success"}
        )
    
    async def get_high_quality_executions(
        self,
        min_quality: float = 0.8,
        limit: int = 50
    ) -> list[EpisodicMemoryRecord]:
        """
        Retrieves high-quality executions for learning.
        
        Args:
            min_quality: Minimum quality score threshold
            limit: Maximum number of records
            
        Returns:
            List of high-quality execution records
        """
        return await self.search(
            query="",
            filters={"min_quality": min_quality}
        )
