"""
Memory Retrieval Engine for the Data Samanvayah Agent (DSA).

This module implements the unified memory retrieval engine, responsible for 
querying semantic, episodic, and procedural memory backends, merging their 
results, and applying a configurable weighted ranking algorithm. 

It produces a single, unified list of ranked memories with human-readable 
explanations for why each memory was selected, enabling the Planner and 
Supervisor agents to make highly informed decisions.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.memory.base import BaseMemory
from src.memory.schemas import (
    EpisodicMemoryRecord,
    MemoryType,
    ProceduralRule,
    RetrievalCandidate,
    RetrievalResult,
    SemanticMemoryRecord,
    TaskType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval Context & Configuration
# ---------------------------------------------------------------------------

class RetrievalContext(BaseModel):
    """
    Contextual features of the current execution used to evaluate memory relevance.
    
    This object is passed to the retrieval engine to filter and rank memories
    based on how well they match the current dataset and task characteristics.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query_text: str = Field(..., min_length=1, description="Natural language query or dataset summary.")
    task_type: TaskType = Field(default=TaskType.UNKNOWN, description="The ML task type for the current run.")
    dataset_rows: int = Field(default=0, ge=0, description="Number of rows in the current dataset.")
    dataset_columns: int = Field(default=0, ge=0, description="Number of columns in the current dataset.")
    missing_value_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of missing values.")
    target_column_type: Optional[str] = Field(default=None, description="Inferred type of the target column.")
    execution_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the current execution for recency calculations."
    )
    custom_features: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional features for custom scoring strategies."
    )


class ScoringWeights(BaseModel):
    """
    Configurable weights for the memory ranking algorithm.
    
    All weights must sum to 1.0. These determine the relative importance 
    of each scoring dimension during the retrieval merge phase.
    """
    
    model_config = ConfigDict(frozen=True)
    
    semantic_similarity: float = Field(default=0.30, ge=0.0, le=1.0)
    historical_quality: float = Field(default=0.25, ge=0.0, le=1.0)
    recency: float = Field(default=0.15, ge=0.0, le=1.0)
    task_compatibility: float = Field(default=0.15, ge=0.0, le=1.0)
    dataset_similarity: float = Field(default=0.15, ge=0.0, le=1.0)
    
    @model_validator(mode="after")
    def _validate_weights_sum(self) -> "ScoringWeights":
        """Ensures all weights sum exactly to 1.0."""
        total = (
            self.semantic_similarity + 
            self.historical_quality + 
            self.recency + 
            self.task_compatibility + 
            self.dataset_similarity
        )
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, but got {total:.4f}")
        return self


# ---------------------------------------------------------------------------
# Unified Retrieval Schemas
# ---------------------------------------------------------------------------

class UnifiedRetrievalCandidate(BaseModel):
    """
    A single memory record that has been scored, ranked, and merged 
    from the different memory backends.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_id: str = Field(..., description="Unique identifier of the memory record.")
    memory_type: MemoryType = Field(..., description="Type of memory (semantic, episodic, procedural).")
    record: Any = Field(..., description="The actual memory record object (Semantic, Episodic, or Procedural).")
    
    # Dimension Scores (0.0 to 1.0)
    semantic_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    historical_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_compatibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dataset_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Final Score
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Explanation
    explanation: str = Field(default="", description="Human-readable explanation of why this memory was selected.")


class UnifiedRetrievalResult(BaseModel):
    """
    The final output of the Memory Retrieval Engine, containing 
    all ranked and scored memory candidates.
    """
    
    context: RetrievalContext
    candidates: list[UnifiedRetrievalCandidate] = Field(default_factory=list)
    total_retrieved: int = Field(default=0)
    retrieval_time_ms: float = Field(default=0.0)
    strategy_used: str = Field(default="weighted_sum")


# ---------------------------------------------------------------------------
# Scoring Strategy Interface & Implementations
# ---------------------------------------------------------------------------

class BaseScoringStrategy(ABC):
    """Abstract base class for memory scoring strategies."""
    
    @abstractmethod
    def calculate_final_score(self, candidate: UnifiedRetrievalCandidate, weights: ScoringWeights) -> float:
        """Calculates the final weighted score for a candidate."""
        pass
    
    @abstractmethod
    def generate_explanation(self, candidate: UnifiedRetrievalCandidate, weights: ScoringWeights) -> str:
        """Generates a human-readable explanation for the candidate's ranking."""
        pass


class WeightedSumScoringStrategy(BaseScoringStrategy):
    """
    Default scoring strategy using a linear weighted sum of dimension scores.
    
    This strategy is highly interpretable and allows fine-grained control 
    over the retrieval behavior via the ScoringWeights configuration.
    """
    
    def calculate_final_score(self, candidate: UnifiedRetrievalCandidate, weights: ScoringWeights) -> float:
        return (
            (candidate.semantic_similarity_score * weights.semantic_similarity) +
            (candidate.historical_quality_score * weights.historical_quality) +
            (candidate.recency_score * weights.recency) +
            (candidate.task_compatibility_score * weights.task_compatibility) +
            (candidate.dataset_similarity_score * weights.dataset_similarity)
        )
    
    def generate_explanation(self, candidate: UnifiedRetrievalCandidate, weights: ScoringWeights) -> str:
        parts = [f"Memory Type: {candidate.memory_type.value}"]
        
        if candidate.semantic_similarity_score > 0:
            parts.append(f"Semantic Similarity: {candidate.semantic_similarity_score:.2f} (Weight: {weights.semantic_similarity})")
        if candidate.historical_quality_score > 0:
            parts.append(f"Historical Quality: {candidate.historical_quality_score:.2f} (Weight: {weights.historical_quality})")
        if candidate.recency_score > 0:
            parts.append(f"Recency: {candidate.recency_score:.2f} (Weight: {weights.recency})")
        if candidate.task_compatibility_score > 0:
            parts.append(f"Task Match: {candidate.task_compatibility_score:.2f} (Weight: {weights.task_compatibility})")
        if candidate.dataset_similarity_score > 0:
            parts.append(f"Dataset Similarity: {candidate.dataset_similarity_score:.2f} (Weight: {weights.dataset_similarity})")
            
        parts.append(f"Final Score: {candidate.final_score:.3f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Memory Retrieval Engine
# ---------------------------------------------------------------------------

class MemoryRetrievalEngine:
    """
    Central engine for querying, merging, and ranking memories across all backends.
    
    This engine abstracts the complexity of querying multiple heterogeneous 
    memory stores (vector DBs, relational DBs, JSON rule stores) and presents 
    a unified, ranked interface to the Planner and Supervisor agents.
    """
    
    def __init__(
        self,
        semantic_memory: BaseMemory,
        episodic_memory: BaseMemory,
        procedural_memory: BaseMemory,
        weights: ScoringWeights | None = None,
        scoring_strategy: BaseScoringStrategy | None = None,
        top_k: int = 10,
    ) -> None:
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        
        self.weights = weights or ScoringWeights()
        self.scoring_strategy = scoring_strategy or WeightedSumScoringStrategy()
        self.top_k = top_k
        
        logger.info("MemoryRetrievalEngine initialized.")

    async def retrieve(self, context: RetrievalContext) -> UnifiedRetrievalResult:
        """
        Executes the full retrieval pipeline: query, normalize, score, rank, and explain.
        
        Args:
            context: The contextual features of the current execution.
            
        Returns:
            A UnifiedRetrievalResult containing ranked candidates.
        """
        import time
        start_time = time.perf_counter()
        
        logger.info(f"Starting memory retrieval for task: {context.task_type}")
        
        # 1. Query all backends concurrently
        semantic_results, episodic_results, procedural_results = await self._query_backends(context)
        
        # 2. Convert backend-specific results into UnifiedRetrievalCandidates
        raw_candidates = await self._convert_to_unified_candidates(
            context, semantic_results, episodic_results, procedural_results
        )
        
        # 3. Normalize scores across all candidates to a 0.0 - 1.0 scale
        normalized_candidates = self._normalize_scores(raw_candidates)
        
        # 4. Calculate final scores and generate explanations
        ranked_candidates = self._score_and_rank(normalized_candidates)
        
        # 5. Truncate to top_k
        final_candidates = ranked_candidates[:self.top_k]
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Retrieval complete. Returned {len(final_candidates)} candidates in {duration_ms:.2f}ms.")
        
        return UnifiedRetrievalResult(
            context=context,
            candidates=final_candidates,
            total_retrieved=len(raw_candidates),
            retrieval_time_ms=duration_ms,
            strategy_used=self.scoring_strategy.__class__.__name__
        )

    async def _query_backends(
        self, context: RetrievalContext
    ) -> tuple[RetrievalResult, RetrievalResult, RetrievalResult]:
        """Queries all three memory backends concurrently."""
        import asyncio
        
        # Construct backend-specific queries/filters
        semantic_query = context.query_text
        episodic_filters = {
            "task_type": context.task_type.value,
            "min_quality": 0.5  # Only retrieve decent historical runs
        }
        procedural_query = {
            "task_type": context.task_type.value,
            "rows": context.dataset_rows,
            "columns": context.dataset_columns
        }
        
        # Execute queries concurrently
        try:
            semantic_res, episodic_res, procedural_res = await asyncio.gather(
                self.semantic_memory.retrieve(query=semantic_query, top_k=self.top_k * 2),
                self.episodic_memory.retrieve(query="", top_k=self.top_k * 2, filters=episodic_filters),
                self.procedural_memory.retrieve(query=str(procedural_query), top_k=self.top_k * 2),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Failed to query memory backends: {e}")
            # Fallback to empty results if backends fail
            semantic_res, episodic_res, procedural_res = RetrievalResult(), RetrievalResult(), RetrievalResult()
            
        # Handle exceptions from gather
        if isinstance(semantic_res, Exception): semantic_res = RetrievalResult()
        if isinstance(episodic_res, Exception): episodic_res = RetrievalResult()
        if isinstance(procedural_res, Exception): procedural_res = RetrievalResult()
        
        return semantic_res, episodic_res, procedural_res

    async def _convert_to_unified_candidates(
        self,
        context: RetrievalContext,
        semantic_res: RetrievalResult,
        episodic_res: RetrievalResult,
        procedural_res: RetrievalResult,
    ) -> list[UnifiedRetrievalCandidate]:
        """Maps backend-specific RetrievalCandidates to UnifiedRetrievalCandidates."""
        candidates = []
        
        # Process Semantic
        for item in semantic_res.retrieved_items:
            record = await self.semantic_memory.retrieve_by_id(item.memory_id)
            if record:
                candidates.append(UnifiedRetrievalCandidate(
                    memory_id=item.memory_id,
                    memory_type=MemoryType.SEMANTIC,
                    record=record,
                    semantic_similarity_score=item.similarity,
                    historical_quality_score=record.quality.quality_score if hasattr(record, 'quality') else 0.0,
                ))
                
        # Process Episodic
        for item in episodic_res.retrieved_items:
            record = await self.episodic_memory.retrieve_by_id(item.memory_id)
            if record:
                candidates.append(UnifiedRetrievalCandidate(
                    memory_id=item.memory_id,
                    memory_type=MemoryType.EPISODIC,
                    record=record,
                    semantic_similarity_score=0.0,  # Episodic is filtered, not semantic
                    historical_quality_score=item.quality,
                ))
                
        # Process Procedural
        for item in procedural_res.retrieved_items:
            record = await self.procedural_memory.retrieve_by_id(item.memory_id)
            if record:
                candidates.append(UnifiedRetrievalCandidate(
                    memory_id=item.memory_id,
                    memory_type=MemoryType.PROCEDURAL,
                    record=record,
                    semantic_similarity_score=0.0,
                    historical_quality_score=record.confidence if hasattr(record, 'confidence') else item.quality,
                ))
                
        return candidates

    def _normalize_scores(self, candidates: list[UnifiedRetrievalCandidate]) -> list[UnifiedRetrievalCandidate]:
        """
        Normalizes dimension scores across all candidates to a 0.0 - 1.0 scale.
        
        Uses Min-Max normalization. If all values are identical, defaults to 0.5 
        to prevent zero-division and maintain relative neutrality.
        """
        if not candidates:
            return candidates
            
        dimensions = [
            "semantic_similarity_score",
            "historical_quality_score",
            "task_compatibility_score",
            "dataset_similarity_score"
        ]
        
        for dim in dimensions:
            values = [getattr(c, dim) for c in candidates]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            
            for candidate in candidates:
                current_val = getattr(candidate, dim)
                if range_val > 1e-6:
                    normalized = (current_val - min_val) / range_val
                else:
                    normalized = 0.5 if current_val > 0 else 0.0
                setattr(candidate, dim, normalized)
                
        # Calculate Recency Score specifically
        now = datetime.now(timezone.utc)
        for candidate in candidates:
            record = candidate.record
            created_at = getattr(record.metadata, 'created_at', None) if hasattr(record, 'metadata') else None
            
            if created_at:
                # Exponential decay: score drops by 50% every 30 days
                days_old = (now - created_at).total_seconds() / 86400
                candidate.recency_score = math.exp(-0.023 * days_old)  # ~0.5 at 30 days
            else:
                candidate.recency_score = 0.0
                
            # Calculate Task Compatibility
            if hasattr(record, 'task_type') and record.task_type == candidate.record.task_type if hasattr(candidate.record, 'task_type') else False:
                candidate.task_compatibility_score = 1.0
            elif hasattr(record, 'task_type') and record.task_type == TaskType.UNKNOWN:
                candidate.task_compatibility_score = 0.5
            else:
                candidate.task_compatibility_score = 0.0
                
            # Calculate Dataset Similarity (Simple heuristic based on row count ratio)
            # This is a placeholder for a more complex schema overlap calculation
            candidate.dataset_similarity_score = 0.5  # Default neutral
            
        return candidates

    def _score_and_rank(self, candidates: list[UnifiedRetrievalCandidate]) -> list[UnifiedRetrievalCandidate]:
        """Calculates final scores, generates explanations, and sorts candidates."""
        for candidate in candidates:
            candidate.final_score = self.scoring_strategy.calculate_final_score(candidate, self.weights)
            candidate.explanation = self.scoring_strategy.generate_explanation(candidate, self.weights)
            
        # Sort descending by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates
