"""
Procedural Memory Backend for the Data Samanvayah Agent (DSA).

This module implements the procedural memory layer, responsible for storing,
evaluating, and retrieving learned IF-THEN rules and heuristics. It allows
the Planner and Supervisor agents to make informed decisions based on historical
success patterns without needing to know the underlying storage implementation.

Features:
- IF-THEN rule storage with complex condition matching.
- Confidence tracking and decay over time (forgetting curve).
- Conflict resolution and rule prioritization via composite scoring.
- Activation and success history tracking.
- High-level recommendation APIs for downstream agents.
- Thread-safe in-memory store with JSON persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    MemoryQuality,
    MemoryStatistics,
    MemoryType,
    ProceduralRule,
    RetrievalCandidate,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ProceduralMemoryConfig(BaseModel):
    """Configuration for the procedural memory backend."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Persistence
    persistence_path: str = "./data/procedural_memory.json"
    auto_save_interval_seconds: int = 300
    
    # Rule Management
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_rules_per_context: int = Field(default=5, gt=0)
    
    # Decay & Scoring Weights
    decay_rate: float = Field(default=0.05, ge=0.0)  # Confidence decay per day
    weight_confidence: float = 0.40
    weight_success_rate: float = 0.30
    weight_recency: float = 0.15
    weight_activation: float = 0.15


# ---------------------------------------------------------------------------
# Rule Matching & Scoring Engine
# ---------------------------------------------------------------------------

class RuleMatcher:
    """Utility class for evaluating conditions and scoring rules."""
    
    @staticmethod
    def evaluate_conditions(rule: ProceduralRule, context: dict[str, Any]) -> bool:
        """
        Checks if a rule's conditions are satisfied by the given context.
        
        Context keys should match rule condition keys.
        Condition format: {"feature_name_operator": threshold_value}
        Supported operators: _lt, _gt, _lte, _gte, _eq, _neq, _in, _not_in
        """
        for condition_key, threshold in rule.conditions.items():
            # Parse operator
            if "_" in condition_key:
                parts = condition_key.rsplit("_", 1)
                feature_name = parts[0]
                operator = parts[1]
            else:
                feature_name = condition_key
                operator = "eq"
            
            if feature_name not in context:
                return False
                
            value = context[feature_name]
            
            if operator == "lt" and not (value < threshold): return False
            elif operator == "gt" and not (value > threshold): return False
            elif operator == "lte" and not (value <= threshold): return False
            elif operator == "gte" and not (value >= threshold): return False
            elif operator == "eq" and not (value == threshold): return False
            elif operator == "neq" and not (value != threshold): return False
            elif operator == "in" and not (value in threshold): return False
            elif operator == "not_in" and not (value not in threshold): return False
            
        return True

    @staticmethod
    def calculate_composite_score(rule: ProceduralRule, config: ProceduralMemoryConfig) -> float:
        """
        Calculates a composite score for rule prioritization.
        
        Score = (confidence * w1) + (success_rate * w2) + (recency * w3) + (activation * w4)
        """
        # 1. Confidence
        conf_score = rule.confidence
        
        # 2. Success Rate
        success_score = rule.success_rate
        
        # 3. Recency (Exponential decay based on days since last validated)
        if rule.last_validated:
            days_since = (datetime.now(timezone.utc) - rule.last_validated).total_seconds() / 86400
            recency_score = math.exp(-config.decay_rate * days_since)
        else:
            recency_score = 0.5  # Default for new rules
            
        # 4. Activation (Logarithmic scaling to prevent highly used rules from dominating)
        activation_score = min(1.0, math.log1p(rule.times_applied) / 10.0)
        
        return (
            (conf_score * config.weight_confidence) +
            (success_score * config.weight_success_rate) +
            (recency_score * config.weight_recency) +
            (activation_score * config.weight_activation)
        )


# ---------------------------------------------------------------------------
# Procedural Memory Backend
# ---------------------------------------------------------------------------

class ProceduralMemory(BaseMemory):
    """
    Concrete implementation of BaseMemory for procedural (rule-based) storage.
    
    Provides an in-memory, thread-safe rule engine with JSON persistence.
    It exposes high-level recommendation APIs for the Planner and Supervisor.
    """
    
    def __init__(self, config: ProceduralMemoryConfig) -> None:
        self.config = config
        self._store: dict[str, ProceduralRule] = {}
        self._lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> None:
        """Loads rules from the persistence file into memory."""
        try:
            path = Path(self.config.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if path.exists():
                async with self._lock:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._store = {
                            k: ProceduralRule.model_validate(v) 
                            for k, v in data.items()
                        }
                logger.info(f"Loaded {len(self._store)} procedural rules from {path}")
            else:
                logger.info("No existing procedural memory file found. Starting fresh.")
                
            # Start auto-save loop
            self._save_task = asyncio.create_task(self._auto_save_loop())
            
        except Exception as e:
            raise MemoryConnectionError(f"Failed to load procedural memory: {e}") from e

    async def disconnect(self) -> None:
        """Saves current state and stops background tasks."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        
        await self._persist_to_disk()
        logger.info("Procedural memory disconnected and saved.")

    async def health_check(self) -> dict[str, Any]:
        """Checks if the memory is loaded and accessible."""
        return {
            "status": "healthy",
            "backend": "in_memory_json",
            "rule_count": len(self._store),
            "persistence_path": self.config.persistence_path
        }

    async def store(self, record: ProceduralRule) -> str:
        """Stores or updates a single procedural rule."""
        async with self._lock:
            # Apply decay before storing new rules to keep stats fresh
            self._apply_decay(record)
            self._store[record.id] = record
            logger.info(f"Stored procedural rule: {record.rule_name}")
        return record.id

    async def store_batch(self, records: list[ProceduralRule]) -> list[str]:
        """Stores multiple rules in a single transaction."""
        async with self._lock:
            ids = []
            for record in records:
                self._apply_decay(record)
                self._store[record.id] = record
                ids.append(record.id)
            logger.info(f"Stored {len(records)} procedural rules.")
        return ids

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieves rules. If query is a JSON string, it treats it as a context for matching.
        Otherwise, it filters by rule_name or tags.
        """
        try:
            # Check if query is a JSON context for rule matching
            context = None
            try:
                context = json.loads(query)
            except json.JSONDecodeError:
                pass

            async with self._lock:
                candidates = []
                for rule in self._store.values():
                    if not rule.is_active:
                        continue
                        
                    score = 0.0
                    reasoning = ""
                    
                    if context and RuleMatcher.evaluate_conditions(rule, context):
                        score = RuleMatcher.calculate_composite_score(rule, self.config)
                        reasoning = f"Matched context. Composite score: {score:.3f}"
                    elif not context:
                        # Text search on name or description
                        if query.lower() in rule.rule_name.lower() or query.lower() in rule.description.lower():
                            score = rule.confidence
                            reasoning = "Text match on name/description."
                    
                    if score > 0:
                        candidates.append(RetrievalCandidate(
                            memory_id=rule.id,
                            similarity=rule.confidence,
                            quality=rule.success_rate,
                            recency_score=1.0, # Simplified for procedural
                            combined_score=score,
                            reasoning=reasoning
                        ))
                
                # Sort by composite score
                candidates.sort(key=lambda x: x.combined_score, reverse=True)
                candidates = candidates[:top_k]
                
            return RetrievalResult(
                query=query,
                retrieved_items=candidates,
                execution_time=0.0,
                strategy_used="rule_matching" if context else "text_search"
            )
        except Exception as e:
            raise MemoryRetrievalError(f"Failed to retrieve rules: {e}") from e

    async def retrieve_by_id(self, record_id: str) -> ProceduralRule | None:
        """Retrieves a specific rule by ID."""
        async with self._lock:
            return self._store.get(record_id)

    async def update(self, record: ProceduralRule) -> bool:
        """Updates an existing rule."""
        async with self._lock:
            if record.id not in self._store:
                raise MemoryNotFoundError(f"Rule {record.id} not found.")
            self._store[record.id] = record
            return True

    async def delete(self, record_id: str) -> bool:
        """Deletes a rule by ID."""
        async with self._lock:
            if record_id in self._store:
                del self._store[record_id]
                return True
            return False

    async def clear(self) -> int:
        """Clears all rules."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    async def search(self, query: str, filters: dict[str, Any] | None = None) -> list[ProceduralRule]:
        """Searches for rules matching the query."""
        result = await self.retrieve(query, top_k=1000, filters=filters)
        rules = []
        for candidate in result.retrieved_items:
            rule = await self.retrieve_by_id(candidate.memory_id)
            if rule:
                rules.append(rule)
        return rules

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Returns the total number of active rules."""
        async with self._lock:
            return len([r for r in self._store.values() if r.is_active])

    async def statistics(self) -> MemoryStatistics:
        """Computes statistics for the procedural memory."""
        async with self._lock:
            rules = list(self._store.values())
            total = len(rules)
            avg_conf = sum(r.confidence for r in rules) / total if total > 0 else 0.0
            avg_success = sum(r.success_rate for r in rules) / total if total > 0 else 0.0
            
            return MemoryStatistics(
                total_records=total,
                semantic_records=0,
                episodic_records=0,
                procedural_records=total,
                average_similarity=0.0,
                average_quality=avg_success,
                database_size=0
            )

    async def backup(self, destination: str) -> str:
        """Backs up the JSON file."""
        import shutil
        try:
            shutil.copy2(self.config.persistence_path, destination)
            return destination
        except Exception as e:
            raise MemoryBaseError(f"Backup failed: {e}") from e

    async def restore(self, source: str) -> bool:
        """Restores from a JSON backup."""
        import shutil
        try:
            shutil.copy2(source, self.config.persistence_path)
            await self.connect() # Reload
            return True
        except Exception as e:
            raise MemoryBaseError(f"Restore failed: {e}") from e

    async def optimize(self) -> dict[str, Any]:
        """Removes inactive or low-confidence rules."""
        async with self._lock:
            initial_count = len(self._store)
            self._store = {
                k: v for k, v in self._store.items() 
                if v.is_active and v.confidence >= self.config.min_confidence_threshold
            }
            removed = initial_count - len(self._store)
            return {"status": "optimized", "rules_removed": removed}

    async def close(self) -> None:
        """Closes the backend."""
        await self.disconnect()

    # -----------------------------------------------------------------------
    # High-Level Recommendation APIs (For Planner & Supervisor)
    # -----------------------------------------------------------------------

    async def get_recommendations(
        self,
        context: dict[str, Any],
        top_k: int | None = None
    ) -> list[ProceduralRule]:
        """
        Returns the best matching rules for a given dataset context.
        
        This is the primary API consumed by the Planner and Supervisor.
        It handles condition matching, scoring, and conflict resolution.
        
        Args:
            context: Dictionary of dataset features (e.g., {"rows": 1000, "task_type": "classification"}).
            top_k: Maximum number of rules to return. Defaults to config.max_rules_per_context.
            
        Returns:
            List of ProceduralRule objects sorted by composite score.
        """
        if top_k is None:
            top_k = self.config.max_rules_per_context
            
        result = await self.retrieve(query=json.dumps(context), top_k=top_k)
        
        rules = []
        for candidate in result.retrieved_items:
            rule = await self.retrieve_by_id(candidate.memory_id)
            if rule:
                rules.append(rule)
                
        return rules

    async def record_rule_outcome(
        self,
        rule_id: str,
        success: bool,
        execution_id: str | None = None
    ) -> None:
        """
        Updates a rule's success history and confidence based on pipeline outcome.
        
        Args:
            rule_id: The ID of the rule that was applied.
            success: Whether the application of the rule led to a successful outcome.
            execution_id: Optional ID of the execution for tracing.
        """
        async with self._lock:
            if rule_id not in self._store:
                logger.warning(f"Attempted to update outcome for unknown rule: {rule_id}")
                return
                
            rule = self._store[rule_id]
            rule.times_applied += 1
            rule.last_validated = datetime.now(timezone.utc)
            
            if success:
                rule.success_count += 1
                # Boost confidence slightly
                rule.confidence = min(1.0, rule.confidence + 0.05)
            else:
                rule.failure_count += 1
                # Penalize confidence
                rule.confidence = max(0.0, rule.confidence - 0.10)
                
            # Deactivate if confidence drops too low
            if rule.confidence < self.config.min_confidence_threshold:
                rule.is_active = False
                logger.info(f"Rule {rule.rule_name} deactivated due to low confidence ({rule.confidence:.2f})")
                
            logger.info(f"Updated outcome for rule {rule.rule_name}: success={success}, new_conf={rule.confidence:.2f}")

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _apply_decay(self, rule: ProceduralRule) -> None:
        """Applies time-based decay to a rule's confidence."""
        if rule.last_validated:
            days_since = (datetime.now(timezone.utc) - rule.last_validated).total_seconds() / 86400
            if days_since > 0:
                decay_amount = self.config.decay_rate * days_since
                rule.confidence = max(0.0, rule.confidence - decay_amount)

    async def _auto_save_loop(self) -> None:
        """Periodically saves the in-memory store to disk."""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval_seconds)
                await self._persist_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    async def _persist_to_disk(self) -> None:
        """Writes the current in-memory store to the JSON file."""
        async with self._lock:
            try:
                data = {k: v.model_dump(mode="json") for k, v in self._store.items()}
                path = Path(self.config.persistence_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temp file first to prevent corruption
                temp_path = path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                temp_path.replace(path)
                
            except Exception as e:
                logger.error(f"Failed to persist procedural memory: {e}")
