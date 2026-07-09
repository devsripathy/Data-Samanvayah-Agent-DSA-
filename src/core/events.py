"""
Event-driven architecture module for the Data Samanvayah Agent (DSA).

This module defines the core event models and the asynchronous EventBus 
used for decoupled communication between agents, observability integrations, 
and external systems.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Awaitable, Callable, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Base Models
# ---------------------------------------------------------------------------

class Severity(StrEnum):
    """Log and event severity levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseEvent(BaseModel):
    """Base model for all DSA events."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    payload: Any = Field(default_factory=dict)
    severity: Severity = Severity.INFO
    event_type: str = "base_event"


# ---------------------------------------------------------------------------
# Concrete Event Classes
# ---------------------------------------------------------------------------

class DatasetLoadedEvent(BaseEvent):
    """Fired when a dataset is successfully ingested and loaded into memory."""
    event_type: Literal["dataset_loaded"] = "dataset_loaded"


class MemoryRetrievedEvent(BaseEvent):
    """Fired when relevant historical context is retrieved from the vector store."""
    event_type: Literal["memory_retrieved"] = "memory_retrieved"


class QualityCompletedEvent(BaseEvent):
    """Fired when the Quality agent finishes initial data profiling."""
    event_type: Literal["quality_completed"] = "quality_completed"


class PlanningCompletedEvent(BaseEvent):
    """Fired when the Planner agent generates the execution strategy."""
    event_type: Literal["planning_completed"] = "planning_completed"


class EDACompletedEvent(BaseEvent):
    """Fired when the Explorer agent finishes EDA and preprocessing."""
    event_type: Literal["eda_completed"] = "eda_completed"


class TrainingStartedEvent(BaseEvent):
    """Fired when the Trainer agent begins the model training loop."""
    event_type: Literal["training_started"] = "training_started"


class TrainingCompletedEvent(BaseEvent):
    """Fired when the Trainer agent finishes training and evaluates models."""
    event_type: Literal["training_completed"] = "training_completed"


class CriticFinishedEvent(BaseEvent):
    """Fired when the Critic agent evaluates the training results."""
    event_type: Literal["critic_finished"] = "critic_finished"


class MemoryStoredEvent(BaseEvent):
    """Fired when execution context is saved to the vector store."""
    event_type: Literal["memory_stored"] = "memory_stored"


class WorkflowFinishedEvent(BaseEvent):
    """Fired when the entire DSA pipeline completes or fails."""
    event_type: Literal["workflow_finished"] = "workflow_finished"


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

# Type alias for event subscribers (supports both sync and async callbacks)
Subscriber = Callable[[BaseEvent], Awaitable[None] | None]


class EventBus:
    """
    Asynchronous event bus for publishing and subscribing to DSA events.
    
    Supports both synchronous and asynchronous subscriber callbacks, 
    maintains a history of published events, and ensures thread-safe 
    operations via asyncio locks.
    """
    
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._history: list[BaseEvent] = []
        self._lock = asyncio.Lock()
        
    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        """
        Registers a callback for a specific event type.
        
        Args:
            event_type: The string identifier of the event to subscribe to.
            callback: The async or sync function to execute when the event is published.
        """
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            
    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        """
        Removes a callback from a specific event type.
        
        Args:
            event_type: The string identifier of the event.
            callback: The function to remove.
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            
    async def publish(self, event: BaseEvent) -> None:
        """
        Publishes an event to all registered subscribers and records it in history.
        
        Args:
            event: The event instance to publish.
        """
        async with self._lock:
            self._history.append(event)
            
        event_type = event.event_type
        subscribers = self._subscribers.get(event_type, [])
        
        tasks = []
        for sub in subscribers:
            if inspect.iscoroutinefunction(sub):
                tasks.append(self._safe_execute(sub, event))
            else:
                # Run sync functions in a thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, sub, event))
                
        if tasks:
            await asyncio.gather(*tasks)
            
    async def _safe_execute(self, callback: Subscriber, event: BaseEvent) -> None:
        """Executes a subscriber callback and catches any exceptions."""
        try:
            await callback(event)  # type: ignore[misc]
        except Exception as e:
            logger.error(
                f"Error in event subscriber {callback.__name__} for "
                f"{event.event_type}: {e}", 
                exc_info=True
            )
            
    def history(self, event_type: Optional[str] = None) -> list[BaseEvent]:
        """
        Retrieves the history of published events.
        
        Args:
            event_type: Optional filter to retrieve only specific event types.
            
        Returns:
            A list of BaseEvent instances.
        """
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]


# Singleton instance for global access
event_bus = EventBus()
