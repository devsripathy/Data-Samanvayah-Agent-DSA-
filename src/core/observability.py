"""
LangSmith and observability integration for the Data Samanvayah Agent.

This module provides tracing context managers, trace initialization, and 
observability utilities for monitoring agent decision-making and execution flow.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from src.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangSmith Configuration
# ---------------------------------------------------------------------------

def configure_langsmith() -> bool:
    """
    Initializes LangSmith tracing based on configuration settings.
    
    Returns:
        True if successfully configured, False otherwise.
    """
    settings = get_settings()
    
    if not settings.observability.langsmith_enabled:
        logger.info("LangSmith tracing is disabled.")
        return False
    
    # Set environment variables for LangSmith
    if settings.observability.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.observability.langsmith_api_key.get_secret_value()
    
    if settings.observability.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.observability.langsmith_project
    
    # Enable tracing
    os.environ["LANGSMITH_TRACING"] = "true"
    
    logger.info(f"LangSmith configured with project: {settings.observability.langsmith_project}")
    return True


# ---------------------------------------------------------------------------
# Tracing Context Managers
# ---------------------------------------------------------------------------

@contextmanager
def trace_agent_execution(
    agent_name: str,
    session_id: str,
    thread_id: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for tracing agent execution with LangSmith.
    
    Creates a named trace for each agent node execution, automatically
    capturing timing, metadata, and any raised exceptions.
    
    Example:
        with trace_agent_execution("planner", session_id, thread_id) as trace_ctx:
            trace_ctx["input"] = state
            # ... agent logic ...
            trace_ctx["output"] = result
    
    Args:
        agent_name: Name of the agent being executed (e.g., "supervisor", "critic").
        session_id: Unique session identifier for grouping traces.
        thread_id: Thread identifier for execution continuity.
        metadata: Optional metadata to attach to the trace.
        
    Yields:
        A context dictionary for storing trace inputs/outputs.
    """
    settings = get_settings()
    if not settings.observability.langsmith_enabled:
        # Return a no-op context if tracing is disabled
        yield {"input": None, "output": None}
        return
    
    try:
        from langsmith import trace
    except ImportError:
        logger.warning("langsmith not installed. Tracing disabled.")
        yield {"input": None, "output": None}
        return
    
    trace_context = {
        "input": None,
        "output": None,
        "agent": agent_name,
        "session_id": session_id,
        "thread_id": thread_id,
        "metadata": metadata or {},
    }
    
    trace_name = f"agent_{agent_name}_{session_id[:8]}"
    
    @trace(trace_name, run_type="agent")
    def traced_execution():
        return trace_context
    
    try:
        traced_execution()
        yield trace_context
        logger.debug(f"Successfully traced agent execution: {agent_name}")
    except Exception as e:
        logger.error(f"Error during agent execution: {agent_name} - {str(e)}")
        trace_context["error"] = str(e)
        raise


@contextmanager
def trace_decision_point(
    decision_name: str,
    session_id: str,
    decision_data: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for tracing critical decision points (e.g., routing, approval).
    
    Example:
        with trace_decision_point("critic_approval", session_id, decision_data) as ctx:
            ctx["decision"] = "approved"
            ctx["reasoning"] = "Accuracy > 90%"
    
    Args:
        decision_name: Human-readable name of the decision being made.
        session_id: Session identifier.
        decision_data: Optional input data for the decision.
        
    Yields:
        A context dictionary for storing decision details.
    """
    settings = get_settings()
    if not settings.observability.langsmith_enabled:
        yield {"decision": None, "reasoning": None}
        return
    
    try:
        from langsmith import trace
    except ImportError:
        yield {"decision": None, "reasoning": None}
        return
    
    decision_context = {
        "decision": None,
        "reasoning": None,
        "input_data": decision_data or {},
        "session_id": session_id,
    }
    
    @trace(f"decision_{decision_name}", run_type="chain")
    def traced_decision():
        return decision_context
    
    try:
        traced_decision()
        yield decision_context
        logger.debug(f"Traced decision point: {decision_name}")
    except Exception as e:
        logger.error(f"Error during decision point: {decision_name} - {str(e)}")
        decision_context["error"] = str(e)
        raise


# ---------------------------------------------------------------------------
# LangSmith Metrics & Annotations
# ---------------------------------------------------------------------------

def log_trace_event(
    event_type: str,
    agent_name: str,
    session_id: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Logs a custom event to the trace for enhanced observability.
    
    Args:
        event_type: Type of event (e.g., "agent_start", "agent_complete", "retry").
        agent_name: Name of the agent (if applicable).
        session_id: Session identifier.
        details: Optional additional details.
    """
    settings = get_settings()
    if not settings.observability.langsmith_enabled:
        return
    
    try:
        from langsmith.client import Client
        
        client = Client()
        event_info = {
            "event_type": event_type,
            "agent": agent_name,
            "session_id": session_id,
            **(details or {}),
        }
        
        logger.info(f"LangSmith Event: {event_info}")
        # Events are automatically captured in the trace context
        
    except ImportError:
        logger.debug("langsmith not available for event logging")
    except Exception as e:
        logger.warning(f"Failed to log trace event: {e}")


def annotate_trace_metadata(
    session_id: str,
    thread_id: str,
    key: str,
    value: Any,
) -> None:
    """
    Adds custom metadata to the current trace for better observability.
    
    Args:
        session_id: Session identifier.
        thread_id: Thread identifier.
        key: Metadata key.
        value: Metadata value.
    """
    settings = get_settings()
    if not settings.observability.langsmith_enabled:
        return
    
    try:
        from langsmith import get_run_tree
        
        run = get_run_tree()
        if run and hasattr(run, 'metadata'):
            if not isinstance(run.metadata, dict):
                run.metadata = {}
            run.metadata[key] = {
                "session_id": session_id,
                "thread_id": thread_id,
                "value": str(value),
            }
            logger.debug(f"Annotated trace with metadata: {key}")
    except (ImportError, Exception) as e:
        logger.debug(f"Could not annotate trace: {e}")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_observability() -> None:
    """
    Initializes all observability and tracing systems.
    Should be called once at application startup.
    """
    settings = get_settings()
    
    if settings.observability.langsmith_enabled:
        configure_langsmith()
        logger.info("Observability systems initialized.")
    else:
        logger.info("Observability disabled in configuration.")
