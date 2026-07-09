"""
Constants module for the Data Samanvayah Agent (DSA).

This module defines all global constants, enumerations, and default 
configuration values used throughout the DSA workflow. It ensures 
consistency and type safety across all agent nodes and utility functions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


# ---------------------------------------------------------------------------
# Task Types
# ---------------------------------------------------------------------------

class TaskType(StrEnum):
    """Supported machine learning and AI task types."""
    
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    NLP = "nlp"
    VISION = "vision"


# ---------------------------------------------------------------------------
# Agent Names
# ---------------------------------------------------------------------------

class AgentName(StrEnum):
    """Names of the agents in the LangGraph workflow."""
    
    SUPERVISOR = "supervisor"
    MEMORY = "memory"
    PLANNER = "planner"
    QUALITY = "quality"
    EXPLORER = "explorer"
    TRAINER = "trainer"
    CRITIC = "critic"


# ---------------------------------------------------------------------------
# Execution Status
# ---------------------------------------------------------------------------

class ExecutionStatus(StrEnum):
    """Possible execution statuses for the DSA pipeline."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# Default Thresholds
# ---------------------------------------------------------------------------

MISSING_VALUE_THRESHOLD: Final[float] = 0.40
"""Maximum allowed percentage of missing values in a column before dropping."""

CORRELATION_THRESHOLD: Final[float] = 0.85
"""Threshold for identifying highly correlated features for removal."""

MEMORY_SIMILARITY_THRESHOLD: Final[float] = 0.75
"""Minimum cosine similarity score to retrieve a memory context."""

CONFIDENCE_THRESHOLD: Final[float] = 0.70
"""Minimum confidence score required for the Planner's suggestions."""

MAX_RETRY_LIMIT: Final[int] = 3
"""Maximum number of retry loops allowed for the Critic agent."""


# ---------------------------------------------------------------------------
# Vector Search Settings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: Final[str] = "text-embedding-3-small"
"""Default embedding model used for vectorizing memory and data."""

VECTOR_DB_PATH: Final[str] = "./data/vector_store"
"""Local file path for the vector database (if using local persistence)."""

ARTIFACTS_DIR: Final[str] = "./artifacts"
"""Directory where trained models and execution artifacts are saved."""


# ---------------------------------------------------------------------------
# Logging Constants
# ---------------------------------------------------------------------------

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
"""Standard log formatting string."""

LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Standard date formatting string for logs."""

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
"""Default logging level for the application."""


# ---------------------------------------------------------------------------
# Environment Variable Names
# ---------------------------------------------------------------------------

ENV_LLM_PROVIDER: Final[str] = "LLM_PROVIDER"
"""Environment variable for the LLM provider."""

ENV_LLM_MODEL: Final[str] = "LLM_MODEL"
"""Environment variable for the LLM model name."""

ENV_OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"
"""Environment variable for the OpenAI API key."""

ENV_VECTOR_STORE_URL: Final[str] = "VECTOR_STORE_URL"
"""Environment variable for the vector store connection URL."""

ENV_VECTOR_STORE_COLLECTION: Final[str] = "VECTOR_STORE_COLLECTION"
"""Environment variable for the vector store collection name."""

ENV_MAX_RETRIES: Final[str] = "MAX_RETRIES"
"""Environment variable for overriding the default max retry limit."""

ENV_LOG_LEVEL: Final[str] = "LOG_LEVEL"
"""Environment variable for overriding the default log level."""
