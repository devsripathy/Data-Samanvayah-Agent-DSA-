"""
Utility functions for the DSA Memory subsystem.

This module provides a collection of pure, reusable, and independently 
testable utility functions used across the memory layer. It includes 
helpers for hashing, serialization, timestamp management, similarity 
calculations, statistics aggregation, and validation.

All functions are designed to be stateless (pure) unless explicitly 
managing internal cache state, ensuring high testability and reliability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

import numpy as np

# ---------------------------------------------------------------------------
# Type Aliases & Constants
# ---------------------------------------------------------------------------

T = TypeVar("T")
DataFrameLike = Any  # Duck-typed for pandas DataFrame or similar structures

HASH_ALGORITHM = "sha256"
DEFAULT_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Checksum & Hashing Utilities
# ---------------------------------------------------------------------------

def calculate_string_checksum(text: str, algorithm: str = HASH_ALGORITHM) -> str:
    """
    Calculates a cryptographic checksum for a given string.
    
    Args:
        text: The input string to hash.
        algorithm: The hashing algorithm to use (default: sha256).
        
    Returns:
        Hexadecimal string representation of the hash.
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode(DEFAULT_ENCODING))
    return hash_obj.hexdigest()


def calculate_file_checksum(file_path: str, algorithm: str = HASH_ALGORITHM, chunk_size: int = 8192) -> str:
    """
    Calculates a cryptographic checksum for a file by reading it in chunks.
    
    Args:
        file_path: Path to the file.
        algorithm: The hashing algorithm to use.
        chunk_size: Size of each chunk to read (in bytes).
        
    Returns:
        Hexadecimal string representation of the file hash.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def generate_schema_hash(schema: dict[str, str]) -> str:
    """
    Generates a deterministic hash for a dataset schema.
    
    The schema dictionary is sorted by keys to ensure the hash is 
    consistent regardless of the original key order.
    
    Args:
        schema: Dictionary mapping column names to their data types.
        
    Returns:
        Hexadecimal string representation of the schema hash.
    """
    sorted_schema = OrderedDict(sorted(schema.items()))
    schema_string = json.dumps(sorted_schema, sort_keys=True)
    return calculate_string_checksum(schema_string)


def generate_dataset_fingerprint(
    schema: dict[str, str], 
    sample_data: str | None = None,
    row_count: int = 0
) -> str:
    """
    Generates a unique fingerprint for a dataset based on its schema and content.
    
    This fingerprint is used to identify identical datasets across different 
    executions, even if they are stored in different locations.
    
    Args:
        schema: Dictionary mapping column names to data types.
        sample_data: A string representation of a data sample (e.g., first 5 rows as CSV).
        row_count: Total number of rows in the dataset.
        
    Returns:
        A composite hexadecimal fingerprint string.
    """
    schema_hash = generate_schema_hash(schema)
    content_hash = calculate_string_checksum(sample_data) if sample_data else "empty"
    composite_input = f"{schema_hash}:{content_hash}:{row_count}"
    return calculate_string_checksum(composite_input)


# ---------------------------------------------------------------------------
# Metadata Extraction
# ---------------------------------------------------------------------------

def extract_dataset_metadata(df: DataFrameLike) -> dict[str, Any]:
    """
    Extracts structural and statistical metadata from a DataFrame-like object.
    
    Uses duck-typing to support pandas, polars, or any object with 
    `.shape`, `.columns`, and `.dtypes` attributes.
    
    Args:
        df: The dataset object (e.g., pandas DataFrame).
        
    Returns:
        Dictionary containing rows, columns, dtypes, and missing value counts.
    """
    metadata: dict[str, Any] = {
        "rows": 0,
        "columns": 0,
        "schema": {},
        "missing_counts": {}
    }
    
    if hasattr(df, "shape"):
        metadata["rows"] = df.shape[0]
        metadata["columns"] = df.shape[1]
        
    if hasattr(df, "columns"):
        metadata["schema"] = {
            str(col): str(dtype) for col, dtype in zip(df.columns, df.dtypes)
        }
        
    if hasattr(df, "isnull"):
        try:
            metadata["missing_counts"] = df.isnull().sum().to_dict()
        except Exception:
            metadata["missing_counts"] = {}
            
    return metadata


# ---------------------------------------------------------------------------
# Serialization & Deserialization
# ---------------------------------------------------------------------------

class MemoryJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles non-serializable types commonly 
    found in memory records (datetime, UUID, numpy types, sets).
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump(mode="json")
        return super().default(obj)


def serialize_to_json(data: Any, indent: int | None = None) -> str:
    """
    Serializes complex Python objects into a JSON string.
    
    Args:
        data: The object to serialize.
        indent: Number of spaces for indentation (None for compact).
        
    Returns:
        JSON formatted string.
    """
    return json.dumps(data, cls=MemoryJSONEncoder, indent=indent)


def deserialize_from_json(json_str: str) -> Any:
    """
    Deserializes a JSON string back into Python objects.
    
    Note: Datetime strings are not automatically converted back to 
    datetime objects to prevent unexpected type coercion. Use 
    `parse_timestamp` if conversion is needed.
    
    Args:
        json_str: The JSON string to deserialize.
        
    Returns:
        Deserialized Python object (dict, list, etc.).
    """
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Timestamp Utilities
# ---------------------------------------------------------------------------

def get_utc_now() -> datetime:
    """
    Returns the current UTC time as a timezone-aware datetime object.
    
    Returns:
        Current UTC datetime.
    """
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime, fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    """
    Formats a datetime object into a string.
    
    Args:
        dt: The datetime object to format.
        fmt: The strftime format string.
        
    Returns:
        Formatted timestamp string.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime(fmt)


def parse_timestamp(ts_str: str, fmt: str | None = None) -> datetime:
    """
    Parses a timestamp string into a timezone-aware UTC datetime object.
    
    Args:
        ts_str: The timestamp string.
        fmt: Optional strftime format. If None, attempts ISO format parsing.
        
    Returns:
        Timezone-aware UTC datetime.
        
    Raises:
        ValueError: If the string cannot be parsed.
    """
    if fmt:
        dt = datetime.strptime(ts_str, fmt)
    else:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_recency_score(
    target_date: datetime, 
    reference_date: datetime | None = None, 
    half_life_days: float = 30.0
) -> float:
    """
    Calculates an exponential decay recency score between 0.0 and 1.0.
    
    A score of 1.0 means the target date is exactly the reference date.
    The score halves every `half_life_days`.
    
    Args:
        target_date: The date of the event (e.g., memory creation).
        reference_date: The current date. Defaults to UTC now.
        half_life_days: The number of days for the score to decay by 50%.
        
    Returns:
        Float score between 0.0 and 1.0.
    """
    if reference_date is None:
        reference_date = get_utc_now()
        
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)
        
    days_diff = (reference_date - target_date).total_seconds() / 86400.0
    
    if days_diff < 0:
        return 1.0
        
    decay_rate = math.log(2) / half_life_days
    return math.exp(-decay_rate * days_diff)


# ---------------------------------------------------------------------------
# Similarity & Math Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    
    Args:
        vec_a: First vector.
        vec_b: Second vector.
        
    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """
    Calculates the Jaccard similarity coefficient between two sets.
    
    Useful for comparing schema overlap or tag similarity.
    
    Args:
        set_a: First set of items.
        set_b: Second set of items.
        
    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Statistics Aggregation
# ---------------------------------------------------------------------------

def calculate_descriptive_stats(values: list[float]) -> dict[str, float]:
    """
    Calculates comprehensive descriptive statistics for a list of numbers.
    
    Args:
        values: List of numerical values.
        
    Returns:
        Dictionary containing count, mean, median, std_dev, min, max, 
        and percentiles (25th, 75th). Returns empty dict if list is empty.
    """
    if not values:
        return {}
        
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    return {
        "count": float(n),
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "std_dev": statistics.stdev(sorted_vals) if n > 1 else 0.0,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "percentile_25": sorted_vals[int(n * 0.25)],
        "percentile_75": sorted_vals[int(n * 0.75)],
    }


def aggregate_quality_scores(scores: list[float], weights: list[float] | None = None) -> float:
    """
    Calculates a weighted average of quality scores.
    
    Args:
        scores: List of quality scores (0.0 to 1.0).
        weights: Optional list of weights. Must match length of scores.
                 Defaults to equal weights.
                 
    Returns:
        Weighted average score. Returns 0.0 if inputs are empty.
        
    Raises:
        ValueError: If lengths of scores and weights do not match.
    """
    if not scores:
        return 0.0
        
    if weights is None:
        weights = [1.0] * len(scores)
        
    if len(scores) != len(weights):
        raise ValueError("Length of scores and weights must match.")
        
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
        
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def is_valid_uuid(uuid_str: str) -> bool:
    """
    Validates if a string is a properly formatted UUID.
    
    Args:
        uuid_str: The string to validate.
        
    Returns:
        True if valid UUID, False otherwise.
    """
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False


def validate_schema_dict(schema: dict[str, str]) -> bool:
    """
    Validates that a schema dictionary has string keys and string values.
    
    Args:
        schema: The schema dictionary to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(schema, dict):
        return False
    return all(isinstance(k, str) and isinstance(v, str) for k, v in schema.items())


# ---------------------------------------------------------------------------
# Cache Utilities
# ---------------------------------------------------------------------------

class AsyncLRUCache:
    """
    A simple, thread-safe, asynchronous LRU cache.
    
    Unlike `functools.lru_cache`, this correctly caches the *result* 
    of async functions rather than the coroutine object.
    """
    
    def __init__(self, max_size: int = 128) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Creates a deterministic string key from function arguments."""
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return calculate_string_checksum(":".join(key_parts))

    async def get(self, key: str) -> Any:
        """Retrieves an item from the cache."""
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    async def put(self, key: str, value: Any) -> None:
        """Inserts or updates an item in the cache."""
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clears all items from the cache."""
        self._cache.clear()


def async_lru_cache(max_size: int = 128) -> Callable:
    """
    Decorator that adds an async LRU cache to an asynchronous function.
    
    Args:
        max_size: Maximum number of items to store in the cache.
        
    Returns:
        Decorator function.
        
    Example:
        @async_lru_cache(max_size=50)
        async def get_expensive_data(key: str) -> dict:
            ...
    """
    cache_instance = AsyncLRUCache(max_size)
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache_instance._make_key(args, kwargs)
            result = await cache_instance.get(key)
            if result is None:
                result = await func(*args, **kwargs)
                await cache_instance.put(key, result)
            return result
        return wrapper
    return decorator
