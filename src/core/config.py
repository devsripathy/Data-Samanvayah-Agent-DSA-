"""
Configuration module for the Data Samanvayah Agent (DSA).

This module centralizes all configuration settings, loading them from 
environment variables (.env), a YAML configuration file (config.yaml), 
and fallback defaults. It uses Pydantic v2 BaseSettings for robust 
validation and type safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, PostgresDsn
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


# ---------------------------------------------------------------------------
# Custom YAML Settings Source
# ---------------------------------------------------------------------------

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    Custom Pydantic settings source to load configuration from a YAML file.
    """

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        self.yaml_data = self._load_yaml()

    def _load_yaml(self) -> dict[str, Any]:
        """Loads and parses the config.yaml file if it exists."""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Required method for PydanticBaseSettingsSource, not used here."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Returns the parsed YAML data to be merged with other settings sources."""
        return self.yaml_data


# ---------------------------------------------------------------------------
# Nested Configuration Models
# ---------------------------------------------------------------------------

class LLMSettings(BaseModel):
    """Configuration for Large Language Model providers."""
    provider: Literal["openai", "ollama", "deepseek", "vllm"] = "openai"
    model: str = "gpt-4o"
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class DatabaseSettings(BaseModel):
    """Configuration for relational database connections."""
    type: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: Path = Path("./data/dsa.db")
    postgres_dsn: Optional[PostgresDsn] = None


class VectorStoreSettings(BaseModel):
    """Configuration for vector database connections."""
    type: Literal["chromadb"] = "chromadb"
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "dsa_memory"
    persist_directory: Path = Path("./data/chroma_db")


class ObservabilitySettings(BaseModel):
    """Configuration for tracing and observability platforms."""
    langsmith_enabled: bool = False
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "dsa-agent"
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[SecretStr] = None
    langfuse_secret_key: Optional[SecretStr] = None
    langfuse_host: str = "https://cloud.langfuse.com"


class CacheSettings(BaseModel):
    """Configuration for caching layers."""
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"


class ArtifactSettings(BaseModel):
    """Configuration for file and artifact storage directories."""
    base_dir: Path = Path("./artifacts")
    models_dir: Path = Path("./artifacts/models")
    logs_dir: Path = Path("./artifacts/logs")
    reports_dir: Path = Path("./artifacts/reports")


class LoggingSettings(BaseModel):
    """Configuration for application logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file_path: Optional[Path] = None


class EmbeddingSettings(BaseModel):
    """Configuration for text embedding models."""
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    provider: Literal["openai", "ollama", "huggingface"] = "openai"


class AutoMLSettings(BaseModel):
    """Configuration for AutoML training parameters."""
    max_trials: int = Field(default=50, gt=0)
    timeout_seconds: int = Field(default=3600, gt=0)
    metric: Literal["accuracy", "f1", "roc_auc", "rmse", "mae"] = "accuracy"
    n_jobs: int = -1


class RetrySettings(BaseModel):
    """Configuration for retry mechanisms and backoff strategies."""
    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=2.0, gt=0)
    max_delay_seconds: float = Field(default=60.0, gt=0)


class VisualizationSettings(BaseModel):
    """Configuration for data visualization and plotting."""
    default_theme: str = "plotly_white"
    max_plot_rows: int = Field(default=10000, gt=0)
    output_format: Literal["png", "html", "svg"] = "html"


# ---------------------------------------------------------------------------
# Main Settings Class
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Main configuration class for the Data Samanvayah Agent.
    
    Loads settings from (in order of precedence):
    1. Initialization arguments
    2. Environment variables (.env)
    3. YAML configuration file (config.yaml)
    4. Fallback defaults defined in the models
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_assignment=True,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    automl: AutoMLSettings = Field(default_factory=AutoMLSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customizes the settings sources to include the YAML configuration file.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


# ---------------------------------------------------------------------------
# Singleton Accessor
# ---------------------------------------------------------------------------

_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Returns a singleton instance of the Settings class.
    
    This ensures that the configuration is loaded and validated only once 
    during the application lifecycle, improving performance and consistency.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
