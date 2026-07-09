"""Centralized configuration for DSA."""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    
    # Vector Store / Memory
    vector_store_url: str = "http://localhost:6333"
    vector_store_collection: str = "dsa_memory"
    
    # Execution
    max_retries: int = 3
    log_level: str = "INFO"

settings = Settings()
