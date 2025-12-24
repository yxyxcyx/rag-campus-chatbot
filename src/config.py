# config.py

"""
Centralized Configuration Module

Single source of truth for all environment variables with type validation.
Fail-fast: Application refuses to start if critical configuration is missing.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


# Load .env file once at module import
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    All configuration is loaded from environment variables.
    Critical settings will cause startup failure if missing.
    """
    
    # === Critical Settings (Must be present) ===
    groq_api_key: str = Field(
        ...,
        description="Groq API key for LLM access. Required."
    )
    
    # === Database Settings ===
    chroma_db_path: str = Field(
        default="./chroma_db",
        description="Path to ChromaDB persistent storage"
    )
    collection_name: str = Field(
        default="collection",
        description="ChromaDB collection name"
    )
    
    # === Redis Settings (for Celery) ===
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # === Hybrid Search Settings ===
    enable_hybrid_search: bool = Field(
        default=False,
        description="Enable hybrid BM25 + Vector search"
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 in hybrid search (0.0-1.0)"
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid search (0.0-1.0)"
    )
    
    # === Diversity Filter Settings ===
    use_diversity_filter: bool = Field(
        default=True,
        description="Enable diversity filtering to reduce redundant results"
    )
    diversity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for diversity filter (0.0-1.0)"
    )
    
    # === Model Settings ===
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace embedding model name. Options: all-MiniLM-L6-v2 (fast, 22M params), all-mpnet-base-v2 (better quality, 110M params)"
    )
    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    llm_model_name: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq LLM model name"
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens for LLM response"
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for LLM API calls in seconds"
    )
    
    # === Retrieval Settings ===
    n_initial_retrieval: int = Field(
        default=20,
        ge=1,
        description="Number of documents for initial retrieval"
    )
    n_final_results: int = Field(
        default=5,
        ge=1,
        description="Number of final documents after reranking"
    )
    
    # === Cache Settings ===
    cache_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic cache hits"
    )
    cache_max_age_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum age for cache entries in hours"
    )
    
    # === Application Settings ===
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    api_version: str = Field(
        default="2.0.0",
        description="API version string"
    )
    
    @field_validator("groq_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is not empty."""
        if not v or not v.strip():
            raise ValueError(
                "GROQ_API_KEY is required and cannot be empty. "
                "Please set it in your .env file or environment variables."
            )
        return v.strip()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {valid_levels}"
            )
        return v_upper
    
    @field_validator("bm25_weight", "vector_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ensure weights are valid floats."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Weight must be a number, got {type(v)}")
        return float(v)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Map environment variable names (case-insensitive)
        populate_by_name = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once
    and the same instance is returned on subsequent calls.
    
    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()


def validate_startup_config() -> Settings:
    """
    Validate configuration at startup.
    
    Call this at application startup to ensure all config is valid.
    Will raise clear error messages if configuration is invalid.
    
    Returns:
        Settings: Validated settings instance
        
    Raises:
        SystemExit: If critical configuration is missing or invalid
    """
    try:
        settings = get_settings()
        return settings
    except Exception as e:
        # Format error message for clarity
        error_msg = str(e)
        print(f"\n{'='*60}")
        print("CONFIGURATION ERROR - Application cannot start")
        print(f"{'='*60}")
        print(f"\n{error_msg}\n")
        print("Please check your .env file and environment variables.")
        print(f"{'='*60}\n")
        raise SystemExit(1)
