"""Centralized configuration, loaded from environment / .env.

See `.env.example` for the full list of settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Frontier providers (at least one required for the frontier arm)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Open-weights endpoint (OpenAI-compatible)
    open_model_base_url: str = "http://localhost:11434/v1"
    open_model_name: str = "qwen2.5:7b"
    open_model_api_key: str = "ollama"

    # Vector store
    database_url: str = "postgresql://postgres:postgres@localhost:5432/biomed_rag"

    # Retrieval / embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 8


settings = Settings()
