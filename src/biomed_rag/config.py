"""Centralized configuration, loaded from environment / .env.

See `.env.example` for the full list of settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # All model traffic routes through LLMGateway (OSS, OpenAI-compatible).
    # Frontier (Claude/GPT) and open (DeepSeek/Kimi/Qwen) are all model IDs
    # behind this one endpoint. Provider keys are added BYOK in the LLMGateway
    # dashboard, so the app only needs the gateway key.
    llmgateway_api_key: str | None = None
    llmgateway_base_url: str = "https://api.llmgateway.io/v1"

    # Vector store
    database_url: str = "postgresql://postgres:postgres@localhost:5432/biomed_rag"

    # Retrieval / embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 8


settings = Settings()
