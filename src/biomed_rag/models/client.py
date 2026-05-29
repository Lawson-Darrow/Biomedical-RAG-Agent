"""Model-agnostic client. Every model (frontier + open) is reached through the
one LLMGateway OpenAI-compatible endpoint, so swapping models is just an ID change.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from biomed_rag.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.llmgateway_api_key:
        raise RuntimeError("LLMGATEWAY_API_KEY not set — add it to .env")
    return OpenAI(api_key=settings.llmgateway_api_key, base_url=settings.llmgateway_base_url)


def chat(model: str, messages: list[dict], **kwargs) -> str:
    """One chat completion; returns the assistant text."""
    resp = get_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content or ""


def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts; returns one vector per input, in order."""
    resp = get_client().embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
