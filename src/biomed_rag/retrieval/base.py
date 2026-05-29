"""Shared retrieval types so the agent works with any retriever (BM25, hybrid, …)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biomed_rag.ingest.pubmedqa import Passage


@dataclass
class Hit:
    passage: Passage
    score: float  # retriever-native score (BM25 score, or fused RRF score)
    dense_sim: float = 0.0  # best dense cosine similarity, when available (drives abstention)


class Retriever(Protocol):
    def search(self, query: str, k: int = 8) -> list[Hit]: ...
