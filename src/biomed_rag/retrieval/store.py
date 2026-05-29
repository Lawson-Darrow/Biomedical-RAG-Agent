"""Lexical retrieval for the Milestone 2 skeleton (BM25, pure-Python via rank-bm25).

Why BM25 here: the LLMGateway "coding plan" blocks the embeddings endpoint (403),
and BM25 is anyway one half of the planned hybrid retriever — so this is real
progress, not a stopgap. Milestone 3 adds the dense arm (local OSS embeddings via
sentence-transformers, per the OSS-tooling principle) and fuses the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from biomed_rag.ingest.pubmedqa import Passage

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    passage: Passage
    score: float


class InMemoryIndex:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self._bm25 = BM25Okapi([_tokenize(p.text) for p in passages])

    def search(self, query: str, k: int = 8) -> list[Hit]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.passages, scores), key=lambda t: t[1], reverse=True)
        return [Hit(p, float(s)) for p, s in ranked[:k]]
