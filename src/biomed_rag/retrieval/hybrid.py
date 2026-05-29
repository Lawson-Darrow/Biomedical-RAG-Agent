"""Hybrid retrieval: BM25 (lexical) + pgvector dense, fused with Reciprocal Rank Fusion.

RRF is parameter-light and robust — it fuses by rank, so the two arms' incomparable
score scales don't matter. `dense_sim` is carried through for the abstention gate.
"""

from __future__ import annotations

import re

import psycopg
from rank_bm25 import BM25Okapi

from biomed_rag.ingest.pubmedqa import Passage
from biomed_rag.retrieval.base import Hit
from biomed_rag.retrieval.pgvector_store import dense_search

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HybridIndex:
    def __init__(
        self,
        passages: list[Passage],
        conn: psycopg.Connection,
        rrf_k: int = 60,
        candidate_k: int = 30,
    ):
        self.passages = passages
        self.conn = conn
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._by_key = {(p.doc_id, p.idx): p for p in passages}
        self._bm25 = BM25Okapi([_tokenize(p.text) for p in passages])

    def search(self, query: str, k: int = 8) -> list[Hit]:
        # Dense arm (pgvector): rank + similarity per passage key.
        dense = dense_search(self.conn, query, self.candidate_k)
        dense_rank = {(d, i): (rank, sim) for rank, (d, i, sim) in enumerate(dense)}

        # Lexical arm (BM25): rank per passage key.
        bm = self._bm25.get_scores(_tokenize(query))
        bm_top = sorted(range(len(self.passages)), key=lambda i: bm[i], reverse=True)
        bm_rank = {
            (self.passages[i].doc_id, self.passages[i].idx): rank
            for rank, i in enumerate(bm_top[: self.candidate_k])
        }

        # Reciprocal Rank Fusion.
        fused = []
        for key in set(dense_rank) | set(bm_rank):
            score = 0.0
            if key in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[key][0])
            if key in bm_rank:
                score += 1.0 / (self.rrf_k + bm_rank[key])
            sim = dense_rank[key][1] if key in dense_rank else 0.0
            fused.append((key, score, sim))

        fused.sort(key=lambda t: t[1], reverse=True)
        return [Hit(self._by_key[key], score, sim) for key, score, sim in fused[:k]]
