"""Hybrid retrieval: dense (sentence-transformers) + BM25 over a pgvector store.

Milestone 1 stub. Implementation lands in Milestone 3:
    - embed passages and upsert into pgvector
    - BM25 index for lexical recall
    - fuse dense + lexical scores; optional reranker
    - return top-k passages with provenance for citation
"""
