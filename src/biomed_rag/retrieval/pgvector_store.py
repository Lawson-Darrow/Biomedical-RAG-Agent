"""Postgres + pgvector store for the dense arm.

Holds passages and their embeddings; dense search is exact cosine (`<=>`) — fine
at PubMedQA scale. An ANN index (ivfflat/hnsw) is a one-liner to add when the
corpus grows to the PMC slice.
"""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from biomed_rag.config import settings
from biomed_rag.ingest.pubmedqa import Passage
from biomed_rag.retrieval.embeddings import DENSE_DIM, encode_passages, encode_query


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS passages (
                id        bigserial PRIMARY KEY,
                doc_id    text NOT NULL,
                idx       int  NOT NULL,
                text      text NOT NULL,
                embedding vector({DENSE_DIM}),
                UNIQUE (doc_id, idx)
            )
            """
        )
    conn.commit()


def ingest(conn: psycopg.Connection, passages: list[Passage]) -> int:
    """Embed and upsert passages. Idempotent on (doc_id, idx)."""
    vectors = encode_passages([p.text for p in passages])
    rows = [
        (p.doc_id, p.idx, p.text, np.asarray(v, dtype=np.float32))
        for p, v in zip(passages, vectors)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO passages (doc_id, idx, text, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id, idx)
            DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def dense_search(conn: psycopg.Connection, query: str, k: int) -> list[tuple[str, int, float]]:
    """Return [(doc_id, idx, cosine_sim)] for the top-k nearest passages."""
    qv = np.asarray(encode_query(query), dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, idx, 1 - (embedding <=> %s) AS sim "
            "FROM passages ORDER BY embedding <=> %s LIMIT %s",
            (qv, qv, k),
        )
        return [(doc_id, idx, float(sim)) for doc_id, idx, sim in cur.fetchall()]
