"""Dense embeddings via fastembed (ONNX, OSS, no torch — works on Python 3.14).

BGE models are asymmetric: queries get an instruction prefix, passages don't.
fastembed's `query_embed` / `embed` handle that distinction for us.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384


@lru_cache(maxsize=2)
def _model(name: str) -> TextEmbedding:
    return TextEmbedding(name)


def encode_passages(texts: list[str], model: str = DENSE_MODEL) -> list[list[float]]:
    return [[float(x) for x in v] for v in _model(model).embed(texts)]


def encode_query(text: str, model: str = DENSE_MODEL) -> list[float]:
    return [float(x) for x in next(iter(_model(model).query_embed([text])))]
