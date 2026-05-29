"""Pure metric functions — no I/O, no model calls.

Retrieval metrics treat a passage as relevant to a question if it comes from that
question's own source document. Keys are (doc_id, idx) tuples.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

Key = Hashable


def hit_rate(retrieved: Sequence[Key], relevant: set[Key]) -> float:
    return 1.0 if any(r in relevant for r in retrieved) else 0.0


def recall_at_k(retrieved: Sequence[Key], relevant: set[Key]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def mrr(retrieved: Sequence[Key], relevant: set[Key]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[Key], relevant: set[Key], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def macro_f1(preds: Sequence[str], golds: Sequence[str], labels=("yes", "no", "maybe")) -> float:
    f1s = []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(preds, golds))
        fp = sum(p == lab and g != lab for p, g in zip(preds, golds))
        fn = sum(p != lab and g == lab for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)
