"""Unit tests for the pure metric functions (no network / no model calls)."""

import math

from biomed_rag.eval import metrics


def test_hit_recall_mrr_first_relevant():
    retrieved = [("d1", 0), ("d2", 0), ("d1", 1)]
    relevant = {("d1", 0), ("d1", 1)}
    assert metrics.hit_rate(retrieved, relevant) == 1.0
    assert metrics.recall_at_k(retrieved, relevant) == 1.0  # both relevant retrieved
    assert metrics.mrr(retrieved, relevant) == 1.0  # first item is relevant


def test_mrr_and_ndcg_by_rank():
    retrieved = [("x", 0), ("d1", 0)]  # relevant at rank 2
    relevant = {("d1", 0)}
    assert metrics.mrr(retrieved, relevant) == 0.5
    assert abs(metrics.ndcg_at_k(retrieved, relevant, 6) - 1.0 / math.log2(3)) < 1e-9


def test_no_relevant_or_no_hit():
    assert metrics.hit_rate([("a", 0)], {("b", 0)}) == 0.0
    assert metrics.recall_at_k([("a", 0)], set()) == 0.0
    assert metrics.mrr([("a", 0)], {("b", 0)}) == 0.0


def test_task_metrics():
    preds = ["yes", "no", "maybe", "yes"]
    golds = ["yes", "no", "yes", "yes"]
    assert metrics.accuracy(preds, golds) == 0.75
    assert 0.0 <= metrics.macro_f1(preds, golds) <= 1.0

    perfect = ["yes", "no", "maybe"]
    assert metrics.macro_f1(perfect, perfect) == 1.0
