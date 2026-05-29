"""Evaluation harness: run the agent over a set of examples and aggregate metrics.

Retrieval happens once per question and the same hits feed the answer, the
retrieval metrics, and the judge — so [n] citations line up across all three.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from biomed_rag.agent.rag import synthesize
from biomed_rag.eval import metrics
from biomed_rag.eval.judge import JUDGE_MODEL, judge_answer
from biomed_rag.ingest.pubmedqa import Example
from biomed_rag.retrieval.base import Retriever

# Off-topic / out-of-corpus questions: a correct system should ABSTAIN on these.
OFF_TOPIC = [
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "What is the best programming language for web development?",
    "How do I bake a sourdough loaf?",
    "What was the closing price of the S&P 500 yesterday?",
]


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(
    examples: list[Example],
    index: Retriever,
    model: str,
    judge_model: str = JUDGE_MODEL,
    k: int = 6,
    abstain_threshold: float | None = None,
    with_judge: bool = True,
    workers: int = 1,
) -> dict:
    # Phase 1 — retrieval (sequential; the pgvector connection isn't thread-safe).
    # Cheap relative to the LLM calls, and lets retrieval metrics be computed here.
    hits_by_ex = [index.search(ex.question, k=k) for ex in examples]
    recalls, mrrs, ndcgs, hitrates = [], [], [], []
    for ex, hits in zip(examples, hits_by_ex):
        keys = [(h.passage.doc_id, h.passage.idx) for h in hits]
        relevant = {(ex.qid, i) for i in range(len(ex.contexts))}
        recalls.append(metrics.recall_at_k(keys, relevant))
        mrrs.append(metrics.mrr(keys, relevant))
        ndcgs.append(metrics.ndcg_at_k(keys, relevant, k))
        hitrates.append(metrics.hit_rate(keys, relevant))

    # Phase 2 — generate + judge per example (parallelizable: pure HTTP, no shared DB).
    def work(i: int) -> dict:
        ex, hits = examples[i], hits_by_ex[i]
        res = synthesize(ex.question, hits, model=model, abstain_threshold=abstain_threshold)
        rec = {
            "qid": ex.qid,
            "gold": ex.final_decision,
            "pred": res.decision,
            "abstained": res.abstained,
            "n_citations": len(res.citations),
        }
        if not res.abstained and with_judge:
            context = "\n\n".join(f"[{n}] {h.passage.text}" for n, h in enumerate(hits, 1))
            j = judge_answer(
                ex.question, context, res.answer, [c["n"] for c in res.citations], model=judge_model
            )
            rec |= {
                "faithfulness": round(j.faithfulness, 3),
                "hallucination": round(j.hallucination_rate, 3),
                "citation_acc": round(j.citation_accuracy, 3),
                "n_claims": j.n_claims,
                "_cited": bool(res.citations),
            }
        return rec

    idxs = range(len(examples))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(work, idxs))
    else:
        records = [work(i) for i in idxs]

    # Aggregate (order preserved by pool.map).
    preds = [r["pred"] for r in records]
    golds = [r["gold"] for r in records]
    n_abstain = sum(r["abstained"] for r in records)
    # faithfulness/hallucination over answers with >=1 claim; citation acc over answers that cited.
    faiths = [r["faithfulness"] for r in records if r.get("n_claims", 0) > 0]
    hallucs = [r["hallucination"] for r in records if r.get("n_claims", 0) > 0]
    cit_accs = [r["citation_acc"] for r in records if r.get("_cited")]

    return {
        "config": {
            "model": model,
            "judge_model": judge_model if with_judge else None,
            "k": k,
            "abstain_threshold": abstain_threshold,
            "n": len(examples),
        },
        "n_abstained": n_abstain,
        "retrieval": {
            f"recall@{k}": _avg(recalls),
            "mrr": _avg(mrrs),
            f"ndcg@{k}": _avg(ndcgs),
            f"hit_rate@{k}": _avg(hitrates),
        },
        "task": {"accuracy": metrics.accuracy(preds, golds), "macro_f1": metrics.macro_f1(preds, golds)},
        "grounding": {
            "faithfulness": _avg(faiths),
            "hallucination_rate": _avg(hallucs),
            "citation_accuracy": _avg(cit_accs),
            "n_judged": len(faiths),
            "n_cited": len(cit_accs),
        },
        "records": records,
    }


def abstention_probe(
    questions: list[str],
    index: Retriever,
    model: str,
    k: int = 6,
    abstain_threshold: float | None = None,
) -> float:
    """Fraction of questions the system abstained on (higher is better for off-topic)."""
    abstained = 0
    for q in questions:
        res = synthesize(q, index.search(q, k=k), model=model, abstain_threshold=abstain_threshold)
        abstained += res.abstained
    return abstained / len(questions) if questions else 0.0


def closed_book_eval(examples: list[Example], model: str, workers: int = 1) -> dict:
    """Task accuracy with no retrieval (grounding-ablation baseline)."""
    from biomed_rag.agent.rag import answer_closed_book

    def work(i: int) -> tuple[str, str]:
        ex = examples[i]
        return answer_closed_book(ex.question, model).decision, ex.final_decision

    idxs = range(len(examples))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(work, idxs))
    else:
        pairs = [work(i) for i in idxs]

    preds = [p for p, _ in pairs]
    golds = [g for _, g in pairs]
    return {
        "accuracy": metrics.accuracy(preds, golds),
        "macro_f1": metrics.macro_f1(preds, golds),
        "n": len(examples),
    }


def abstention_sweep(
    answerable: list[str],
    off_topic: list[str],
    index: Retriever,
    k: int,
    thresholds: list[float],
) -> dict[float, dict[str, float]]:
    """Tune the abstention threshold without any LLM calls.

    The gate is purely retrieval-based (abstain if best dense_sim < threshold), so we
    only need each question's best dense similarity, then sweep thresholds over it.
    """
    ans_sims = [max((h.dense_sim for h in index.search(q, k=k)), default=0.0) for q in answerable]
    off_sims = [max((h.dense_sim for h in index.search(q, k=k)), default=0.0) for q in off_topic]
    return {
        t: {
            "answerable_abstain": sum(s < t for s in ans_sims) / len(ans_sims) if ans_sims else 0.0,
            "off_topic_abstain": sum(s < t for s in off_sims) / len(off_sims) if off_sims else 0.0,
        }
        for t in thresholds
    }
