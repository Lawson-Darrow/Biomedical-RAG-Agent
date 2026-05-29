"""Evaluation harness: run the agent over a set of examples and aggregate metrics.

Retrieval happens once per question and the same hits feed the answer, the
retrieval metrics, and the judge — so [n] citations line up across all three.
"""

from __future__ import annotations

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
) -> dict:
    preds: list[str] = []
    golds: list[str] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    hitrates: list[float] = []
    faiths: list[float] = []
    hallucs: list[float] = []
    cit_accs: list[float] = []
    records: list[dict] = []
    n_abstain = 0

    for ex in examples:
        hits = index.search(ex.question, k=k)
        retrieved_keys = [(h.passage.doc_id, h.passage.idx) for h in hits]
        relevant = {(ex.qid, i) for i in range(len(ex.contexts))}

        recalls.append(metrics.recall_at_k(retrieved_keys, relevant))
        mrrs.append(metrics.mrr(retrieved_keys, relevant))
        ndcgs.append(metrics.ndcg_at_k(retrieved_keys, relevant, k))
        hitrates.append(metrics.hit_rate(retrieved_keys, relevant))

        res = synthesize(ex.question, hits, model=model, abstain_threshold=abstain_threshold)
        preds.append(res.decision)
        golds.append(ex.final_decision)

        rec = {
            "qid": ex.qid,
            "gold": ex.final_decision,
            "pred": res.decision,
            "abstained": res.abstained,
            "n_citations": len(res.citations),
        }

        if res.abstained:
            n_abstain += 1
        elif with_judge:
            context = "\n\n".join(f"[{n}] {h.passage.text}" for n, h in enumerate(hits, 1))
            j = judge_answer(
                ex.question, context, res.answer, [c["n"] for c in res.citations], model=judge_model
            )
            # Only aggregate where the metric is defined: faithfulness/hallucination
            # over answers with >=1 claim; citation accuracy over answers that cited.
            if j.n_claims > 0:
                faiths.append(j.faithfulness)
                hallucs.append(j.hallucination_rate)
            if res.citations:
                cit_accs.append(j.citation_accuracy)
            rec |= {
                "faithfulness": round(j.faithfulness, 3),
                "hallucination": round(j.hallucination_rate, 3),
                "citation_acc": round(j.citation_accuracy, 3),
                "n_claims": j.n_claims,
            }

        records.append(rec)

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
