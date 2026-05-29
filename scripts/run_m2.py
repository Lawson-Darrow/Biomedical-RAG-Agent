"""Milestone 2 — end-to-end skeleton: retrieve → cited answer → one metric.

Loads a small PubMedQA labeled slice, pools the context passages into one
in-memory index, answers each question with citations via a single model through
LLMGateway, and reports task accuracy (predicted decision vs PubMedQA's
final_decision). This proves the loop; the full metric suite arrives in M4.

Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_m2.py --n 15 --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse

from biomed_rag.agent.rag import answer
from biomed_rag.ingest.pubmedqa import build_corpus, load_examples
from biomed_rag.retrieval.store import InMemoryIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of questions")
    ap.add_argument("--model", default="gpt-4.1-mini", help="LLMGateway model id")
    ap.add_argument("--k", type=int, default=6, help="passages retrieved per question")
    args = ap.parse_args()

    examples = load_examples(limit=args.n)
    corpus = build_corpus(examples)
    print(f"loaded {len(examples)} questions, {len(corpus)} passages | model={args.model} k={args.k}")

    index = InMemoryIndex(corpus)

    correct = abstained = 0
    for i, ex in enumerate(examples, 1):
        res = answer(ex.question, index, model=args.model, k=args.k)
        ok = res.decision == ex.final_decision
        correct += ok
        abstained += res.abstained
        print(
            f"{i:2d}. gold={ex.final_decision:5s} pred={res.decision or '?':5s} "
            f"{'OK' if ok else 'XX'} cites={len(res.citations)} abstain={res.abstained}"
        )

    n = len(examples)
    print(f"\naccuracy: {correct}/{n} = {correct / n:.1%}   abstained: {abstained}/{n}")


if __name__ == "__main__":
    main()
