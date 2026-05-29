"""Milestone 3 — hybrid retrieval (BM25 + pgvector dense, RRF) + abstention.

1. Ingest the PubMedQA passage corpus into pgvector (embeddings via fastembed).
2. Report retrieval hit-rate@k for BM25 vs dense vs hybrid — a passage is
   "relevant" to a question if it comes from that question's own source doc.
3. Run the agent over a subset with the retrieval-gated abstention threshold,
   and report task accuracy.

Prereq: docker compose up -d   (Postgres + pgvector)
Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_m3.py --n 100 --eval-n 15
"""

from __future__ import annotations

import argparse

from biomed_rag.agent.rag import answer
from biomed_rag.ingest.pubmedqa import build_corpus, load_examples
from biomed_rag.retrieval.hybrid import HybridIndex
from biomed_rag.retrieval.pgvector_store import connect, dense_search, ingest, init_schema, reset_corpus
from biomed_rag.retrieval.store import InMemoryIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="questions/corpus for retrieval eval")
    ap.add_argument("--eval-n", type=int, default=15, help="subset for LLM task accuracy")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--abstain", type=float, default=0.5, help="min dense_sim to answer")
    args = ap.parse_args()

    examples = load_examples(limit=args.n)
    corpus = build_corpus(examples)

    conn = connect()
    init_schema(conn)
    reset_corpus(conn)
    n_ingested = ingest(conn, corpus)
    print(f"corpus: {len(examples)} questions, {len(corpus)} passages ({n_ingested} upserted to pgvector)")

    bm25 = InMemoryIndex(corpus)
    hybrid = HybridIndex(corpus, conn)
    k = args.k

    # Retrieval hit-rate@k: did top-k include a passage from the question's own doc?
    bm = dn = hy = 0
    for ex in examples:
        bm += any(h.passage.doc_id == ex.qid for h in bm25.search(ex.question, k))
        dn += any(d == ex.qid for d, _, _ in dense_search(conn, ex.question, k))
        hy += any(h.passage.doc_id == ex.qid for h in hybrid.search(ex.question, k))
    n = len(examples)
    print(f"\nretrieval hit-rate@{k} over {n} questions:")
    print(f"  bm25   : {bm/n:.1%}")
    print(f"  dense  : {dn/n:.1%}")
    print(f"  hybrid : {hy/n:.1%}")

    # Task accuracy on a subset, hybrid retriever + abstention gate.
    print(f"\ntask accuracy (hybrid, model={args.model}, abstain<{args.abstain}) over {args.eval_n}:")
    correct = abstained = 0
    for i, ex in enumerate(examples[: args.eval_n], 1):
        res = answer(ex.question, hybrid, model=args.model, k=k, abstain_threshold=args.abstain)
        ok = res.decision == ex.final_decision and not res.abstained
        correct += ok
        abstained += res.abstained
        print(
            f"{i:2d}. gold={ex.final_decision:5s} pred={res.decision or '?':5s} "
            f"{'OK' if ok else ('AB' if res.abstained else 'XX')} cites={len(res.citations)}"
        )
    m = args.eval_n
    print(f"\naccuracy: {correct}/{m} = {correct/m:.1%}   abstained: {abstained}/{m}")
    conn.close()


if __name__ == "__main__":
    main()
