"""Tune the retrieval-gated abstention threshold (no LLM calls — pure retrieval).

Sweeps thresholds over the best dense similarity per question, reporting how often
the gate would abstain on answerable PubMedQA questions (want low) vs off-topic
questions (want high). Pick the threshold with the best separation.

Prereq: docker compose up -d
Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/tune_abstain.py --corpus-n 300 --n 100
"""

from __future__ import annotations

import argparse

from biomed_rag.eval.harness import OFF_TOPIC, abstention_sweep
from biomed_rag.ingest.pubmedqa import build_corpus, load_examples
from biomed_rag.retrieval.hybrid import HybridIndex
from biomed_rag.retrieval.pgvector_store import connect, ingest, init_schema, reset_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-n", type=int, default=300)
    ap.add_argument("--n", type=int, default=100, help="answerable questions to sweep over")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    examples = load_examples(limit=args.corpus_n)
    corpus = build_corpus(examples)
    conn = connect()
    init_schema(conn)
    reset_corpus(conn)
    ingest(conn, corpus)
    index = HybridIndex(corpus, conn)

    answerable = [ex.question for ex in examples[: args.n]]
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    sweep = abstention_sweep(answerable, OFF_TOPIC, index, args.k, thresholds)
    conn.close()

    print(f"abstention sweep | answerable n={len(answerable)} off_topic n={len(OFF_TOPIC)}\n")
    print("  thresh | answerable_abstain (lower better) | off_topic_abstain (higher better)")
    for t in thresholds:
        s = sweep[t]
        print(f"   {t:.2f}  |        {s['answerable_abstain']:.3f}            |        {s['off_topic_abstain']:.3f}")


if __name__ == "__main__":
    main()
