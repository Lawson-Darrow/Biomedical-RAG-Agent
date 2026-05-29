"""Milestone 4 — run the full evaluation harness and write a reproducible report.

Ingests a corpus into pgvector, builds the hybrid retriever, evaluates a fixed
slice (retrieval + task + grounding metrics via LLM judge), probes abstention on
off-topic questions, prints a summary, and writes JSON to eval_results/.

Prereq: docker compose up -d
Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_eval.py --corpus-n 300 --n 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biomed_rag.eval.harness import OFF_TOPIC, abstention_probe, evaluate
from biomed_rag.ingest.pubmedqa import build_corpus, load_examples
from biomed_rag.retrieval.hybrid import HybridIndex
from biomed_rag.retrieval.pgvector_store import connect, ingest, init_schema, reset_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-n", type=int, default=300, help="questions whose passages form the corpus")
    ap.add_argument("--n", type=int, default=40, help="questions to evaluate (first N of the corpus)")
    ap.add_argument("--model", default="gpt-4.1-mini", help="generator model")
    ap.add_argument("--judge-model", default="gpt-4.1", help="judge model")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--abstain", type=float, default=0.5, help="min dense_sim to answer")
    ap.add_argument("--no-judge", action="store_true", help="skip the grounding judge (no judge calls)")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    examples = load_examples(limit=args.corpus_n)
    corpus = build_corpus(examples)
    conn = connect()
    init_schema(conn)
    reset_corpus(conn)
    ingest(conn, corpus)
    print(f"corpus: {len(examples)} questions, {len(corpus)} passages in pgvector")

    index = HybridIndex(corpus, conn)
    eval_examples = examples[: args.n]
    print(f"evaluating {len(eval_examples)} questions | model={args.model} judge={args.judge_model}\n")

    report = evaluate(
        eval_examples,
        index,
        model=args.model,
        judge_model=args.judge_model,
        k=args.k,
        abstain_threshold=args.abstain,
        with_judge=not args.no_judge,
    )

    off_topic_abstain = abstention_probe(OFF_TOPIC, index, model=args.model, k=args.k, abstain_threshold=args.abstain)
    report["abstention"] = {
        "answerable_abstain_rate": report["n_abstained"] / len(eval_examples),
        "off_topic_abstain_rate": off_topic_abstain,
        "off_topic_n": len(OFF_TOPIC),
    }
    conn.close()

    # --- summary ---
    r, t, g, a = report["retrieval"], report["task"], report["grounding"], report["abstention"]
    print("== retrieval ==")
    for key, val in r.items():
        print(f"  {key:12s}: {val:.3f}")
    print("== task ==")
    print(f"  accuracy    : {t['accuracy']:.3f}")
    print(f"  macro_f1    : {t['macro_f1']:.3f}")
    print(f"== grounding (judge={args.judge_model}, n_judged={g['n_judged']}) ==")
    print(f"  faithfulness     : {g['faithfulness']:.3f}")
    print(f"  hallucination    : {g['hallucination_rate']:.3f}")
    print(f"  citation_accuracy: {g['citation_accuracy']:.3f}")
    print("== abstention ==")
    print(f"  on answerable (lower better): {a['answerable_abstain_rate']:.3f}")
    print(f"  on off-topic  (higher better): {a['off_topic_abstain_rate']:.3f}")

    out = Path(args.out) if args.out else Path("eval_results") / f"eval_{args.model.replace('/', '-')}_n{args.n}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
