"""Milestone 5 — frontier vs. open comparison across models on one harness.

Ingests the corpus once, then runs the M4 eval for each generator model (same
examples, same retrieval, same fixed judge). Pings each model first and runs only
the working ones, logging exclusions (no silent gaps). Records the gateway's
resolved model string per model (provider/region provenance) and avg latency.

Prereq: docker compose up -d
Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_compare.py --n 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from biomed_rag.eval.harness import OFF_TOPIC, abstention_probe, closed_book_eval, evaluate
from biomed_rag.ingest.pubmedqa import build_corpus, load_examples
from biomed_rag.models.client import get_client
from biomed_rag.retrieval.hybrid import HybridIndex
from biomed_rag.retrieval.pgvector_store import connect, ingest, init_schema, reset_corpus

DEFAULT_MODELS = [
    ("gpt-4.1-mini", "frontier"),
    ("claude-haiku-4-5", "frontier"),
    ("deepseek-v3.2", "open"),
    ("qwen3-235b-a22b-instruct-2507", "open"),
    ("qwen-flash", "open"),
    ("kimi-k2", "open"),
]


def ping(model: str) -> str:
    """Return the gateway-resolved model string, or raise."""
    r = get_client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": "OK"}], max_tokens=3, temperature=0
    )
    return r.model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-n", type=int, default=300)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--judge-model", default="gpt-4.1")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--abstain", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=6, help="concurrent gen+judge calls")
    ap.add_argument("--ablation-model", default="gpt-4.1-mini", help="closed-book baseline model ('' to skip)")
    args = ap.parse_args()

    # Judge must work; fall back to gpt-4.1-mini if the preferred judge is gated.
    judge = args.judge_model
    try:
        ping(judge)
    except Exception as e:  # noqa: BLE001
        print(f"judge {judge} unavailable ({type(e).__name__}); falling back to gpt-4.1-mini")
        judge = "gpt-4.1-mini"

    examples = load_examples(limit=args.corpus_n)
    corpus = build_corpus(examples)
    conn = connect()
    init_schema(conn)
    reset_corpus(conn)
    ingest(conn, corpus)
    index = HybridIndex(corpus, conn)
    eval_examples = examples[: args.n]
    print(f"corpus {len(corpus)} passages | eval {len(eval_examples)} q | judge={judge}\n")

    rows = []
    for model, tier in DEFAULT_MODELS:
        try:
            resolved = ping(model)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {model:32s} ({type(e).__name__})")
            continue

        t0 = time.time()
        rep = evaluate(
            eval_examples, index, model=model, judge_model=judge,
            k=args.k, abstain_threshold=args.abstain, workers=args.workers,
        )
        dt = time.time() - t0
        ab = {
            "answerable": rep["n_abstained"] / len(eval_examples),
            "off_topic": abstention_probe(OFF_TOPIC, index, model=model, k=args.k, abstain_threshold=args.abstain),
        }
        rows.append({
            "model": model, "tier": tier, "resolved": resolved,
            "latency_s_per_q": round(dt / len(eval_examples), 2),
            **rep["retrieval"], **rep["task"], **rep["grounding"], "abstention": ab,
        })
        print(f"  OK   {model:32s} -> {resolved}")

    # Grounding ablation: closed-book (no retrieval) accuracy for the reference model.
    if args.ablation_model:
        cb = closed_book_eval(eval_examples, args.ablation_model, workers=args.workers)
        rows.append({
            "model": f"{args.ablation_model} (closed-book)",
            "tier": "ablation",
            "resolved": "(no retrieval)",
            "accuracy": cb["accuracy"],
            "macro_f1": cb["macro_f1"],
        })
        print(f"  OK   closed-book {args.ablation_model} acc={cb['accuracy']:.2f}")
    conn.close()

    # --- markdown table artifact (ASCII-safe for Windows consoles) ---
    def cell(r: dict, key: str | None = None, sub: str | None = None) -> str:
        v = (r.get("abstention") or {}).get(sub) if sub else r.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "-"

    hdr = ("| model | tier | acc | macroF1 | faith | halluc | cit_acc | recall@6 "
           "| abst(ans) | abst(off) | s/q |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in rows:
        lat = r.get("latency_s_per_q")
        lines.append(
            f"| `{r['model']}` | {r['tier']} | {cell(r, 'accuracy')} | {cell(r, 'macro_f1')} | "
            f"{cell(r, 'faithfulness')} | {cell(r, 'hallucination_rate')} | {cell(r, 'citation_accuracy')} | "
            f"{cell(r, f'recall@{args.k}')} | {cell(r, sub='answerable')} | {cell(r, sub='off_topic')} | "
            f"{lat if isinstance(lat, (int, float)) else '-'} |"
        )
    table = "\n".join(lines)
    note = "abst(ans): lower is better; abst(off): higher is better."

    # Write artifacts BEFORE printing, so a console-encoding hiccup can't lose results.
    Path("eval_results").mkdir(exist_ok=True)
    meta = {"judge": judge, "n": len(eval_examples), "corpus_passages": len(corpus), "k": args.k, "abstain": args.abstain}
    Path("eval_results/comparison.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8"
    )
    Path("eval_results/comparison.md").write_text(
        f"# Frontier vs. open: biomedical RAG\n\n"
        f"judge={judge}, n={len(eval_examples)}, corpus={len(corpus)} passages, "
        f"hybrid retrieval (BM25+BGE/RRF), k={args.k}, abstain<{args.abstain}. {note}\n\n"
        f"Resolved providers: " + "; ".join(f"`{r['model']}` -> `{r['resolved']}`" for r in rows) + "\n\n"
        + table + "\n",
        encoding="utf-8",
    )
    print("\n" + table)
    print(note)
    print("wrote eval_results/comparison.json + comparison.md")


if __name__ == "__main__":
    main()
