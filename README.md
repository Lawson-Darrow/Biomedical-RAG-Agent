# Biomedical RAG Agent

A clinician/researcher-facing agent that answers biomedical questions by retrieving
from open-access literature (PubMed/PMC) and returning **cited, grounded answers** —
benchmarked across **frontier vs. open-weight models** with a rigorous faithfulness eval.

> **Framing:** this is an *evidence-retrieval* tool for clinicians and researchers, **not**
> patient-facing medical advice. Answers are grounded in retrieved literature and the agent
> abstains when the evidence is insufficient.

## Why

It pairs a clickable demo (the agent) with a paper-style evaluation study (the eval harness).
The differentiator is not "a RAG chatbot" — it's the measurement: retrieval quality, answer
faithfulness, hallucination rate, citation accuracy, and abstention correctness, reported
across frontier and open-weight models on public biomedical benchmarks.

## Architecture

```
ingest → chunk → embed → vector store (pgvector) + BM25
      → hybrid retriever (optional reranker)
      → model-agnostic synthesizer → cited answer (+ abstain)
      → eval harness runs the pipeline over benchmarks → metrics report
```

| Layer        | Package                  | Role                                                       |
|--------------|--------------------------|------------------------------------------------------------|
| Ingestion    | `biomed_rag.ingest`      | Acquire corpus (PubMedQA contexts + PMC-OA slice), chunk   |
| Retrieval    | `biomed_rag.retrieval`   | Dense (fastembed/BGE) + BM25 hybrid (RRF) over pgvector     |
| Models       | `biomed_rag.models`      | Model-agnostic LLM interface — frontier + open, via LLMGateway |
| Agent        | `biomed_rag.agent`       | Retrieve → synthesize cited answer → abstain               |
| Eval         | `biomed_rag.eval`        | Metric suite + benchmark runners + report                  |
| API          | `api/`                   | FastAPI service backing the web demo                       |
| Frontend     | `frontend/`              | React + Vite demo (Milestone 6)                            |

## Eval metrics

- **Retrieval:** Recall@k, nDCG/MRR
- **Grounding/faithfulness:** claim-level support vs. retrieved context
- **Hallucination rate:** unsupported claims per answer
- **Citation accuracy:** cited passage actually supports the claim
- **Abstention correctness:** declines when evidence is weak — and is that right
- **Task accuracy:** PubMedQA (yes/no/maybe); BioASQ stretch set
- **Comparison axis:** all of the above × {frontier, open} + cost & latency

## Status

Milestone 5 — frontier vs. open comparison + grounding ablation. Full results in
[`eval_results/comparison.md`](eval_results/comparison.md). Judge `gpt-4.1`, hybrid
retrieval, **150 questions / 1363-passage corpus**:

| model | tier | acc | macroF1 | faith | halluc | cit_acc | recall@6 | abst(off)↑ |
|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | frontier | 0.63 | 0.53 | 0.94 | 0.06 | 0.84 | 0.82 | 0.80 |
| claude-haiku-4-5 | frontier | 0.58 | 0.53 | 0.96 | 0.04 | 0.87 | 0.82 | 1.00 |
| deepseek-v3.2 | open | 0.60 | 0.51 | **0.98** | **0.02** | **0.97** | 0.82 | 1.00 |
| qwen-flash | open | 0.59 | 0.50 | 0.91 | 0.09 | 0.93 | 0.82 | 0.60 |
| gpt-4.1-mini (closed-book) | ablation | 0.45 | 0.37 | – | – | – | – | – |

Findings:
1. **Retrieval works:** closed-book → hybrid RAG lifts accuracy **0.45 → 0.63 (+18 pts)**.
2. **Open ≥ frontier on grounding:** DeepSeek-V3.2 leads faithfulness (0.98), hallucination (0.02), and citation accuracy (0.97), matching frontier accuracy.
3. **Caution differs:** Qwen-flash answers aggressively (worst faithfulness); Claude/DeepSeek abstain conservatively. `recall@6` is identical across models (harness sanity check).

```bash
docker compose up -d                                                  # Postgres + pgvector
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_eval.py --n 40     # single model, full metrics
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_compare.py --n 150 # frontier vs open + ablation
PYTHONPATH=src .venv/Scripts/python.exe scripts/tune_abstain.py        # abstention threshold sweep (free)
```

(`run_m2.py` = BM25-only skeleton; `run_m3.py` = hybrid retrieval diagnostics.) See [SPEC.md](SPEC.md) for milestones.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"
cp .env.example .env                               # then fill in keys
```

## License

TBD (add before going public).
