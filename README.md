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

Milestone 5 — frontier vs. open comparison on the shared harness. Full results in
[`eval_results/comparison.md`](eval_results/comparison.md). Judge `gpt-4.1`, hybrid
retrieval, 20 questions / 1028-passage corpus:

| model | tier | acc | faith | halluc | cit_acc | recall@6 | abst(off)↑ |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | frontier | 0.55 | 0.99 | 0.01 | 0.91 | 0.82 | 0.80 |
| claude-haiku-4-5 | frontier | 0.55 | 0.99 | 0.01 | 1.00 | 0.82 | 1.00 |
| deepseek-v3.2 | open | 0.55 | 1.00 | 0.00 | 0.90 | 0.82 | 0.80 |
| qwen-flash | open | 0.60 | 0.92 | 0.08 | 0.97 | 0.82 | 0.80 |

Finding: **open models are competitive with frontier** on grounded biomedical RAG
(within noise at N=20). `recall@6` is identical across models (retrieval is
model-independent — a harness sanity check). Provenance is captured per model.

```bash
docker compose up -d                                                # Postgres + pgvector
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_eval.py --n 40   # single model, full metrics
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_compare.py --n 20  # frontier vs open
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
