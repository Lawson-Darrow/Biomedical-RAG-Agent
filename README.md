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

Milestone 3 — hybrid retrieval working: BM25 + dense (fastembed/BGE) fused via RRF
over pgvector, with a retrieval-gated abstention threshold and inline citations.
Retrieval hit-rate@6 on 100 questions: **hybrid 100% > dense 99% > bm25 97%**.

```bash
docker compose up -d                                          # Postgres + pgvector
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_m3.py --n 100 --eval-n 15
```

(`scripts/run_m2.py` is the earlier BM25-only skeleton.) See [SPEC.md](SPEC.md) for milestones.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"
cp .env.example .env                               # then fill in keys
```

## License

TBD (add before going public).
