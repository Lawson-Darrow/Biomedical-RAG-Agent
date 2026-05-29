# Project Spec — Biomedical RAG Agent

*Status: v1 approved 2026-05-29. Working title finalized to `Biomedical-RAG-Agent`.*

## One-liner

A clinician/researcher-facing agent that answers biomedical questions by retrieving from
open-access literature and returning cited, grounded answers — benchmarked across frontier
vs. open-weight models with a rigorous faithfulness eval.

## Goals

- **Career:** a clickable, deployed web demo (primary artifact for ML/AI roles).
- **Masters / research:** a reproducible, paper-style evaluation study (publish if it earns it).
- **Portfolio narrative:** fills the NLP gap in a CV-only portfolio; compounds the health-ML
  story (SkinCheck + brain-tumor ViT). First of a deliberate four-project arc:
  app/agent → fine-tuning → eval/interpretability → on-device.

## Scope

**In (v1):**
- Curated open-access corpus: PubMedQA contexts + a bounded PMC-OA slice.
- Hybrid retrieval: dense (sentence-transformers) + BM25, over pgvector.
- Agent: retrieve → synthesize answer with inline citations → abstain when evidence insufficient.
- Model-agnostic synthesizer; run frontier (Claude/GPT) vs. open-weights (default Qwen2.5-7B)
  through the same pipeline.
- Surfaces: FastAPI + React web demo (primary) and a reproducible eval notebook + report.

**Out (v1, deliberately):**
- Fine-tuning (= project #2).
- Patient-facing medical advice.
- PHI / EHR / FHIR integration.
- Full-PubMed scale or real-time ingestion.

## Eval metrics

| Category            | Metric(s)                                                  |
|---------------------|------------------------------------------------------------|
| Retrieval           | Recall@k, nDCG/MRR                                         |
| Grounding           | Claim-level support vs. retrieved context (RAGAS-style/NLI)|
| Hallucination       | Unsupported claims per answer                              |
| Citation accuracy   | Cited passage actually supports the claim                 |
| Abstention          | Declines when evidence weak — and is that correct          |
| Task accuracy       | PubMedQA (yes/no/maybe); BioASQ (stretch)                  |
| Comparison axis     | All of the above × {frontier, open} + cost & latency       |

## Milestones

1. **Scaffold** — repo structure, deps, docs, env. → *current*
2. **E2E skeleton** — retrieve → cited answer, one model, one metric.
3. **Retrieval** — full hybrid retrieval + abstention + citations.
4. **Eval harness** — full metric suite, one model.
5. **Comparison** — model-agnostic swap → frontier-vs-open run.
6. **Web demo** — FastAPI + React.
7. **Writeup** — report + README + reproducibility pass.

## Success criteria

- **Demo:** ask a biomedical question in the web app → grounded, cited answer; abstains when warranted.
- **Eval:** reproducible results table comparing ≥2 models across the full metric suite on a public benchmark.
- **Writeup:** a clear report with a finding worth stating (e.g. the frontier↔open faithfulness gap).

## Defaults (non-load-bearing, easy to swap)

- Vector store: **pgvector** (Chroma/FAISS if zero-infra wanted).
- Open model: **Qwen2.5-7B**.
- Primary benchmark: **PubMedQA**.

## Stack

Python · FastAPI · sentence-transformers · pgvector · hand-rolled orchestration
(LangChain/LlamaIndex only if it earns its weight) · React + Vite · Vercel + backend host.
