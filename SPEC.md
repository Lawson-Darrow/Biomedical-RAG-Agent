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
- Hybrid retrieval: dense (fastembed/BGE — ONNX, OSS, no torch) + BM25, fused via RRF, over pgvector.
- Agent: retrieve → synthesize answer with inline citations → abstain when evidence insufficient.
- Model-agnostic synthesizer; run frontier (Claude, GPT) vs. an open spectrum
  (DeepSeek, Kimi/Moonshot, Qwen/Alibaba) through the same pipeline, all routed via
  LLMGateway (OSS, OpenAI-compatible, BYOK).
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
3. **Retrieval** — full hybrid retrieval + abstention + citations. ✅ done
   (fastembed/BGE dense + BM25 fused via RRF over pgvector; retrieval-gated abstention;
   hit-rate@6: hybrid 100% > dense 99% > bm25 97% on 100q/330 passages).
4. **Eval harness** — full metric suite, one model. ✅ done
   (retrieval: recall/mrr/ndcg/hit@k · task: accuracy/macro-F1 · grounding via
   LLM-judge: faithfulness/hallucination/citation-acc · abstention both directions;
   reproducible JSON in `eval_results/`).
5. **Comparison** — model-agnostic swap → frontier-vs-open run. ✅ done
   (`scripts/run_compare.py`; results in `eval_results/comparison.md`).
   **Pinning:** LLMGateway auto-routes but `response.model` returns full provenance
   (`provider/model:region`, e.g. `alibaba/deepseek-v3.2:cn-beijing`), captured per
   run — explicit pinning is a follow-up if needed. Plan limits: `qwen-max`/big
   `qwen3` are plan-gated and `kimi-k2` mis-routes upstream, so both are excluded
   (logged, not silently dropped).
6. **Web demo** — FastAPI + React.
7. **Writeup** — report + README + reproducibility pass.

## Success criteria

- **Demo:** ask a biomedical question in the web app → grounded, cited answer; abstains when warranted.
- **Eval:** reproducible results table comparing ≥2 models across the full metric suite on a public benchmark.
- **Writeup:** a clear report with a finding worth stating (e.g. the frontier↔open faithfulness gap).

## Defaults (non-load-bearing, easy to swap)

- Model gateway: **LLMGateway** (OSS, OpenAI-compatible, BYOK).
- Vector store: **pgvector** (Chroma/FAISS if zero-infra wanted).
- Open arm: **DeepSeek + Kimi (Moonshot) + Qwen (Alibaba)**; exact IDs pinned from the catalog.
- Primary benchmark: **PubMedQA**.

## Tooling principle

Prefer OSS tools wherever a viable option exists (e.g. LLMGateway over a closed router,
pgvector over a proprietary store). Lower lock-in, better reproducibility, and a possible
upstream-contribution angle.

## Stack

Python · FastAPI · fastembed (BGE, ONNX) · rank-bm25 · pgvector (Docker) · LLMGateway ·
hand-rolled orchestration (LangChain/LlamaIndex only if it earns its weight) ·
React + Vite · Vercel + backend host.
