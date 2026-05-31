# Grounded Biomedical RAG: how much do retrieval and model choice actually buy you?

A small, reproducible study behind the [Biomedical RAG Agent](README.md). I built a
clinician/researcher-facing retrieval-augmented agent over open-access biomedical
literature and measured the things that usually go unmeasured in "I built a RAG"
projects: retrieval quality, answer **faithfulness**, hallucination rate, citation
accuracy, and abstention — across frontier and open-weight models, on one harness.

## TL;DR

- **Retrieval is doing real work.** Closed-book (no retrieval) scores **0.45** task
  accuracy; the same model with hybrid RAG scores **0.63** — a **+18-point** lift.
- **Open models are competitive with frontier**, and *lead* on grounding:
  DeepSeek-V3.2 has the best faithfulness (0.98), lowest hallucination (0.02), and best
  citation accuracy (0.97) of the four models tested, at frontier-level accuracy.
- **Models differ in caution.** Qwen-flash answers aggressively (worst faithfulness,
  rarely abstains); Claude-haiku and DeepSeek abstain conservatively.
- These are baseline numbers (N=150, single judge) — directional, not a significance
  claim. The point is the *measurement harness*, which makes any of this falsifiable.

## Problem

LLMs hallucinate, and in a biomedical setting that is unacceptable. Retrieval-augmented
generation is the standard mitigation, but most portfolio RAG projects stop at "it
answers questions." The interesting questions are: *does retrieval actually reduce
hallucination, by how much, and does an expensive frontier model beat a cheap open one
on grounded answering?* This project is built to answer those with numbers.

Framing: this is an **evidence-retrieval tool for clinicians/researchers**, not
patient-facing medical advice. It cites sources and abstains when evidence is weak.

## System

```
ingest → chunk → embed → pgvector + BM25
      → hybrid retriever (Reciprocal Rank Fusion)
      → model-agnostic synthesizer → grounded, cited answer (or abstain)
      → eval harness → retrieval / task / grounding / abstention metrics
```

- **Corpus:** PubMedQA labeled set (PQA-L). 400 questions' contexts pooled into a single
  1,363-passage corpus, so retrieval is non-trivial (a question's relevant passages sit
  among everyone else's).
- **Retrieval:** hybrid — dense (BAAI/bge-small-en-v1.5 via `fastembed`, ONNX, no torch)
  in **pgvector**, fused with lexical **BM25** by Reciprocal Rank Fusion.
- **Models:** all routed through **LLMGateway** (OpenAI-compatible) so swapping a model
  is one ID change. Frontier: `gpt-4.1-mini`, `claude-haiku-4-5`. Open: `deepseek-v3.2`,
  `qwen-flash`. Provenance (`provider/model:region`) is recorded per run.
- **Abstention:** retrieval-gated (decline if best dense similarity < 0.4) plus the
  model's own judgment.
- **Grounding judge:** a fixed stronger model (`gpt-4.1`) decomposes each answer into
  atomic claims, verifies each against the retrieved passages, and checks that cited
  passages support the answer.

## Setup

Judge `gpt-4.1`, hybrid retrieval, k=6, abstain threshold 0.4, N=150 evaluation
questions over the 1,363-passage corpus. Temperature 0 throughout. Selection is
deterministic. Raw results: [`eval_results/comparison.md`](eval_results/comparison.md).

## Results

### Retrieval (model-independent)

| recall@6 | MRR | nDCG@6 | hit@6 |
|---|---|---|---|
| 0.82 | 0.95 | 0.83 | 0.98 |

Identical across all generator models, as it should be (retrieval runs before the LLM) —
a sanity check that the harness isn't leaking model effects into retrieval.

### Grounding ablation — does retrieval help?

| config (gpt-4.1-mini) | accuracy | macro-F1 |
|---|---|---|
| closed-book (no retrieval) | 0.45 | 0.37 |
| **hybrid RAG** | **0.63** | **0.53** |

Retrieval lifts accuracy **+18 points** and macro-F1 **+16**. This is the headline: the
system's core mechanism measurably works.

### Frontier vs. open

| model | tier | acc | macro-F1 | faithfulness | hallucination | citation-acc | abst(off-topic) |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | frontier | 0.63 | 0.53 | 0.94 | 0.06 | 0.84 | 0.80 |
| claude-haiku-4-5 | frontier | 0.58 | 0.53 | 0.96 | 0.04 | 0.87 | 1.00 |
| deepseek-v3.2 | open | 0.60 | 0.51 | **0.98** | **0.02** | **0.97** | 1.00 |
| qwen-flash | open | 0.59 | 0.50 | 0.91 | 0.09 | 0.93 | 0.60 |

**Open models hold their own.** DeepSeek-V3.2 matches frontier accuracy and leads every
grounding metric. The assumption that you trade away quality by going open does not hold
on this task.

### Abstention

Claude-haiku and DeepSeek abstain conservatively (≈0.25 on answerable, 1.00 on off-topic);
Qwen-flash rarely abstains (0.03 / 0.60) and pays with the worst faithfulness. A threshold
sweep showed the dense-similarity gate is a weak discriminator (answerable and off-topic
both cluster near 0.5 cosine) — so abstention is largely **model-driven**, not gated.

## Limitations (honest)

- **Scale:** N=150, single judge run. Differences between models are within noise;
  treat as directional.
- **Corpus:** only 400 questions' contexts. Many real questions abstain because the topic
  isn't present (correct behavior, but limits the demo), and off-topic queries can still
  hit spurious lexical matches (e.g. a PubMed abstract mentioning "Paris").
- **Judge bias:** a single `gpt-4.1` judge; same provider family as one generator. A
  multi-judge or cross-family panel would harden the grounding numbers.
- **Model set:** `kimi-k2` and large `qwen3` were excluded — gateway routing / plan limits.
- **Quantization/provider:** LLMGateway auto-routes; provenance is captured but precision
  isn't explicitly pinned.

## Reproduce

```bash
docker compose up -d                                                   # Postgres + pgvector
pip install -e ".[dev]"                                                # or the runtime deps
PYTHONPATH=src python scripts/run_compare.py --corpus-n 400 --n 150     # comparison + ablation
PYTHONPATH=src python scripts/run_eval.py --n 40                        # single-model full metrics
PYTHONPATH=src python scripts/tune_abstain.py                           # abstention sweep (no LLM)
```

Set `LLMGATEWAY_API_KEY` in `.env`. Results land in `eval_results/`.

## What I'd do next

Scale to PubMedQA's full 500-question test split for significance; add a cross-family
judge panel; expand the corpus with a PMC open-access slice; pin provider/precision for
the open models.
