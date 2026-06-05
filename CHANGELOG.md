# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[SemVer](https://semver.org/) (pre-1.0: minor = breaking allowed).

## [Unreleased]

## [0.1.0] - 2026-06-05

Initial public release. Grounded biomedical RAG agent with a frontier-vs-open
evaluation harness.

### Added
- Hybrid retrieval: dense (fastembed/BGE) plus BM25 with reciprocal-rank fusion
  over pgvector.
- Model-agnostic synthesizer producing cited, grounded answers with an abstain
  state when evidence is weak; all model traffic routes through an
  OpenAI-compatible gateway (frontier and open-weight models).
- Evaluation harness: retrieval (Recall@k, nDCG, MRR), claim-level faithfulness,
  hallucination rate, citation accuracy, abstention correctness, and task
  accuracy on PubMedQA.
- Frontier-vs-open comparison study (150 questions, 1363-passage corpus) with a
  closed-book ablation; results in `eval_results/comparison.md`.
- FastAPI backend and React/Vite demo frontend.

### Known limitations
- Reported on a 150-question slice; broader benchmarks (BioASQ) are a stretch goal.
- Confidence/abstention thresholds are tuned on the eval set, not separately
  calibrated.
- Evidence-retrieval tool for research, not a patient-facing medical device.
