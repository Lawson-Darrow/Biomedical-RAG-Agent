---
title: "Does retrieval actually make LLMs more honest? I measured it on biomedical questions."
date: 2026-05-31
tags: [llm, rag, nlp, evaluation, biomedical]
summary: I built a grounded biomedical RAG agent and measured what usually goes unmeasured — faithfulness, hallucination, citation accuracy — across frontier and open models. Retrieval bought +18 points of accuracy, and the cheap open model held its own.
---

# Does retrieval actually make LLMs more honest? I measured it on biomedical questions.

Most "I built a RAG app" projects stop at *it answers questions*. That's the boring part.
The interesting questions are the ones with numbers attached:

- Does retrieval actually reduce hallucination, and by how much?
- Does an expensive frontier model beat a cheap open-weight one at *grounded* answering?
- When the evidence isn't there, does the system know to shut up?

So I built a biomedical RAG agent designed to answer those — an evidence tool over
open-access literature (PubMedQA) that cites its sources and abstains when it can't
support an answer — and then I built the harness to measure it.

## The setup, briefly

Hybrid retrieval (dense BGE embeddings in **pgvector** + lexical **BM25**, fused with
Reciprocal Rank Fusion), a model-agnostic synthesizer behind an OpenAI-compatible
gateway so swapping models is one line, and an LLM-as-judge that breaks each answer into
atomic claims and checks every one against the retrieved passages. Everything open-source
where it could be — including the model gateway.

## What I found

**1. Retrieval is doing real work.** With no retrieval (closed-book), the model got
**45%** of PubMedQA questions right. With hybrid RAG, the *same model* got **63%** — an
**18-point** jump. That's the entire thesis of RAG, quantified on one chart.

**2. The cheap open model held its own — and was the most honest.** Across four models
(GPT-4.1-mini, Claude-haiku, DeepSeek-V3.2, Qwen-flash), the open **DeepSeek-V3.2**
matched frontier accuracy *and* led every grounding metric: highest faithfulness (0.98),
lowest hallucination (0.02), best citation accuracy (0.97). The "you give up quality
going open" assumption didn't survive contact with the data.

**3. Models have personalities.** Qwen-flash answers aggressively — it rarely abstains and
pays for it with the worst faithfulness. Claude-haiku and DeepSeek are cautious, declining
when evidence is thin. That behavioral axis matters more than a single accuracy number if
you're deploying one of these in a clinical context.

## The honest part

This is a baseline, not a paper: 150 questions, one judge, a small corpus. I'm reporting
it as directional. And the failures were instructive — ask it "what's the capital of
France?" and it occasionally finds a PubMed abstract that mentions *Paris* and answers
from it. That's a real lesson about how lexical retrieval can manufacture false
confidence, and exactly why the faithfulness metric exists.

## Why I built it this way

The point was never the chatbot — it was the **measurement**. Anyone can wire an LLM to a
vector store. The thing worth showing is a harness that makes "is this answer grounded?"
a number you can move, regress against, and compare models on. That's the difference
between a demo and an experiment.

Code, full results, and the live demo: [github.com/Lawson-Darrow/Biomedical-RAG-Agent](https://github.com/Lawson-Darrow/Biomedical-RAG-Agent).
