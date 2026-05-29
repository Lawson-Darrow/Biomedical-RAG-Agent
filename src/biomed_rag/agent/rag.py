"""The RAG agent: retrieve → synthesize a grounded, cited answer → abstain.

Kept model-agnostic on purpose: the prompt asks for JSON only (no provider-specific
response_format), so the exact same call works across every model we compare.
"""

from __future__ import annotations

from dataclasses import dataclass

from biomed_rag.models.client import chat
from biomed_rag.retrieval.base import Hit, Retriever
from biomed_rag.util import safe_json

SYSTEM = (
    "You are a biomedical evidence assistant for clinicians and researchers. "
    "Answer ONLY from the numbered context passages. Cite the passages you use by "
    "their number, like [1] or [2]. If the passages lack enough evidence to answer, "
    "set \"abstain\" to true and say what is missing. Do not use outside knowledge."
)

PROMPT = """Question: {question}

Context passages:
{context}

Respond in strict JSON with these keys:
  "answer": grounded answer string with inline [n] citations,
  "decision": one of "yes", "no", "maybe",
  "citations": list of integers (the passage numbers you relied on),
  "abstain": boolean (true if the evidence is insufficient).
Output JSON only — no prose outside the JSON object."""


@dataclass
class RagAnswer:
    answer: str
    decision: str
    citations: list[dict]  # {n, doc_id, text}
    abstained: bool
    raw: str


def synthesize(
    question: str,
    hits: list[Hit],
    model: str,
    abstain_threshold: float | None = None,
) -> RagAnswer:
    """Turn retrieved passages into a grounded, cited answer (no retrieval here).

    Retrieval-gated abstention: if the best dense match is too weak, decline
    without spending a model call. Only fires when dense signal is present
    (BM25-only hits carry dense_sim=0, so pass abstain_threshold=None there).
    """
    if abstain_threshold is not None and hits:
        best_sim = max(h.dense_sim for h in hits)
        if best_sim < abstain_threshold:
            return RagAnswer(
                answer="Insufficient evidence in the retrieved literature to answer.",
                decision="maybe",
                citations=[],
                abstained=True,
                raw=f"(gated: best dense_sim={best_sim:.3f} < {abstain_threshold})",
            )

    context = "\n\n".join(
        f"[{n}] (doc {h.passage.doc_id}) {h.passage.text}" for n, h in enumerate(hits, 1)
    )
    raw = chat(
        model,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(question=question, context=context)},
        ],
        temperature=0,
    )
    data = safe_json(raw)

    cited = []
    for n in data.get("citations", []):
        if isinstance(n, int) and 1 <= n <= len(hits):
            p = hits[n - 1].passage
            cited.append({"n": n, "doc_id": p.doc_id, "text": p.text})

    return RagAnswer(
        answer=data.get("answer", ""),
        decision=str(data.get("decision", "")).lower().strip(),
        citations=cited,
        abstained=bool(data.get("abstain", False)),
        raw=raw,
    )


def answer(
    question: str,
    index: Retriever,
    model: str,
    k: int = 8,
    abstain_threshold: float | None = None,
) -> RagAnswer:
    """Convenience: retrieve top-k then synthesize."""
    return synthesize(question, index.search(question, k=k), model, abstain_threshold)


_CB_SYSTEM = (
    "You are a biomedical expert. Answer the question from your own knowledge. "
    "Do not claim sources you were not given."
)
_CB_PROMPT = """Question: {question}

Respond in strict JSON:
  "answer": a concise answer,
  "decision": one of "yes", "no", "maybe".
JSON only."""


def answer_closed_book(question: str, model: str) -> RagAnswer:
    """No retrieval: the model answers from parametric knowledge. The grounding
    ablation baseline — quantifies what retrieval buys over closed-book."""
    raw = chat(
        model,
        [
            {"role": "system", "content": _CB_SYSTEM},
            {"role": "user", "content": _CB_PROMPT.format(question=question)},
        ],
        temperature=0,
    )
    data = safe_json(raw)
    return RagAnswer(
        answer=data.get("answer", ""),
        decision=str(data.get("decision", "")).lower().strip(),
        citations=[],
        abstained=False,
        raw=raw,
    )
