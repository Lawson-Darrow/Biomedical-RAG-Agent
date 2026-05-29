"""The RAG agent: retrieve → synthesize a grounded, cited answer → abstain.

Kept model-agnostic on purpose: the prompt asks for JSON only (no provider-specific
response_format), so the exact same call works across every model we compare.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from biomed_rag.models.client import chat
from biomed_rag.retrieval.store import InMemoryIndex

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


def _safe_json(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def answer(question: str, index: InMemoryIndex, model: str, k: int = 8) -> RagAnswer:
    hits = index.search(question, k=k)
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
    data = _safe_json(raw)

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
