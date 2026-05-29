"""LLM-as-judge for grounding metrics.

One judge call per answer: decompose into atomic claims, verify each against the
retrieved passages, and check that cited passages actually support the answer.
Use a strong, fixed judge model (ideally a different family than the generator,
to limit self-grading bias).
"""

from __future__ import annotations

from dataclasses import dataclass

from biomed_rag.models.client import chat
from biomed_rag.util import safe_json

JUDGE_MODEL = "gpt-4.1"

_SYSTEM = (
    "You are a strict evaluator of biomedical answers. Judge claims ONLY against the "
    "provided passages — never outside knowledge. If a claim is not supported by the "
    "passages, mark it unsupported."
)

_PROMPT = """Question: {question}

Numbered passages the answer was given:
{context}

Answer under evaluation:
{answer}

The answer cited these passage numbers: {citations}

Return strict JSON:
1. Decompose the answer into atomic factual claims.
2. For each claim: is it supported by the passages? List supporting passage numbers.
3. For each cited passage number: does it actually support the answer's content?

Shape:
{{"claims": [{{"text": "...", "supported": true, "support": [1, 2]}}],
  "citations": [{{"n": 1, "supports": true}}]}}
JSON only."""


@dataclass
class Judgment:
    faithfulness: float  # supported claims / total claims
    hallucination_rate: float  # unsupported claims / total claims
    citation_accuracy: float  # valid citations / total cited (1.0 if no citations)
    n_claims: int
    raw: str


def judge_answer(
    question: str,
    context: str,
    answer_text: str,
    citations: list[int],
    model: str = JUDGE_MODEL,
) -> Judgment:
    raw = chat(
        model,
        [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _PROMPT.format(
                    question=question,
                    context=context,
                    answer=answer_text,
                    citations=citations or "(none)",
                ),
            },
        ],
        temperature=0,
    )
    data = safe_json(raw)

    claims = data.get("claims", []) or []
    n = len(claims)
    supported = sum(1 for c in claims if c.get("supported"))
    faithfulness = supported / n if n else 0.0
    hallucination = (n - supported) / n if n else 0.0

    cits = data.get("citations", []) or []
    citation_accuracy = (sum(1 for c in cits if c.get("supports")) / len(cits)) if cits else 1.0

    return Judgment(faithfulness, hallucination, citation_accuracy, n, raw)
