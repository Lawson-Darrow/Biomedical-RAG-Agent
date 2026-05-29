"""Load the PubMedQA labeled set (PQA-L) and build a passage corpus.

We pull the official labeled JSON straight from the PubMedQA repo (no `datasets`
dependency for the skeleton). Each record carries a question, a few context
passages, and an expert `final_decision` (yes/no/maybe) we use as ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

PQAL_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
CACHE = Path("data") / "pubmedqa_labeled.json"


@dataclass
class Passage:
    doc_id: str  # source PMID
    idx: int  # passage index within the doc
    text: str


@dataclass
class Example:
    qid: str  # source PMID
    question: str
    contexts: list[str]
    final_decision: str  # yes | no | maybe
    long_answer: str


def _raw() -> dict:
    CACHE.parent.mkdir(exist_ok=True)
    if not CACHE.exists():
        resp = httpx.get(PQAL_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        CACHE.write_text(resp.text, encoding="utf-8")
    return json.loads(CACHE.read_text(encoding="utf-8"))


def load_examples(limit: int | None = None) -> list[Example]:
    out: list[Example] = []
    for pmid, rec in _raw().items():
        out.append(
            Example(
                qid=pmid,
                question=rec["QUESTION"],
                contexts=rec.get("CONTEXTS") or [],
                final_decision=(rec.get("final_decision") or "").lower().strip(),
                long_answer=rec.get("LONG_ANSWER", ""),
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def build_corpus(examples: list[Example]) -> list[Passage]:
    """Pool every context passage into one retrieval corpus (retrieval is then
    non-trivial: the right passages for a question sit among everyone else's)."""
    return [
        Passage(doc_id=ex.qid, idx=i, text=text)
        for ex in examples
        for i, text in enumerate(ex.contexts)
    ]
