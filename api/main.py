"""FastAPI service backing the web demo.

Milestone 1 stub: exposes a health check and a stubbed /ask endpoint so the
contract is visible. Real retrieval+synthesis wiring lands in Milestone 2+.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Biomedical RAG Agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    source_id: str
    passage: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # TODO(Milestone 2): wire biomed_rag.agent — retrieve, synthesize, abstain.
    raise NotImplementedError("Agent pipeline not yet implemented (Milestone 2).")
