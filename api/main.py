"""FastAPI service backing the web demo.

Loads the corpus from pgvector at startup, builds the hybrid retriever, and serves
grounded, cited answers (with abstention) from any plan-included model.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn api.main:app --reload
Prereq: docker compose up -d, and a populated corpus (e.g. scripts/run_eval.py once).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from biomed_rag.agent.rag import answer
from biomed_rag.retrieval.hybrid import HybridIndex
from biomed_rag.retrieval.pgvector_store import connect, load_passages

ABSTAIN_THRESHOLD = 0.4
DEFAULT_MODEL = "gpt-4.1-mini"
MODELS = ["gpt-4.1-mini", "claude-haiku-4-5", "deepseek-v3.2", "qwen-flash"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = connect()
    passages = load_passages(conn)
    state["conn"] = conn
    state["n_passages"] = len(passages)
    state["index"] = HybridIndex(passages, conn) if passages else None
    yield
    conn.close()


app = FastAPI(title="Biomedical RAG Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    question: str
    model: str | None = None
    top_k: int | None = None


class Citation(BaseModel):
    n: int
    doc_id: str
    passage: str


class AskResponse(BaseModel):
    answer: str
    decision: str
    citations: list[Citation]
    abstained: bool
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_passages": state.get("n_passages", 0)}


@app.get("/models")
def models() -> dict:
    return {"models": MODELS, "default": DEFAULT_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    index = state.get("index")
    if index is None:
        raise HTTPException(503, "Corpus is empty — run scripts/run_eval.py once to ingest.")
    model = req.model or DEFAULT_MODEL
    res = answer(
        req.question, index, model=model, k=req.top_k or 6, abstain_threshold=ABSTAIN_THRESHOLD
    )
    return AskResponse(
        answer=res.answer,
        decision=res.decision,
        citations=[Citation(n=c["n"], doc_id=c["doc_id"], passage=c["text"]) for c in res.citations],
        abstained=res.abstained,
        model=model,
    )
