"""Probe LLMGateway for the comparison set: which IDs are valid, what each
request actually resolves to (provider/precision provenance for reproducibility),
latency, and token usage. Never prints the key.

Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/probe_models.py
"""

from __future__ import annotations

import time

from openai import OpenAI

from biomed_rag.config import settings

client = OpenAI(api_key=settings.llmgateway_api_key, base_url=settings.llmgateway_base_url)

ids = {m.id for m in client.models.list().data}

# Help pick a clean general-purpose Qwen chat model (exclude image/coder/audio/vision).
_skip = ("image", "coder", "embed", "tts", "asr", "audio", "-vl", "omni", "rerank")
qwen_chat = sorted(i for i in ids if "qwen" in i.lower() and not any(s in i.lower() for s in _skip))
print("qwen chat candidates:", qwen_chat[:20])

CANDIDATES = [
    "gpt-4.1-mini",
    "claude-3-7-sonnet",
    "claude-haiku-4-5",
    "deepseek-v3.2",
    "kimi-k2",
    "qwen-max",
    "qwen-plus",
    "qwen-flash",
]

print("\nprovenance probe:")
for m in CANDIDATES:
    if m not in ids:
        print(f"  {m:22s} NOT AVAILABLE")
        continue
    t = time.time()
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            temperature=0,
        )
        dt = time.time() - t
        d = r.model_dump()
        prov = d.get("provider") or d.get("served_by") or d.get("system_fingerprint") or "?"
        u = d.get("usage") or {}
        print(
            f"  {m:22s} resolved={r.model!r:28s} provider={prov} "
            f"latency={dt:.2f}s tokens={u.get('total_tokens')} out={r.choices[0].message.content!r}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {m:22s} ERROR {type(e).__name__}: {e}")
