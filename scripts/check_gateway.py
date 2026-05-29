"""Connectivity + capability check for LLMGateway.

Reads .env directly (no package install needed). Lists what the configured
BYOK key exposes, then does a tiny live chat completion. Never prints the key.

Run: .venv/Scripts/python.exe scripts/check_gateway.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("LLMGATEWAY_API_KEY")
base_url = os.getenv("LLMGATEWAY_BASE_URL", "https://api.llmgateway.io/v1")

if not api_key:
    sys.exit("LLMGATEWAY_API_KEY not set in .env")

print(f"base_url: {base_url}")
client = OpenAI(api_key=api_key, base_url=base_url)

# 1) What models can we reach?
try:
    ids = sorted(m.id for m in client.models.list().data)
except Exception as e:  # noqa: BLE001
    sys.exit(f"models.list() failed: {type(e).__name__}: {e}")

print(f"total models reachable: {len(ids)}")
families = {
    "claude": "claude",
    "gpt": "gpt",
    "deepseek": "deepseek",
    "kimi/moonshot": "kimi",
    "qwen": "qwen",
    "embeddings": "embed",
}
found = {}
for label, needle in families.items():
    hits = [i for i in ids if needle in i.lower()]
    if label == "kimi/moonshot":
        hits = [i for i in ids if "kimi" in i.lower() or "moonshot" in i.lower()]
    found[label] = hits
    print(f"  {label:14s}: {len(hits):3d}  {hits[:6]}")

# 2) Tiny live completion against a cheap available chat model.
preferred = ["gpt-4o-mini", "gpt-5-mini", "claude-3-5-haiku", "qwen2.5-7b-instruct"]
chat_id = next((p for p in preferred if p in ids), None)
if chat_id is None:
    chat_id = next((i for i in ids if "embed" not in i.lower()), None)

if chat_id:
    try:
        r = client.chat.completions.create(
            model=chat_id,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            temperature=0,
        )
        print(f"\nlive completion via '{chat_id}': {r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        print(f"\nlive completion via '{chat_id}' FAILED: {type(e).__name__}: {e}")
else:
    print("\nno chat model id found to test")
