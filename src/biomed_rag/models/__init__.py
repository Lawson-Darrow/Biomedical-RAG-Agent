"""Model-agnostic LLM interface so the same pipeline runs frontier and open-weight models.

Milestone 1 stub. Implementation lands in Milestone 2 (one model) / Milestone 5 (swap):
    - a thin `LLMClient` protocol: `.generate(messages, **opts) -> str`
    - adapters: Anthropic, OpenAI, and an OpenAI-compatible open-weights endpoint (Qwen2.5-7B)
    - the comparison axis (frontier vs open) is driven entirely through this interface
"""
