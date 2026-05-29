"""Model-agnostic LLM access — every model (frontier + open) via LLMGateway.

See `client.py` for `get_client`, `chat`, and `embed`. Swapping models is an
ID change, which is what drives the frontier-vs-open comparison axis.
"""
