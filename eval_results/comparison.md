# Frontier vs. open: biomedical RAG

judge=gpt-4.1, n=20, corpus=1028 passages, hybrid retrieval (BM25+BGE/RRF), k=6, abstain<0.5. abst(ans): lower is better; abst(off): higher is better.

Resolved providers: `gpt-4.1-mini` -> `openai/gpt-4.1-mini`; `claude-haiku-4-5` -> `vertex-anthropic/claude-haiku-4-5`; `deepseek-v3.2` -> `deepseek/deepseek-v3.2`; `qwen-flash` -> `alibaba/qwen-flash:singapore`

| model | tier | acc | macroF1 | faith | halluc | cit_acc | recall@6 | abst(ans) | abst(off) | s/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-4.1-mini` | frontier | 0.55 | 0.35 | 0.99 | 0.01 | 0.91 | 0.82 | 0.20 | 0.80 | 4.49 |
| `claude-haiku-4-5` | frontier | 0.55 | 0.41 | 0.99 | 0.01 | 1.00 | 0.82 | 0.25 | 1.00 | 3.81 |
| `deepseek-v3.2` | open | 0.55 | 0.35 | 1.00 | 0.00 | 0.90 | 0.82 | 0.25 | 0.80 | 4.01 |
| `qwen-flash` | open | 0.60 | 0.35 | 0.92 | 0.08 | 0.97 | 0.82 | 0.10 | 0.80 | 4.21 |
