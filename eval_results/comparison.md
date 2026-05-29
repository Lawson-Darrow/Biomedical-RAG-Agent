# Frontier vs. open: biomedical RAG

judge=gpt-4.1, n=150, corpus=1363 passages, hybrid retrieval (BM25+BGE/RRF), k=6, abstain<0.4. abst(ans): lower is better; abst(off): higher is better.

Resolved providers: `gpt-4.1-mini` -> `openai/gpt-4.1-mini`; `claude-haiku-4-5` -> `anthropic/claude-haiku-4-5`; `deepseek-v3.2` -> `deepseek/deepseek-v3.2`; `qwen-flash` -> `alibaba/qwen-flash:singapore`; `gpt-4.1-mini (closed-book)` -> `(no retrieval)`

| model | tier | acc | macroF1 | faith | halluc | cit_acc | recall@6 | abst(ans) | abst(off) | s/q |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-4.1-mini` | frontier | 0.63 | 0.53 | 0.94 | 0.06 | 0.84 | 0.82 | 0.11 | 0.80 | 0.8 |
| `claude-haiku-4-5` | frontier | 0.58 | 0.53 | 0.96 | 0.04 | 0.87 | 0.82 | 0.24 | 1.00 | 0.66 |
| `deepseek-v3.2` | open | 0.60 | 0.51 | 0.98 | 0.02 | 0.97 | 0.82 | 0.25 | 1.00 | 0.79 |
| `qwen-flash` | open | 0.59 | 0.50 | 0.91 | 0.09 | 0.93 | 0.82 | 0.03 | 0.60 | 0.57 |
| `gpt-4.1-mini (closed-book)` | ablation | 0.45 | 0.37 | - | - | - | - | - | - | - |
