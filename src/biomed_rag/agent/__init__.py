"""The RAG agent: retrieve → synthesize a cited answer → abstain when evidence is weak.

Milestone 1 stub. Implementation lands in Milestone 2-3:
    - retrieve passages via `biomed_rag.retrieval`
    - prompt the synthesizer (`biomed_rag.models`) for an answer with inline citations
    - abstain ("insufficient evidence") when retrieval support is below threshold
    - return a structured answer: text, citations, support scores, abstained flag
"""
