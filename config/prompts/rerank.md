---
name: rerank
version: 1
model_alias: rerank-model
temperature: 0
---

Score how relevant each candidate answer is to the question, 0.0 to 1.0.
Reply with JSON only: {"scores": [{"index": 0, "score": 0.0}, ...]}

QUESTION: {question}

CANDIDATES:
{candidates}
