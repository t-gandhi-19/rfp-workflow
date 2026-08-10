# Answer quality rubric

Scored 1–5 on each dimension by `judge-model` during eval runs, and by a human
on a fixed spot-check sample. Both use this same file, which is what makes
judge-vs-human agreement a meaningful number: the harness reports that agreement
so the judge is measured before it is trusted (build prompt §20).

The judge alias resolves to a different model family than the drafter, so a
model is never grading its own prose.

> Phase 1 defines the dimensions and the anchors. Phase 3 wires this into the
> harness alongside the human spot-check sample.

## Dimensions

### Relevance — does it answer the question actually asked?
| Score | Anchor |
|---|---|
| 5 | Answers the question directly and completely, including every sub-part. |
| 3 | Answers the main thrust but misses a sub-part or drifts into adjacent material. |
| 1 | Describes related capability without answering the question. |

### Specificity — is it concrete enough to be checkable?
| Score | Anchor |
|---|---|
| 5 | Names methods, tools, and artifacts; claims are attributable to a source. |
| 3 | Mixes concrete detail with generic capability statements. |
| 1 | Generic marketing prose that would fit any vendor and any RFP. |

### Directness — does it lead with the answer?
| Score | Anchor |
|---|---|
| 5 | States the answer first, then supports it. |
| 3 | Preamble before the answer, but the answer is present and clear. |
| 1 | The reader must infer the answer from surrounding narrative. |

### Tone — does it read as a professional RFP response?
| Score | Anchor |
|---|---|
| 5 | Measured, factual, no overclaiming; consistent with the response template. |
| 3 | Mostly appropriate with occasional salesy or informal phrasing. |
| 1 | Overclaims, hedges evasively, or breaks register. |

## Scoring rules

- Score only what is written. Do not credit an answer for what it could have said.
- An escalated question is **not** scored on quality; it is scored on whether the
  escalation was correct.
- Guardrail violations are not a quality dimension. They are separate, hard,
  deterministic checks, and a violation fails the answer regardless of its score.
