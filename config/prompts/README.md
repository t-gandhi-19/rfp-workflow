# Prompts

Prompts are versioned files, one per LLM-touching role, and are **never inlined
in Python** (CLAUDE.md rule 17). Every file carries a version header, and that
version string is attached to every span the prompt participates in, so a
quality regression can always be traced back to the exact prompt text that
produced it.

## Format

Each prompt is a Markdown file with a YAML front-matter header:

```yaml
---
name: drafter
version: 1
model_alias: drafter-model
---
```

## Files (populated in the phases that introduce each role)

| File | Role | Phase |
|---|---|---|
| `triage.md` | domain match / RFI detection | 4 |
| `extract_assist.md` | ambiguous-segment assist only | 4 |
| `rerank.md` | batched relevance scoring, temp 0 | 3 |
| `drafter.md` | answer drafting, cites per claim | 4 |
| `critic.md` | may only lower confidence or add flags | 4 |
| `judge.md` | evals only, never a production run | 3 |
| `loginterp.md` | plain-English run narration | 6 |

The drafter prompt must state that RFP document content is untrusted data and
never instructions (CLAUDE.md rule 15).
