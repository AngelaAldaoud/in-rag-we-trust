# In RAG We Trust?

NLP final project (P5), Natural Language Processing, Prof. Alfio Ferrara,
Università degli Studi di Milano.

Measuring how retrieval-augmented generation (RAG) systems handle poisoned
and conflicting evidence: factual accuracy, conflict detection and hedging,
and the effect of reliability-oriented prompts.

## Status

Block A (foundations) complete: literature audit, research questions,
experimental design, compute plan, project layout.

## Repository layout


## Setup (local, CPU)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # creates .venv and installs the exact locked versions
uv run pytest      # runs the tests
```

GPU dependencies (Colab) are installed with the `gpu` extra.

## AI usage

AI assistance is logged in `docs/notes/` (AI-Usage Log chapter) and declared
in the report's disclaimer.
