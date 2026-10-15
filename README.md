# In RAG We Trust?

NLP final project (P5), Natural Language Processing, Prof. Alfio Ferrara,
Università degli Studi di Milano. Author: Angela Aldaoud.

A measurement study of how open instruction-tuned LLMs (Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3) answer questions when part of
the retrieved evidence is deliberately false, and whether reliability-oriented
prompts make them more reliable.

**The report is [`report.pdf`](report.pdf) in this folder.**

## Repository layout

| Path | Contents |
|---|---|
| `report.pdf` | Final report (submitted version) |
| `paper/` | LaTeX source of the report |
| `src/inragwetrust/` | Python package: data, poisoning, retrieval, models, evaluation, analysis |
| `tests/` | Unit tests (`uv run pytest`) |
| `configs/` | Experiment configuration and condition table |
| `data/processed/` | Question set, paraphrases and poisoned passages derived from HotpotQA |
| `notebooks/` | Colab notebooks used for the GPU stages and the analysis |
| `results/` | Frozen result tables, analysis CSVs, paper tables, figures and macros |
| `docs/` | Project notes, decision log and AI-usage log (`docs/notes/main.tex`), dataset card, results summary |

Raw model outputs (`runs/`) are not in the repository (see below).

## Setup (local, CPU)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # creates .venv and installs the exact locked versions
uv run pytest      # runs the tests
```

GPU dependencies (Colab) are installed with the `gpu` extra.

## Reproduce

Every number, table and figure in report.pdf is regenerated from the frozen analysis outputs in results/ (no GPU needed). From the repository root:

    uv sync
    uv run pytest -q
    uv run python -m inragwetrust.analysis.numbers    # paper macros: results/analysis/numbers*.tex
    uv run python -m inragwetrust.analysis.tables     # results/analysis/tables/*.tex
    uv run python -m inragwetrust.analysis.figures    # results/figures/*.pdf and *.png
    cd paper && tectonic report.tex                   # paper/report.pdf

Recomputing the CSVs in results/analysis/ from scratch requires the raw model outputs, available from the author on request (integrity: SHA-256 recorded in docs/notes/main.tex).

## Data and license

This project uses HotpotQA (Yang et al., 2018, "HotpotQA: A Dataset for Diverse,
Explainable Multi-hop Question Answering", EMNLP 2018), released under
CC BY-SA 4.0. The files in `data/processed/` are derived from it and are shared
under the same license.

## AI usage

AI assistance is logged in `docs/notes/main.tex` (AI-Usage Log chapter) and
declared in the report's "Use of AI tools" statement.